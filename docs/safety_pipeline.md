# Safety Pipeline for Discord Commands, Agent Outputs, and Deity-Mode Controls

This document defines a **defense-in-depth safety pipeline** for three traffic classes:

1. inbound user commands/messages,
2. agent-generated outputs,
3. privileged "deity mode" actions.

It is wired to the current implementation surfaces in:

- `src/interfaces/discord_moderation.py`,
- command parsing/execution paths in `src/interfaces/discord_bot.py`,
- simulation command handlers in `src/sim/simulation.py`.

The intent is to establish a minimum required control baseline **before enabling advanced control features** (for example expanded world-state mutation, bulk moderation, or autonomous privileged actions).

---

## 1) Threat model and safety goals

### Primary threats

- Prompt/command injection via Discord messages.
- Abuse of high-impact controls (`spawn`, `kill_agent`, `pause_all`, `inject_event`, etc.).
- Agent output that is abusive, unsafe, or leaks sensitive operational details.
- Privilege escalation through misconfigured role policy or permissive OPA rules.
- Silent failures (actions occur without reliable audit trails).

### Safety goals

- **Authenticate and authorize** every mutating action.
- **Constrain rate and blast radius** for high-impact commands.
- **Log all privileged decisions and outcomes** with enough context for replay/forensics.
- **Fail closed** for unavailable policy backends on privileged paths.
- **Keep policy explicit** (Discord role checks + OPA checks + command-scoped rules).

---

## 2) End-to-end pipeline overview

### A. Inbound user commands/messages

1. Receive Discord message/interaction.
2. Apply base content policy and OPA gate.
3. Parse routing/intent (`/broadcast`, `/dm`, mention routing).
4. Resolve target agent and permission tier.
5. Enforce command-specific rate limits and cooldowns.
6. Validate payload schema and semantic constraints.
7. Emit simulation event with immutable audit envelope.
8. Execute in simulation with post-condition checks.

### B. Agent-generated outputs

1. Intercept output before Discord send.
2. Apply output policy (toxicity, prompt leakage, secrets, jailbreak markers).
3. Redact/block/transform based on policy.
4. Emit output moderation decision event.
5. Send sanitized output and attach trace/audit context.

### C. Deity mode actions

1. Mark action as `privilege_tier=deity`.
2. Require strict role + OPA allow decision.
3. Enforce stricter rate limits and two-stage confirmation where applicable.
4. Execute with compensating transaction guidance when possible.
5. Persist mandatory audit log + simulation event + telemetry span.

---

## 3) Role/permission policy

Use a **tiered model** and bind each command to one tier.

- **Tier 0: Public**
  - Example: `dm`, `broadcast`, `status`, `stats`, `help`.
  - Requires standard content policy and per-user global rate limit.

- **Tier 1: Moderator**
  - Example: `mute`, `unmute`, `penalty`, `reset_memory`, `kb` post moderation paths.
  - Requires moderator/admin role OR equivalent OPA grant.
  - Requires command-level cooldown and action budget checks.

- **Tier 2: Admin**
  - Example: `pause_all`, `kill_agent`, `set_max_rate`, bot `kill`.
  - Requires Discord admin permission plus optional OPA allow.
  - Must be audited with actor ID + command args + result.

- **Tier 3: Deity**
  - Example: world mutation, bulk/irreversible controls, emergency overrides.
  - Requires explicit feature flag enablement, admin role, positive OPA decision, and deity-mode rate limits.
  - Default policy: disabled until checklist in Section 9 is satisfied.

### Policy rules

- Prefer **deny-by-default** for Tier 1+ commands.
- OPA lookup failures on Tier 2/3 must fail closed.
- Every policy decision must produce an auditable decision record (`allow/deny`, reason, policy version).

---

## 4) Rate-limit policy

Apply limits at three levels.

1. **Global per-user slash-command limit**
   - Existing baseline in Discord bot command history window.
   - Keep default `5 commands / 60s` unless tuned by environment.

2. **Command-specific cooldowns**
   - Existing moderation cooldown pattern should be retained for sensitive commands.
   - Add per-command windows for high-impact admin/deity actions.

3. **Privilege-tier throttles**
   - Tier 2: low-frequency (example: max 2 destructive commands / minute / actor).
   - Tier 3: very low-frequency (example: max 1 command / 5 minutes / actor + optional global lock).

### Burst handling

- Return explicit user-facing denial (`rate limited`) without executing side effects.
- Log rejected attempts, including user, channel, command, and current counters.

---

## 5) Audit logging requirements (mandatory)

For all Tier 1+ actions, write a structured audit entry with:

- `timestamp`, `trace_id`, `span_id`,
- `actor_user_id`, Discord `channel_id`,
- `command`, `privilege_tier`,
- normalized `arguments` (with sensitive fields redacted),
- policy decision (`allow/deny`), decision source (`discord_role`, `opa`, `both`),
- rate-limit result,
- execution outcome (`queued`, `executed`, `rejected`, `failed`),
- simulation step and target agent(s) where applicable.

### Storage + integrity requirements

- Append-only event stream or ledger-style sink.
- Keep replay linkage (`replay_path` or event references) for incident reconstruction.
- Set retention and access-control policy for audit logs.

---

## 6) Wiring to existing modules

### `src/interfaces/discord_bot.py` (ingress and slash-command control plane)

Current integration points:

- Message intake policy checks in `on_message` (`allow_message`, `evaluate_with_opa`).
- Routing parsing in `_parse_human_message_routing`.
- Global slash rate limiting via `_rate_limit_check`, `check_command_rate_limit`.
- Privileged command authorization via `_has_control_command_permission` and `has_admin_permission`.

Required safety wiring:

- Add an explicit `privilege_tier` map keyed by command name.
- For Tier 1+ commands, emit structured audit events on both allow and deny.
- Ensure OPA errors on Tier 2/3 commands fail closed and are logged as policy backend failures.
- Add deity-mode feature flag gate and dedicated limiter bucket.
- Include reason codes in user-visible denials (`unauthorized`, `rate_limited`, `policy_unavailable`).

### `src/interfaces/discord_moderation.py` (moderation command boundary)

Current integration points:

- Per-action cooldown + OPA-based checks in `_rate_limit` and `moderation_rate_limit`.
- Moderation command registration for `reset_memory`, `penalty`, `mute`, `unmute`.
- Existing penalty logging through `log_penalty`.

Required safety wiring:

- Enforce moderator-or-admin role policy consistently for all moderation commands (not only a subset).
- Extend moderation deny paths to emit structured audit entries.
- Add per-command max attempts/window to complement cooldown checks.
- Require action reason metadata for punitive actions (`penalty`, `reset_memory`).

### `src/sim/simulation.py` (execution boundary)

Current integration points:

- Inbound human command execution in `_handle_human_command`.
- Control command executor in `handle_control_command`.
- Moderation executor in `handle_moderation_command`.

Required safety wiring:

- Validate command payload schemas again at execution boundary (defense in depth).
- Reject unknown commands and malformed arguments with auditable errors.
- Tag emitted events with actor identity + privilege tier passed from ingress.
- For destructive actions, emit pre/post execution events and final status.
- Preserve idempotency guards where feasible (e.g., duplicate spawn rejection pattern).

---

## 7) Agent-output safety pipeline requirements

Before any agent text is sent to Discord:

1. Classify output risk (toxicity, harassment, self-harm, policy-sensitive patterns).
2. Scan for secrets and operational leakage.
3. Block or redact unsafe fragments.
4. Emit moderation telemetry and audit record.
5. Deliver safe fallback text to the channel when blocking occurs.

Implementation hook recommendation:

- Place output filtering in the send path (`send_simulation_update`/message dispatch path) so all agent-originated content passes one enforceable chokepoint.

---

## 8) Deity mode policy (advanced control features)

Deity mode is **disabled by default**.

When enabled, require all of:

- explicit config flag (`DEITY_MODE_ENABLED=true`),
- admin role check,
- successful OPA evaluation against deity policy package,
- stricter rate limiter bucket,
- mandatory audit event on request and outcome,
- optional two-person confirmation for irreversible actions (recommended).

Deity commands must carry:

- `justification` (human-readable reason),
- `ticket_id` or incident/reference ID,
- `requested_by` and `approved_by` (if dual control is enabled).

---

## 9) Pre-enable checklist for advanced controls

Do **not** enable advanced controls until all checks pass:

- [ ] Command-to-tier matrix is defined and versioned.
- [ ] OPA policies exist for Tier 1/2/3 and are tested for allow/deny/error behavior.
- [ ] Fail-closed behavior verified for policy backend outages on Tier 2/3.
- [ ] Global + per-command + deity-mode rate limits configured and load-tested.
- [ ] Structured audit log sink is operational, queryable, and access-controlled.
- [ ] Simulation execution boundary validates payload schemas and unknown commands.
- [ ] Agent-output moderation is active on all outbound channels.
- [ ] Runbook includes incident response for policy bypass, abuse spikes, and rollback.

---

## 10) Minimal implementation sequencing

1. **Audit first**: add structured audit records for existing privileged paths.
2. **Policy hardening**: enforce explicit tier checks + fail-closed OPA behavior.
3. **Limiter hardening**: add per-command and tier-specific throttles.
4. **Execution hardening**: schema validation and outcome events in simulation handlers.
5. **Deity gate**: introduce feature flag and dedicated policy package.
6. **Output guardrails**: central outbound content moderation.

This order minimizes risk while preserving current behavior for non-privileged flows.

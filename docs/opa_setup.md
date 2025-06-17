# OPA Policy Server Setup

This guide shows how to run a local [Open Policy Agent](https://www.openpolicyagent.org/) (OPA) server for message filtering in Culture.ai.

## 1. Start OPA with Docker

The quickest way to run OPA is via Docker:

```bash
docker run -d --name opa -p 8181:8181 openpolicyagent/opa run --server
```

This launches an OPA server on `http://localhost:8181`. OPA will serve policies from the `/policies` directory in the container. You can mount a directory with your policy files if desired:

```bash
docker run -d --name opa -p 8181:8181 -v $(pwd)/policies:/policies \
    openpolicyagent/opa run --server --watch /policies
```

## 2. Example Policy

Create a file `policies/discord.rego` with a simple allow/deny rule:

```rego
package discord

# Block messages containing banned words
default allow = {"allow": true}

allow = {"allow": false} {
    some word
    word := input.message
    lower(word) == "blocked"
}
```

This policy returns `{"allow": false}` when the message is the word `blocked`.

Run OPA again after adding the policy so it is loaded.

## 3. Environment Variables

Configure Culture.ai to use your OPA service by setting these variables in `.env`:

```env
OPA_URL=http://localhost:8181/v1/data/discord/allow
OPA_BLOCKLIST=foo,bar,baz
```

`OPA_URL` points to the policy endpoint. `OPA_BLOCKLIST` is a comma-separated list of words that will be blocked before the OPA check runs.

With OPA running and these variables set, outgoing messages will be filtered according to your policy.

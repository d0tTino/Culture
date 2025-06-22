# Culture.ai Glossary

Below is an expandable **glossary for Culture.ai**—meant to stay valid whether the simulation
is running in Victorian-era London, on a post-singularity ring-world in 100,000 AD,
or in a purely abstract thought-space.
Every term is grounded in multi-agent-systems literature, game-design patterns,
and discrete-event simulation practice.
Treat this as shared vocabulary for UI labels, API schemas, design docs, and red-team playbooks.

---

| Term | Definition |
| --- | --- |
| **Agent** | An autonomous, state-holding entity that perceives, decides, and acts within the environment. Borrowed from agent-based-modeling (ABM) where each agent follows individual rules yet yields system-level patterns ([gistbok-ltb.ucgis.org][1]). |
| **Role** | A dynamic bundle of capabilities, permissions, and social expectations an agent can assume (e.g., "Caretaker", "Scribe", "Explorer"). Roles can be latent embeddings rather than fixed enums, supporting historical or futuristic settings. |
| **Mission** | A **goal-oriented bundle of tasks** with an expected outcome, resources, and success criteria. It may originate from an external planner (director), a faction, or the agent itself via internal goal-generation ([elvtr.com][2]). |
| **Task** | The atomic, executable step inside a mission—equivalent to a single action plan or "order" in Paradox grand-strategy games ([reddit.com][3]). |
| **Faction** | A group of agents that share governance, resources, or ideology; collapses to a family, guild, corporation, or nation depending on era. |
| **Director / Storyteller** | Optional orchestration component that injects incidents to maintain drama pacing (cf. RimWorld storytellers) ([reddit.com][4]). |
| **Player-Operator** | The human overseer who pauses, speeds-up, or inspects the sim; can inject new break-points and missions. |

---

## Temporal Concepts

| Term | Definition |
| --- | --- |
| **Tick** | The minimal time quantum; one update of agent cognition. |
| **Event** | Timestamped record of a state change or interaction; fundamental unit in discrete-event simulation ([softwaresim.com][5]). |
| **Epoch / Era** | Named span of simulation time used for historical segmentation (e.g., "Industrial Age", "Stellar Expansion"). |
| **Scenario** | A saved initial state + parameter set used to start or replay a simulation. |
| **Snapshot** | Deterministic hashable dump of full sim state taken at a given clock; enables rewind & fast-forward ([web.itu.edu.tr][6]). |
| **Time-Warp Factor** | The multiplier (1×, 5×, pause) controlling how fast the tick queue is drained; UI exposes this like Paradox speed controls. |

---

## Knowledge & Memory

| Term | Definition |
| --- | --- |
| **Episodic Memory** | Raw chronological experiences kept per agent; pruned or summarized nightly ([frontiersin.org][7]). |
| **Semantic Memory** | Consolidated, theme-clustered knowledge extracted from episodes (facts, relations) ([en.wikipedia.org][8]). |
| **Procedural Memory** | Learned skills or scripts (e.g., "forge sword", "launch probe")—invoked as production rules. |
| **Knowledge Board (KB)** | A shared, versioned scrapbook where agents post research, proposals, or designs; UI renders it as "Mission Board" or "Project Feed". |
| **Ledger** | Immutable log of DU/IP transfers (see Economy); doubles as event source for replay. |

---

## World & Spatial Concepts

| Term | Definition |
| --- | --- |
| **Cell / Tile / Node** | Smallest addressable spatial unit; can represent a RimWorld-like grid, a solar-system orbital slot, or an abstract network point. |
| **Region** | Contiguous set of cells with homogeneous rules (biome, polity, server shard). |
| **World Fabric** | The service that owns spatial queries (path-finding, visibility). |
| **Portal** | Any non-Euclidean transport link (wormhole, time-gate, corridor) letting agents jump across regions. |

---

## Social Dynamics

| Term | Definition |
| --- | --- |
| **Affinity** | Signed weight on an edge in the **Relationship Graph**; >0 friendship/loyalty, <0 rivalry/hostility ([library.fiveable.me][9]). |
| **Relationship Graph** | Force-directed visualization of affinities among agents; powered by react-force-graph or Cytoscape ([reddit.com][4], [softwaresim.com][5]). |
| **Social Feed Post** | Public or faction-scoped note authored by an agent; may cost DU/IP to broadcast. |
| **Influence-Points (IP)** | Soft currency representing reputation or persuasive capital; drives voting or favor exchange. |
| **Diplomatic State** | Categorical relation between factions—peace, alliance, war, vassalage—mirrors grand-strategy diplomacy systems ([stellaris.paradoxwikis.com][10]). |

---

## Economy & Resources

| Term | Definition |
| --- | --- |
| **Data-Units (DU)** | Base resource consumed by cognitive or communicative actions (LLM tokens, sensor calls). Price may float via gas-like mechanism. |
| **Resource Token** | Generic placeholder for food, energy, credits, mana—defined by scenario metadata. |
| **Auction / Market** | Mechanism where agents exchange resources or tasks; can be continuous double-auction or periodic bazaar. |
| **Staking** | Locking IP/DU to back proposals or infrastructure, earning dividends or governance weight. |

---

## Simulation Control & Safety

| Term | Definition |
| --- | --- |
| **Guardrail** | Rule that evaluates an event or LLM output for policy violations (e.g., self-harm, disallowed content). |
| **Breakpoint** | User-defined tag that triggers auto-pause when matched by an incoming event (e.g., `violence.human`, `economy.hyperinflation`). |
| **Red-Team Event** | Synthetic adversarial probe inserted to test robustness (jailbreak string, prompt injection). |
| **Safety Dashboard** | UI view that lists guardrail hits, breakpoint triggers, and their resolutions. |

---

## Meta & Infrastructure

| Term | Definition |
| --- | --- |
| **Widget** | Front-end module registered in the `WidgetRegistry` (map, timeline, ledger). |
| **Shard** | Independent sim instance; can federate via message queues. |
| **Event Bus** | Pub/sub layer (Redpanda, NATS) delivering EventRecords to UI and downstream analyzers. |
| **Replay Service** | Provides snapshot + delta streams so UI can scrub the timeline at arbitrary speed. |
| **Plug-in** | Hot-swappable Python or JS package that injects new agent types, worlds, or UI widgets. |

---

### How "Mission" Fits Every Era

* **Medieval Fantasy:** Mission = "Escort the caravan from Riverton to Blackspire" (tasks: hire guards, scout route, negotiate toll).
* **Victorian 1888:** Mission = "Unmask the Ripper" (tasks: collect clues, interview witnesses, petition police).
* **Modern Spy-Ops:** Mission = "Neutralize high-value target in Zone Delta" (tasks: infiltrate, gather intel, exfil).
* **Stellar 100 k AD:** Mission = "Stabilize the neutron-star dyson lattice" (tasks: dispatch drone, calibrate field, report anomaly).
* **Abstract Mindscape:** Mission = "Resolve cognitive dissonance node #42" (tasks: recall memories, integrate belief, emit summary).

The formal schema stays identical: `id`, `goal`, `tasks[]`, `priority`, `status`, `due`, `origin`. Only the descriptive strings and task verbs change.

---

## Sources Consulted

1. Agent-based modeling overview – University of Twente ([gistbok-ltb.ucgis.org][1])
2. Game quest & mission writing tips – ELVTR ([elvtr.com][2])
3. RimWorld story-event system – RimWorld Wiki & subreddit ([rimworldwiki.com][11], [reddit.com][4])
4. Discrete-event simulation primer – SoftwareSim & ITU lecture notes ([softwaresim.com][5], [web.itu.edu.tr][6])
5. Emergent narrative discussion – Medium & ACM paper ([medium.com][12], [dl.acm.org][13])
6. Force-directed social graphs – Fiveable tutorial ([library.fiveable.me][9])
7. Cognitive architectures & memory layers – Soar wiki & Frontiers paper ([en.wikipedia.org][8], [frontiersin.org][7])
8. Grand-strategy interface patterns – Paradox wiki & subreddit ([stellaris.paradoxwikis.com][10], [reddit.com][3])
9. Network graph & Cytoscape examples – Cytoscape docs (opened via react-wrapper) ([softwaresim.com][5])

[1]: https://gistbok-ltb.ucgis.org/27/concept/8003?utm_source=chatgpt.com
[2]: https://elvtr.com/blog/the-art-of-writing-game-quests-and-missions?utm_source=chatgpt.com
[3]: https://www.reddit.com/r/paradoxplaza/comments/r8y0s0/definition_of_grand_strategy_game/?utm_source=chatgpt.com
[4]: https://www.reddit.com/r/RimWorld/comments/nmx5bi/which_events_are_triggered_by_storytellers/?utm_source=chatgpt.com
[5]: https://softwaresim.com/blog/a-gentle-introduction-to-discrete-event-simulation/?utm_source=chatgpt.com
[6]: https://web.itu.edu.tr/~etaner/courses/DES/handouts/discrete_event_simulation_concepts_handouts.pdf?utm_source=chatgpt.com
[7]: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2015.00019/epub?utm_source=chatgpt.com
[8]: https://en.wikipedia.org/wiki/Soar_%28cognitive_architecture%29?utm_source=chatgpt.com
[9]: https://library.fiveable.me/data-visualization/unit-14/force-directed-graphs-social-network-analysis/study-guide/7NI1gU9tCiUaOCfL?utm_source=chatgpt.com
[10]: https://stellaris.paradoxwikis.com/Beginner%27s_guide?utm_source=chatgpt.com
[11]: https://rimworldwiki.com/wiki/Events?utm_source=chatgpt.com
[12]: https://medium.com/%40sundryscribes/emergent-narratives-a-different-kind-of-storytelling-f4f343abc1e8?utm_source=chatgpt.com
[13]: https://dl.acm.org/doi/10.1145/3723498.3723809?utm_source=chatgpt.com

# Scenario Evaluation Hooks

Scenario files may register evaluation hooks that gather metrics during a
simulation run. Add the desired hooks under the `evaluation_hooks` key in the
scenario YAML:

```yaml
name: signature_demo
steps: 30
beats:
  - introduction
  - collaboration
  - resolution
evaluation_hooks:
  - coalitions
  - sentiment
  - collective_du
  - collective_ip
```

The following built-in hooks are available:

| Hook name       | Description |
|-----------------|-------------|
| `coalitions`    | Number of projects with more than one member. |
| `sentiment`     | Average agent mood level. |
| `collective_du` | Total DU (durability units) across agents. |
| `collective_ip` | Total IP (influence points) across agents. |

Results from each hook are recorded in the metrics registry and written to the
simulation event log at the end of every beat, enabling downstream analysis or
plotting.

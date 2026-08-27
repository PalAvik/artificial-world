# Run log

**Rule: no new run starts until the previous run's row is filled in.** See `docs/GATES.md`.

| Date | Run ID | Phase | Config | Arch | torch | Attn | Tokens | MSG (held-out) | Probe acc | Bench retention | GPU-h | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |

Arch / torch / attention backend are not bookkeeping: runs compared against each
other must match on all three. See the hardware policy in `docs/GATES.md`.

## Baselines (fill after Gate 1)

| Tier | MSG mean | 95% CI | acc(V_T) | acc(V_I) | delta | n |
|---|---|---|---|---|---|---|
| A — referential | | | | | | |
| B — orthographic | | | | | | |
| C — relational | | | | | | |

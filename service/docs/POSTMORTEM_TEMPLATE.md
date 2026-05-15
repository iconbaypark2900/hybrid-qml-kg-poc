# Postmortem: <one-line summary>

**Date**: YYYY-MM-DD
**Duration**: HH:MM UTC start → HH:MM UTC resolved (Xh Ym)
**Severity**: SEV-{1,2,3} — see [`SLO_SLA.md`](SLO_SLA.md)
**Author**: <name>
**Status**: Draft | In review | Final

---

## TL;DR

Two sentences. What broke, what was the customer impact, what fixed it.

> Example: "Between 14:02 and 14:38 UTC, /predict returned 503 for 100% of
> requests because the LATEST.txt pointer was overwritten with an empty
> string by a concurrent synthesize_manifest_chain run. We restored the
> previous manifest_id and added a write-lock to the synthesizer."

## Impact

| Metric | Value |
|---|---|
| Tenants affected | (count or "all") |
| Requests failed | (estimate from logs) |
| Requests succeeded (with degraded experience, e.g. fallback) | |
| Quantum jobs lost / requeued | |
| Webhook deliveries failed | |
| Estimated revenue / contractual impact | |

Did we breach an SLO? See [`SLO_SLA.md`](SLO_SLA.md). If yes, list which one.

## Timeline (UTC)

| Time | Event |
|---|---|
| 14:00 | Operator ran `synthesize_manifest_chain --force` while service was processing requests |
| 14:02 | First 5xx alert fires |
| 14:05 | On-call paged |
| 14:10 | Acknowledged; investigating |
| 14:25 | Root cause identified |
| 14:34 | Fix applied (`LATEST.txt` restored to prior model_id) |
| 14:38 | Verified `/status.overall=ok`, error rate back to baseline |
| 14:42 | Status page resolved |

## Root cause

_Two paragraphs, technical. What sequence of events caused the failure._

## What went well

- ...
- ...

## What went poorly

- ...
- ...

## Where we got lucky

- ...

## Action items

| Action | Owner | Severity | Due | Tracker |
|---|---|---|---|---|
| Add write-lock to synthesize_manifest_chain | | P0 | YYYY-MM-DD | |
| Add /admin/reload retry with stale-fallback | | P1 | | |
| Update [MANIFEST_CHAIN_BROKEN runbook](runbooks/MANIFEST_CHAIN_BROKEN.md) with this scenario | | P2 | | |

P0 = before next deploy. P1 = within 1 sprint. P2 = backlog.

## Customer comms

- [ ] Status page incident closed with summary
- [ ] Affected tenants emailed (if SEV-1 or SEV-2)
- [ ] Public postmortem published (if commitment in contract)
- [ ] Internal #incidents channel notified

## Lessons learned (for the team)

What's the one thing you wish you'd known before the incident? Add it
here even if it doesn't translate to an action item.

# On-call rotation

This is a **template**. Fill in your team's actual structure, contacts, and
schedule before going live. Without a real human on the other end of an
alert, you don't have an on-call rotation — you have an alarm clock.

## Coverage requirements

| Role | Coverage | Backup | Escalation |
|---|---|---|---|
| Primary on-call | 24/7, 1-week rotation | Secondary | Engineering lead → CTO |
| Secondary on-call | 24/7, 1-week rotation, offset 1 day from primary | Primary | Engineering lead |
| Security on-call | Business hours, escalate via primary after-hours | Primary | Security lead → CTO |

**Minimum team size**: 3 engineers. With 2, every other week is on-call which burns people out fast. With 3+ rotation can be 1 week on, 2+ weeks off.

## Sample rotation (replace with your roster)

| Week starting | Primary | Secondary |
|---|---|---|
| Mon 2026-05-04 | (engineer A) | (engineer B) |
| Mon 2026-05-11 | (engineer B) | (engineer C) |
| Mon 2026-05-18 | (engineer C) | (engineer A) |
| Mon 2026-05-25 | (engineer A) | (engineer B) |

Maintain the source of truth in PagerDuty / OpsGenie / etc., not in this doc. Auto-publish the schedule to a shared calendar.

## Paging

| Alert | Severity | Page | Acknowledge by |
|---|---|---|---|
| `/healthz` down for >2 min | SEV-1 | Yes (primary, then secondary) | 5 min |
| `/predict` 5xx rate >5% for 5 min | SEV-1 | Yes | 5 min |
| `/predict` p95 latency >2x baseline for 10 min | SEV-3 | Yes (business hours), email otherwise | 30 min |
| Manifest chain broken | SEV-2 | Yes | 15 min |
| Quantum executor unavailable >30 min | SEV-2 | Yes | 15 min |
| `/jobs` queue depth >100 | SEV-3 | Email | 1 hour |
| Webhook delivery <80% for 1 hour | SEV-3 | Email | 1 hour |
| Disk usage >90% on artifacts volume | SEV-3 | Page | 30 min |

## On-call expectations

**Primary**:
- Acknowledge pages per the table above
- Triage with the runbooks under `service/docs/runbooks/`
- Bring in the secondary if multiple SEV-1s overlap
- Write the postmortem (template at [`POSTMORTEM_TEMPLATE.md`](POSTMORTEM_TEMPLATE.md))

**Secondary**:
- Be reachable; don't go off-grid
- Cover if primary is unreachable for >10 min
- Pair on SEV-1s

**Escalation triggers** (page next level):
- 30 min into a SEV-1 without resolution
- Customer impact >10% of tenants
- Suspected data loss or breach (page security immediately)
- Any incident requiring a code change to resolve

## Handoffs (Mon morning)

The outgoing primary writes a brief in #ops:

```
**On-call handoff [outgoing → incoming]**

Incidents this week:
- (none) | (list)

Open action items from incidents:
- (none) | (list with owner + due date)

Open paging noise:
- (any false positives, flapping alerts, etc.)

What to watch for next week:
- (e.g. "deploy on Wed will touch the manifest chain")
```

## Compensation

On-call hours are paid above and beyond regular comp. Standard rates:
- Carrying the pager: flat weekly stipend
- Active page response (acknowledged + worked): hourly rate ≥1.5x base
- Sleep disruption (paged 22:00–06:00 local): additional comp day

If you can't compensate on-call fairly, your rotation will quietly become "the same one or two engineers do it forever" and you'll lose them.

## Tooling

- PagerDuty / OpsGenie / VictorOps for the schedule + paging
- Shared incident channel (#ops, #incidents)
- Status page (e.g. statuspage.io / Better Uptime) — outgoing
- Runbooks index: `service/docs/runbooks/`

## When this doc goes stale

If the team grows past 5 engineers, split primary/secondary by domain (one for service+API, one for ML+training). If a single tenant exceeds 50% of traffic, consider a customer-specific on-call. Revisit this doc quarterly.

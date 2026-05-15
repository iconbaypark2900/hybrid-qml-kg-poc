# Service Level Objectives + Agreements

This document defines what "up" means, how we measure it, and the consequences of breach. Every tier here should be measurable from production telemetry — no aspirational SLOs without a metric to back them.

## Severity tiers

| Severity | Definition | Page on-call | Status page | Postmortem required |
|---|---|---|---|---|
| **SEV-1** | Service fully unavailable for any tenant. /predict, /jobs, /status all 5xx. Data loss. | Yes, immediately | Yes, "Major outage" | Yes |
| **SEV-2** | Subset of functionality unavailable. quantum_strict 503, classical works. Or one tenant down, others fine. | Yes, within 15 min | Yes, "Partial outage" | Yes |
| **SEV-3** | Degraded performance. Latency >2x baseline. Honest fallback firing more than usual. | Business hours | Yes, "Degraded" | Optional |
| **SEV-4** | Internal/cosmetic. Logs noisier than usual. Frontend rendering quirks. | No | No | No |

## SLOs (what we promise ourselves)

| Indicator | Target | Window | Measurement |
|---|---|---|---|
| `/healthz` availability | **99.9%** | 30-day rolling | uptime monitor (pingdom/UptimeRobot) hits `/healthz` every 60s |
| `/predict` p95 latency, `method=classical` | **<500ms** | 7-day rolling | Prometheus histogram `request_latency_ms{path="/predict",method="classical"}` |
| `/predict` p95 latency, `method=quantum_strict` (simulator path) | **<3s** | 7-day rolling | same, `method="quantum_strict"` |
| `/predict` error rate | **<0.5%** | 7-day rolling | `rate(request_total{status=~"5..|4[02][0-9]"}[7d]) / rate(request_total[7d])` |
| `/jobs` time-to-completion p95 (IBM hardware) | **<10 minutes** | 30-day rolling | `JobRecord.completed_at - submitted_at` |
| Manifest chain integrity | **100%** | continuous | `/status.components.manifest_store.state == "ok"` |
| Webhook delivery success | **>95%** | 7-day rolling | `JobRecord.callback_status == "delivered"` rate |

## SLAs (what we promise customers in contracts)

These are looser than the SLOs by design — internal targets must beat external ones to leave headroom.

| Tier | Availability | Credit on breach |
|---|---|---|
| Standard | 99.5% / month | 5% of monthly fee per 0.1% below target |
| Enterprise | 99.9% / month | 10% per 0.1% below |

The internal tracker is the SLO. Breach the SLO → action items in the next sprint. Breach the SLA → contractual remedies + postmortem published to the customer.

## Error budget

For 99.9% monthly availability the error budget is **43 minutes / month** of downtime.

When the rolling 30-day error budget is exhausted:
- Freeze non-critical deploys (no new features)
- Focus engineering on reliability work
- Postpone any change that doesn't have a corresponding backout plan

## Measurement integrity

- The uptime probe must hit `/healthz` (no auth, no DB, no manifest reads). If `/healthz` returns 200 but `/status` reports `degraded`, that's a partial outage and counts toward SEV-2 or SEV-3.
- Latency metrics must come from server-side timing, not client-side (clients exaggerate by including their own retries).
- Don't game the SLO by deploying error-paged-as-503 → 200 — that's how trust collapses.

## SLO review cadence

Quarterly: revisit targets vs. actuals. If we consistently beat 99.95%, raise the SLO. If we consistently miss 99.5%, lower it (and have an honest conversation with sales).

## Open questions for the team

- Do we count IBM Quantum outages against our `/jobs` SLO? (Recommendation: no — they're disclosed dependency. But we must communicate.)
- Do we count tenant-side webhook endpoint failures against our delivery SLO? (Recommendation: no — failure to deliver to a 5xx-returning callback is the tenant's problem, but >2 retries should still be logged.)
- What's our SLO on `synthesize_manifest_chain` runtime? (Probably none yet — it's an admin tool.)

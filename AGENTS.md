## Learned User Preferences

- When implementing an attached Cursor plan, do not modify the plan file; use existing todos and mark them in progress instead of recreating them.
- Prefer merging or consolidating work from extra git worktrees into the primary local clone and continuing on the main working tree.

## Learned Workspace Facts

- The day-to-day UI is still Streamlit (`benchmarking/dashboard.py` / `scripts/launch_dashboard.sh`); the Next.js app under `frontend/` is the in-progress migration aligned with Stitch mockups and `docs/frontend/` until parity is reached.
- For API and Next.js work, the preferred slice order is: read-only latest run from `results/` (e.g. `GET /runs/latest`) plus `/experiments` in Next before pipeline job APIs and simulation polling.

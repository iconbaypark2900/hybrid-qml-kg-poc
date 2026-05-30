# Hetionet · QML — Next.js dashboard

Quantum-classical drug-repurposing dashboard on the Hetionet v1.0 knowledge graph,
ported from the single-file HTML version (`../dashboard_live.html`) to a proper
Next.js 14 application.

## Stack

- **Next.js 14** (App Router) · **React 18** · **TypeScript**
- **Zustand** for shared client state (investigation, guards, UI)
- **Plain CSS** with the original dark-theme variables (no Tailwind / no UI lib)
- **D3** for charts · **Three.js + 3d-force-graph** for 3D KG · **3Dmol.js** for molecule viewer
- All client persistence via **localStorage** — no backend yet

## Project layout

```
hetqml-next/
├── app/                      # Next.js App Router
│   ├── layout.tsx            # root layout — sidebar + main outlet
│   ├── globals.css           # full theme (ported from HTML version)
│   ├── page.tsx              # /  →  redirect /initialize
│   ├── initialize/page.tsx   # 01 · Initialize  (fully ported)
│   ├── experiment/page.tsx   # 02 · Experiment  (stub)
│   ├── validate/page.tsx     # 03 · Validate    (stub)
│   ├── visualize/page.tsx    # 04 · Visualize   (stub)
│   ├── operations/page.tsx   # System · Operations (stub)
│   └── settings/page.tsx     # System · Settings   (stub)
├── components/
│   ├── sidebar/Sidebar.tsx           # collapsible navigation
│   ├── shared/HelpHint.tsx           # hover-tooltip ? icon
│   ├── shared/PageStub.tsx           # placeholder for not-yet-migrated pages
│   └── initialize/                   # 6 panels for the Initialize page
│       ├── InitializePage.tsx
│       ├── InvestigationParameters.tsx
│       ├── CandidateContext.tsx
│       ├── RunPath.tsx
│       ├── LiveKgPreview.tsx
│       ├── EvidencePosture.tsx
│       └── Session.tsx
├── data/
│   ├── types.ts              # TS interfaces for every entity
│   ├── metaedges.ts          # 24 Hetionet metaedges (sums to 2,250,197 edges)
│   ├── diseases.ts           # 77 curated diseases (DOID)
│   ├── compounds.ts          # 80 compounds (DrugBank) with therapeutic class
│   ├── genes.ts              # 86 genes (NCBI Gene) with functional category
│   ├── algorithms.ts         # 32 algorithms across 7 groups
│   ├── guards.ts             # 23 integrity guards across 6 categories
│   └── compoundContext.ts    # rich curated context for top candidates
├── lib/
│   ├── store.ts              # Zustand stores (investigation, guards, UI)
│   ├── persistence.ts        # SSR-safe localStorage wrappers
│   ├── hash.ts               # xmur3 deterministic hash
│   └── scoring.ts            # algorithm scoring + top-model selection
└── hooks/                    # (reserved)
```

## Getting started

```bash
cd hetqml-next
npm install
npm run dev
# open http://localhost:3000
```

## What's done

- ✅ Project scaffolding (Next.js 14 / TS / ESLint)
- ✅ Theme + globals.css ported pixel-perfect from the HTML version
- ✅ Sidebar with collapsible navigation, persistence, and hover tooltip portal
- ✅ Routing for all 6 pages (Initialize, Experiment, Validate, Visualize, Operations, Settings)
- ✅ All Hetionet data ported to TypeScript modules with proper types
- ✅ Zustand stores for investigation state, integrity guards, and UI
- ✅ **Initialize page fully ported** with all 6 panels:
  - Investigation Parameters (4-field form + dataset stats badge + tooltips)
  - Candidate Context (curated/derived fallback, equity callout)
  - Run Path (3 preset cards with reactive algorithm counts)
  - Live KG Preview (static SVG; 3D version pending)
  - Evidence Posture (filterable, expandable, with toggle switches)
  - Session (snapshot save/resume/delete with localStorage)
- ✅ Help-icon hover tooltip system (`HelpHint` component)
- ✅ Deterministic scoring + path-aware top-model selection

## What's pending (port from `dashboard_live.html`)

Each stub page contains a checklist of components remaining to migrate. The HTML
version stays as the source of truth until each page reaches feature parity.

Highest-impact remaining work:
1. **Experiment page** — Benchmark Suite (6 tabs) is the most complex; everything
   else (metric strip, source check, leaderboard) is straightforward once the
   scoring functions in `lib/scoring.ts` are reused.
2. **Visualize page** — 3Dmol.js + Three.js components require `dynamic` imports
   with `{ ssr: false }`. The custom Three.js KG renderer in the HTML version
   (around line 4500) is ready to be lifted into a hook.
3. **Validate page** — D3 radar chart + reliability diagram. Pure D3, ports cleanly.
4. **Operations + Settings** — mostly forms, table, and badge components — no 3D.

## Migration patterns

When migrating a panel, follow the established conventions:

1. Read state from Zustand: `useInvestigationStore((s) => s.compound)` etc.
2. Use `loadJson` / `saveJson` from `lib/persistence.ts` for any localStorage I/O
3. Use `scoreAlgorithm` / `pairScore` / `getTopAlgorithm` from `lib/scoring.ts`
4. Wrap a help icon with `<HelpHint text="..." />` inside any element that has `data-help`
5. For 3D / browser-only deps (3Dmol, three, 3d-force-graph), use:

```tsx
import dynamic from 'next/dynamic';
const MoleculeViewer = dynamic(() => import('./MoleculeViewer'), { ssr: false });
```

## License

Apache 2.0 (dashboard) · CC0 1.0 (Hetionet data)

# Design system (Quantum Slate)

The UI/UX upgrade follows the **Quantum-Biological Precision** spec shipped with the Stitch mockups.

## Canonical specification

| Resource | Location |
|----------|----------|
| Written spec (palette, typography, components, do’s/don’ts) | [Quantum Slate DESIGN.md](../../stitch_knowledge_quantum_logic_biomedical/stitch_knowledge_quantum_logic_biomedical/quantum_slate/DESIGN.md) |
| Reference HTML (Tailwind CDN + structure) | `stitch_knowledge_quantum_logic_biomedical/stitch_knowledge_quantum_logic_biomedical/<screen>/code.html` — e.g. [experiment_overview/code.html](../../stitch_knowledge_quantum_logic_biomedical/stitch_knowledge_quantum_logic_biomedical/experiment_overview/code.html) |

## Core color tokens (from DESIGN.md)

Use these in `tailwind.config` `theme.extend.colors` (Next.js); exact hex may also appear in mockup `tailwind.config` blocks inside `code.html`.

| Token role | Example hex | Usage |
|------------|-------------|--------|
| Primary | `#7bd0ff` | Active observation, primary navigation |
| Secondary | `#d0bcff` | Quantum state indicators |
| Tertiary | `#3cddc7` | Biological / success paths |
| Surface / base | `#0b1326` | Canvas (`surface`, `background`) |
| Surface container low | `#131b2e` | Sidebar-style regions |
| Surface container high | `#222a3d` | Cards, modules |
| On-surface | `#dae2fd` | Primary text on dark surfaces |
| Outline / ghost | `#909097` / `#45464d` | Wires, low-contrast borders |

## Tokens to port into Next.js

From the mockup HTML, extract into **`tailwind.config`** (PostCSS pipeline, not CDN):

- **`theme.extend.colors`** — full set from mockup `tailwind.config` in `code.html` (includes `surface-*`, `primary-*`, `on-*`, `outline-*`, etc.)
- **`fontFamily`** — `headline`: Space Grotesk; `body` / `label`: Inter
- **`borderRadius`** — restrained scale; avoid pill radii except chips (per DESIGN.md)

Use **`next/font/google`** for Inter and Space Grotesk instead of duplicate `<link>` tags.

## Icons

Mockups use **Material Symbols Outlined** (see `<link>` tags in `code.html`). Options in Next.js:

- Keep the Google Fonts stylesheet in the root layout, or
- Switch to a React icon set if bundle size or consistency matters more than pixel parity.

## Layout principles (summary)

From DESIGN.md:

- **No-line rule:** avoid hard 1px borders between major regions; use background shifts (`surface` vs `surface-container-*`).
- **Intentional asymmetry** and **focal density** for data-heavy panels.
- **Glass panels** for overlays; **ghost borders** at low opacity for dense tables.

Do not edit the Stitch `code.html` files for product behavior—they are **design references**. Implement behavior in `frontend/` (once created) and shared components.

## Streamlit note

The current `benchmarking/dashboard.py` uses a different visual language (custom CSS). When migrating, **replace** those styles with the Quantum Slate tokens above rather than blending both themes.

## See also

- [MOCKUP_MAP.md](MOCKUP_MAP.md) — which screen uses which mockup file
- [ROUTES.md](ROUTES.md) — where each mockup lands in the app

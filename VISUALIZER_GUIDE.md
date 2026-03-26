# Quantum-KG Drug Repurposing Visualizer — User Guide

## Overview

`qgg_molecular_viz.html` is a self-contained, single-file interactive dashboard that visualizes every stage of the hybrid quantum-classical drug repurposing pipeline — from knowledge graph embeddings through quantum feature maps to final compound-disease link predictions. No server, no install, no dependencies beyond a modern browser.

---

## How to Open the File

### Option A — Double-click (simplest)
1. Navigate to the `hybrid-qml-kg-poc/` folder on your computer.
2. Double-click `qgg_molecular_viz.html`.
3. Your default browser opens it immediately.

> Works in Chrome 90+, Firefox 88+, Edge 90+, Safari 15+. Chrome is recommended for best WebGL performance.

### Option B — Drag into a browser tab
1. Open Chrome or Firefox.
2. Drag `qgg_molecular_viz.html` from your file explorer and drop it onto an open browser tab.

### Option C — Local dev server (avoids any `file://` CORS quirks)

If you have Python installed:
```bash
cd hybrid-qml-kg-poc/
python -m http.server 8080
# then open http://localhost:8080/qgg_molecular_viz.html
```

Or with Node.js:
```bash
npx serve .
# then open the URL printed in your terminal
```

> The file loads all libraries from CDN (Three.js, D3.js, Chart.js) so an internet connection is required on first load. After the browser caches the scripts, it works offline.

---

## Tab-by-Tab Interaction Guide

The dashboard has **6 tabs** along the top navigation bar. Click any tab label to switch views. Each visualization initializes lazily — it renders the first time you visit that tab.

---

### Tab 1 — Predictions

**What it shows:** A ranked table of compound-disease pairs scored by the stacking ensemble model (PR-AUC 0.7987). Each row shows the compound name, disease name, predicted probability, and a color-coded confidence badge.

**Interacting:**
- **Filter buttons** (`All` / `High` / `Medium` / `Low`) below the table header — click to narrow rows by confidence tier.
- **Click any row** — the dashboard highlights that compound in the Molecular 3D tab and centers that disease node in the KG Graph tab.
- Rows are sorted by score descending; scroll the table to see all predictions.

---

### Tab 2 — ⬡ Molecular 3D

**What it shows:** A real-time WebGL render of the selected compound's 3D atom-bond structure using CPK coloring:

| Atom | Color  |
|------|--------|
| C    | Grey   |
| O    | Red    |
| N    | Blue   |
| S    | Yellow |
| F    | Green  |
| H    | White  |

Bonds are rendered as cylinders aligned between atom pairs using quaternion rotation. Phong shading with ambient + directional light gives the molecule depth and gloss.

**Interacting:**
- **Left-click drag** — orbit the molecule (rotate around its centroid).
- **Scroll wheel** — zoom in / zoom out.
- **Compound selector dropdown** (top-left of viewer) — switch between the molecules in the prediction set; the viewer rebuilds geometry instantly.
- The molecule auto-rotates slowly when you are not interacting — move the mouse over the canvas to take manual control.

---

### Tab 3 — ◎ KG Graph

**What it shows:** A D3.js force-directed graph of a subgraph from Hetionet centered on the selected compound. Nodes are color-coded by entity type:

| Entity Type | Color      |
|-------------|------------|
| Compound    | Cyan/teal  |
| Disease     | Orange-red |
| Gene        | Purple     |
| Pathway     | Green      |

Edge thickness encodes relationship strength. Disease nodes carry a score arc overlay showing the predicted link probability.

**Interacting:**
- **Click and drag any node** — reposition it; the simulation continues.
- **Hover over a node** — a tooltip appears with entity name, type, and (for diseases) the link-prediction score.
- **Hover over an edge** — shows the relationship type (e.g., `binds`, `upregulates`, `treats`).
- **Scroll wheel over the graph area** — zoom the entire graph in/out.
- **Click empty space and drag** — pan the graph.
- **Double-click a disease node** — expands its neighborhood (loads additional gene/pathway nodes if available).
- The compound node is **pinned at the center** (`fx`/`fy` fixed); all other nodes float under force simulation.

---

### Tab 4 — ✦ Embedding Space

**What it shows:** A Three.js 3D scatter plot of RotatE knowledge graph embeddings projected to 3D via PCA. Each point is a node from the KG; compounds and diseases are shown in distinct colors with text sprite labels. Known compound-disease treatment edges are drawn as bright connecting lines.

**Interacting:**
- **Left-click drag** — rotate the entire embedding cloud.
- **Scroll wheel** — zoom.
- **Hover over a point** — label brightens and a tooltip shows entity name and type.
- The cloud slowly auto-rotates to give a sense of 3D structure — mouse movement pauses auto-rotation.
- A subtle starfield background provides spatial depth context.

---

### Tab 5 — ⊕ Quantum Circuit

**What it shows:** A static diagram of the Pauli feature map used by the quantum SVM (QSVC). The circuit has **6 qubits** and **2 repetitions** with ZZ entanglement gates. Each element is drawn on an HTML `<canvas>` element using the Canvas 2D API.

**Elements shown:**
- **H gates** — Hadamard initialization boxes
- **Rz(2φᵢ)** — single-qubit Z-rotation gates colored by qubit index
- **ZZ(φᵢφⱼ)** — two-qubit entanglement gates shown as vertical connectors
- **Barrier** separating repetitions

**Interacting:**
- This tab is informational; the diagram is static.
- The canvas scales with window width — resize your browser to see more detail.

---

### Tab 6 — ▦ Model Comparison

**What it shows:** Three Chart.js charts summarizing model evaluation:

1. **Bar chart** — PR-AUC scores across all models (classical RF, ET, LR; quantum QSVC; stacking ensemble).
2. **Scatter chart** — Precision vs. Recall trade-off per model.
3. **Ablation bar chart** — Impact of removing each component (embeddings, quantum features, ensemble) on PR-AUC.

**Interacting:**
- **Hover over any bar or point** — a tooltip shows exact numeric values.
- **Click a legend item** — toggles that dataset's visibility on/off.
- Charts are responsive and resize with the browser window.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1`–`6` | Jump to tab 1–6 |
| `R` | Reset camera in the active 3D viewer (Molecular or Embedding) |
| `Space` | Toggle auto-rotation in the active 3D viewer |

---

## System Requirements

- **Browser:** Chrome 90+ or Firefox 88+ (WebGL 1.0 required for 3D tabs)
- **Internet:** Required on first load to fetch CDN libraries (~500 KB total)
- **Hardware:** Any GPU with WebGL support; integrated graphics is fine
- **RAM:** ~100 MB browser tab memory for both Three.js renderers active

---

## Troubleshooting

**Black/blank 3D viewer canvas**
- Your browser may have WebGL disabled. Go to `chrome://flags` and enable "Override software rendering list", or visit `chrome://settings/system` and ensure hardware acceleration is on.

**Graph doesn't appear on Tab 3**
- D3 loads from CDN. Confirm you have an internet connection and try a hard refresh (`Ctrl+Shift+R`).

**Molecule doesn't rotate when I drag**
- Click directly on the canvas element (not the tab bar area) and then drag.

**Charts show "no data"**
- This shouldn't occur in the shipped file; if it does, open the browser console (`F12`) and check for CDN load errors.

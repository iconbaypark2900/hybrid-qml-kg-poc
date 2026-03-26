# Visualization Tools Reference

Technical documentation for every library and API used in `qgg_molecular_viz.html`. Each section covers the tool's role in the pipeline, the CDN import, key API patterns used in this project, and pointers to official docs.

---

## Table of Contents

1. [Three.js r128 — 3D Molecular Viewer](#1-threejs-r128--3d-molecular-viewer)
2. [Three.js r128 — Embedding Space Scatter](#2-threejs-r128--embedding-space-scatter)
3. [D3.js v7 — Knowledge Graph Force Layout](#3-d3js-v7--knowledge-graph-force-layout)
4. [Canvas 2D API — Quantum Circuit Diagram](#4-canvas-2d-api--quantum-circuit-diagram)
5. [Chart.js 4.4.1 — Model Comparison Charts](#5-chartjs-441--model-comparison-charts)
6. [Pipeline Connection — How Visuals Map to the ML Workflow](#6-pipeline-connection--how-visuals-map-to-the-ml-workflow)

---

## 1. Three.js r128 — 3D Molecular Viewer

### Role in the project

Renders the 3D atom-bond structure of drug candidates in the **⬡ Molecular 3D** tab. Atoms are spheres; bonds are cylinders. CPK coloring convention is applied per atom element. Manual orbit controls (rotate, zoom) are implemented without OrbitControls (not bundled in r128 CDN build).

### CDN import

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
```

### Core objects used

| Object | Purpose |
|--------|---------|
| `THREE.WebGLRenderer` | Renders scene to `<canvas>` using GPU |
| `THREE.PerspectiveCamera` | 75° FOV camera with near/far clipping |
| `THREE.Scene` | Container for all 3D objects |
| `THREE.Group` | Logical group (`molGroup`) rotated as a unit |
| `THREE.SphereGeometry` | Atom spheres (radius proportional to van der Waals) |
| `THREE.CylinderGeometry` | Bond cylinders (radius 0.1, height = bond length) |
| `THREE.MeshPhongMaterial` | Shiny material with `shininess: 80` |
| `THREE.AmbientLight` | Base fill light (0x404040) |
| `THREE.DirectionalLight` | Key light (0xffffff, intensity 1.0) |
| `THREE.Quaternion.setFromUnitVectors` | Aligns cylinder Y-axis to bond direction vector |

### Bond cylinder alignment pattern

The critical technique for aligning a cylinder between two atoms:

```javascript
const dir = new THREE.Vector3().subVectors(posB, posA).normalize();
const mid = new THREE.Vector3().addVectors(posA, posB).multiplyScalar(0.5);
const len = posA.distanceTo(posB);

const geom = new THREE.CylinderGeometry(0.1, 0.1, len, 8);
const mesh = new THREE.Mesh(geom, bondMat);
mesh.position.copy(mid);
mesh.setRotationFromQuaternion(
  new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),  // cylinder default axis
    dir                           // target bond direction
  )
);
```

Three.js `CylinderGeometry` extends along the Y-axis by default. `setFromUnitVectors(Y_hat, bond_dir)` computes the minimal rotation quaternion to align them.

### Manual orbit controls (no OrbitControls)

r128's CDN build does not include `THREE.OrbitControls`. Mouse orbit is implemented by tracking deltas and applying them to the group's Euler angles:

```javascript
canvas.addEventListener('mousedown', e => { isDragging = true; lastX = e.clientX; lastY = e.clientY; });
canvas.addEventListener('mousemove', e => {
  if (!isDragging) return;
  molGroup.rotation.y += (e.clientX - lastX) * 0.01;
  molGroup.rotation.x += (e.clientY - lastY) * 0.01;
  lastX = e.clientX; lastY = e.clientY;
});
canvas.addEventListener('wheel', e => {
  molCamera.position.z = Math.max(2, Math.min(20, molCamera.position.z + e.deltaY * 0.02));
});
```

### Animation loop

```javascript
function molAnimate() {
  molAnimId = requestAnimationFrame(molAnimate);
  if (!isDragging) molGroup.rotation.y += 0.003;  // auto-rotate
  molRenderer.render(molScene, molCamera);
}
```

`stopMolLoop()` calls `cancelAnimationFrame(molAnimId)` when switching away from the tab to avoid invisible background renders.

### CPK color map

```javascript
const CPK = { C:'#888', O:'#e33', N:'#44f', S:'#ee0', F:'#2d2', H:'#fff' };
```

### Official docs

- Three.js docs: https://threejs.org/docs/index.html#api/en/
- r128 migration notes: https://github.com/mrdoob/three.js/releases/tag/r128
- Quaternion primer: https://threejs.org/docs/#api/en/math/Quaternion

---

## 2. Three.js r128 — Embedding Space Scatter

### Role in the project

Renders the **✦ Embedding Space** tab — a 3D scatter plot of RotatE knowledge graph embeddings projected to 3 dimensions via PCA. Compound nodes (cyan) and disease nodes (orange) are positioned in embedding space; known treatment edges are drawn as bright `THREE.Line` objects between them. Text labels use `THREE.Sprite` with a `CanvasTexture`.

### Data source

Embeddings are pre-computed (RotatE via PyKEEN, then scikit-learn `PCA(n_components=3)`) and hard-coded in the HTML as `EMB_NODES` / `EMB_EDGES`:

```javascript
const EMB_NODES = [
  {id:'c0', label:'Aspirin',       type:'compound', x:-2.4, y:1.6,  z:0.6},
  {id:'d0', label:'Colorectal Ca.',type:'disease',  x: 3.0, y:1.8,  z:1.4},
  // ... 20 total nodes
];
const EMB_EDGES = [['c0','d0'], ['c0','d1'], ...];
```

### Text sprite technique

Canvas 2D is used to render text to a texture, then applied to a `THREE.Sprite` (always faces the camera):

```javascript
function makeSprite(text, color='#fff') {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = color;
  ctx.font = 'bold 28px monospace';
  ctx.fillText(text, 8, 44);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(2, 0.5, 1);
  return sprite;
}
```

### Starfield background

256 `THREE.Points` placed randomly on a large sphere give depth cues:

```javascript
const starGeo = new THREE.BufferGeometry();
const positions = new Float32Array(256 * 3).map(() => (Math.random() - 0.5) * 40);
starGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({color:0xffffff, size:0.08}));
embScene.add(stars);
```

### Separate renderer from molecular viewer

Both viewers use independent `THREE.WebGLRenderer` instances targeting different `<canvas>` elements (`#mol-canvas` and `#emb-canvas`). Each has its own scene, camera, and RAF loop. The loops are started/stopped by the tab switching logic:

```javascript
function switchTab(name) {
  stopMolLoop(); stopEmbLoop();
  // ... show tab ...
  if (name === 'mol')  { initMolViewer(); startMolLoop(); }
  if (name === 'emb')  { initEmbViewer(); startEmbLoop(); }
}
```

---

## 3. D3.js v7 — Knowledge Graph Force Layout

### Role in the project

Drives the **◎ KG Graph** tab. A subset of the Hetionet knowledge graph (centered on the selected compound) is rendered as an SVG force simulation with draggable nodes, hover tooltips, and score arcs on disease nodes.

### CDN import

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
```

### Key D3 modules used

| Module | Purpose |
|--------|---------|
| `d3.forceSimulation` | Physics engine that positions nodes |
| `d3.forceLink` | Spring force pulling linked nodes together |
| `d3.forceManyBody` | Charge repulsion between all nodes |
| `d3.forceCenter` | Gravity toward SVG center |
| `d3.forceCollide` | Prevents node overlap |
| `d3.drag` | Mouse drag behavior for repositioning nodes |
| `d3.zoom` | Pan + zoom the entire SVG |
| `d3.arc` | Score arc overlay on disease nodes |
| `d3.select` / `d3.selectAll` | DOM selection and data binding |

### Force simulation setup

```javascript
const sim = d3.forceSimulation(nodes)
  .force('link',    d3.forceLink(links).id(d => d.id).distance(90).strength(0.6))
  .force('charge',  d3.forceManyBody().strength(-300))
  .force('center',  d3.forceCenter(width / 2, height / 2))
  .force('collide', d3.forceCollide(28));
```

The compound node is pinned at the SVG center by setting `node.fx` and `node.fy` permanently:

```javascript
nodes.find(n => n.type === 'compound').fx = width / 2;
nodes.find(n => n.type === 'compound').fy = height / 2;
```

### Dragging

```javascript
const drag = d3.drag()
  .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
  .on('drag',  (e, d) => { d.fx = e.x; d.fy = e.y; })
  .on('end',   (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; });
```

### Score arc overlay

Disease nodes get a `d3.arc` drawn inside the node circle showing link-prediction probability:

```javascript
const arcPath = d3.arc()
  .innerRadius(nodeRadius - 3)
  .outerRadius(nodeRadius)
  .startAngle(0)
  .endAngle(score * 2 * Math.PI);

nodeGroup.append('path')
  .attr('d', arcPath)
  .attr('fill', scoreColor(score));
```

### Curved edges and arrows

To avoid arrow occlusion by node circles, edge endpoints are inset by the node radius plus an arrowhead offset:

```javascript
sim.on('tick', () => {
  link.attr('d', d => {
    const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y;
    const dist = Math.sqrt(dx*dx + dy*dy);
    const offset = 22 / dist;  // nodeRadius + arrowhead
    return `M${d.source.x + dx*offset},${d.source.y + dy*offset}
            L${d.target.x - dx*offset},${d.target.y - dy*offset}`;
  });
  node.attr('transform', d => `translate(${d.x},${d.y})`);
});
```

Arrowhead markers are defined in SVG `<defs>`:

```javascript
svg.append('defs').append('marker')
  .attr('id', 'arrow')
  .attr('viewBox', '0 -5 10 10')
  .attr('refX', 10).attr('markerWidth', 6).attr('markerHeight', 6)
  .attr('orient', 'auto')
  .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', '#666');
```

### Official docs

- D3 v7 API reference: https://d3js.org/
- Force simulation: https://d3js.org/d3-force
- Drag behavior: https://d3js.org/d3-drag
- Arc generator: https://d3js.org/d3-shape/arc

---

## 4. Canvas 2D API — Quantum Circuit Diagram

### Role in the project

Draws the **⊕ Quantum Circuit** tab — a schematic of the Pauli feature map used by the quantum support vector classifier (QSVC). This is a static informational diagram, not an interactive simulation.

### No external library needed

The diagram is drawn entirely with the browser's built-in `CanvasRenderingContext2D`. No import required.

### Circuit parameters

```javascript
const N_QUBITS = 6;   // feature dimensions → qubit count
const N_REPS   = 2;   // Pauli feature map repetitions
const ENTANGLE = 'ZZ'; // entanglement type
```

These match the `PauliFeatureMap(feature_dimension=6, reps=2, entanglement='full')` call in the Python training script.

### Drawing pattern

```javascript
function renderCircuit() {
  const canvas = document.getElementById('circuit-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Qubit wire lines
  for (let q = 0; q < N_QUBITS; q++) {
    const y = topPad + q * wireSpacing;
    ctx.strokeStyle = '#555';
    ctx.beginPath(); ctx.moveTo(leftPad, y); ctx.lineTo(canvas.width - 20, y); ctx.stroke();
    ctx.fillStyle = '#aaa';
    ctx.fillText(`q${q}`, 10, y + 4);
  }

  // H gate (Hadamard) per qubit
  for (let q = 0; q < N_QUBITS; q++) {
    drawGateBox(ctx, x0, topPad + q * wireSpacing, 'H', '#4a9eff');
  }

  // Rz rotation gates
  for (let q = 0; q < N_QUBITS; q++) {
    drawGateBox(ctx, x1, topPad + q * wireSpacing, `Rz(2\u03c6${q})`, '#9b59b6');
  }

  // ZZ entanglement gates (vertical connectors between qubit pairs)
  for (let q = 0; q < N_QUBITS - 1; q++) {
    drawZZGate(ctx, x2, topPad + q * wireSpacing, topPad + (q+1) * wireSpacing);
  }

  // Barrier separator between repetitions
  drawBarrier(ctx, xBarrier);

  // Second repetition ...
}
```

### Gate box helper

```javascript
function drawGateBox(ctx, x, y, label, color) {
  const w = 48, h = 28;
  ctx.fillStyle = color;
  ctx.fillRect(x - w/2, y - h/2, w, h);
  ctx.strokeStyle = '#fff';
  ctx.strokeRect(x - w/2, y - h/2, w, h);
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(label, x, y + 4);
}
```

### Connection to Qiskit / PennyLane

The circuit depicted corresponds to the Qiskit `PauliFeatureMap` (or equivalently the PennyLane `AngleEmbedding` + `StronglyEntanglingLayers` pattern):

```python
# Qiskit equivalent
from qiskit.circuit.library import PauliFeatureMap
feature_map = PauliFeatureMap(feature_dimension=6, reps=2, paulis=['Z','ZZ'])

# PennyLane equivalent
@qml.qnode(dev)
def circuit(x):
    qml.AngleEmbedding(x, wires=range(6), rotation='Z')
    qml.StronglyEntanglingLayers(weights, wires=range(6))
    return qml.expval(qml.PauliZ(0))
```

### Canvas 2D API reference

- MDN CanvasRenderingContext2D: https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D
- HTML Canvas tutorial: https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial

---

## 5. Chart.js 4.4.1 — Model Comparison Charts

### Role in the project

Populates the **▦ Model Comparison** tab with three interactive charts comparing all trained models on precision, recall, and PR-AUC metrics. Uses the `<canvas>` element as its render target but provides a much higher-level API than the raw Canvas 2D.

### CDN import

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
```

### Chart instances

Three `Chart` objects are created, each targeting a different `<canvas>`:

#### 1. PR-AUC Bar Chart

```javascript
new Chart(document.getElementById('prauc-chart'), {
  type: 'bar',
  data: {
    labels: ['RF', 'ET', 'LR', 'QSVC', 'Ensemble'],
    datasets: [{
      label: 'PR-AUC',
      data: [0.742, 0.751, 0.638, 0.694, 0.7987],
      backgroundColor: ['#4a9eff','#4a9eff','#4a9eff','#9b59b6','#e74c3c'],
    }]
  },
  options: {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: { y: { min: 0.5, max: 1.0, title: { display: true, text: 'PR-AUC' } } }
  }
});
```

#### 2. Precision-Recall Scatter

```javascript
new Chart(document.getElementById('pr-scatter'), {
  type: 'scatter',
  data: {
    datasets: models.map(m => ({
      label: m.name,
      data: [{ x: m.recall, y: m.precision }],
      pointRadius: 8,
    }))
  },
  options: { scales: { x: { title: { text: 'Recall' } }, y: { title: { text: 'Precision' } } } }
});
```

#### 3. Ablation Study Bar Chart

```javascript
new Chart(document.getElementById('ablation-chart'), {
  type: 'bar',
  data: {
    labels: ['Full model', 'No embeddings', 'No quantum', 'No ensemble'],
    datasets: [{ label: 'PR-AUC drop', data: [0.7987, 0.621, 0.743, 0.762], ... }]
  }
});
```

### Chart destruction on re-render

Chart.js v4 requires explicit destruction before re-creating a chart on the same canvas (e.g., when switching compound context):

```javascript
if (window._praucChart) window._praucChart.destroy();
window._praucChart = new Chart(ctx, config);
```

### Official docs

- Chart.js docs: https://www.chartjs.org/docs/latest/
- Bar chart: https://www.chartjs.org/docs/latest/charts/bar.html
- Scatter chart: https://www.chartjs.org/docs/latest/charts/scatter.html

---

## 6. Pipeline Connection — How Visuals Map to the ML Workflow

```
Hetionet KG (47k nodes, 2.2M edges)
        │
        ▼
  PyKEEN RotatE           → ✦ Embedding Space tab
  (entity embeddings,       (PCA 3D scatter of node
   relation embeddings)      positions in latent space)
        │
        ▼
  Pair feature vectors
  (head emb ⊕ tail emb
   + quantum features)
        │
   ┌────┴────────────────────────────────┐
   │                                     │
   ▼                                     ▼
Classical models                  QSVC (quantum kernel)
RF / ET / LR                       Pauli feature map
                                   → ⊕ Quantum Circuit tab
   │                                     │
   └────────────┬────────────────────────┘
                │
                ▼
        Stacking Ensemble          → ▦ Model Comparison tab
        (LR meta-learner,            (PR-AUC bars, P/R scatter,
         PR-AUC = 0.7987)             ablation charts)
                │
                ▼
        Ranked predictions         → Predictions tab
        (compound-disease scores)    (filtered table)
                │
                ├──────────────────→ ⬡ Molecular 3D tab
                │                    (Three.js drug structure)
                │
                └──────────────────→ ◎ KG Graph tab
                                     (D3 Hetionet subgraph)
```

### Summary table

| Tab | Library | Data Source | Key Technique |
|-----|---------|-------------|---------------|
| Predictions | Vanilla JS DOM | Ensemble scores | Table filter + row click |
| ⬡ Molecular 3D | Three.js r128 | SMILES → atom coords | SphereGeometry + CylinderGeometry + Quaternion align |
| ◎ KG Graph | D3.js v7 | Hetionet subgraph | forceSimulation + drag + score arc |
| ✦ Embedding Space | Three.js r128 | RotatE PCA coords | BufferGeometry Points + Sprite labels |
| ⊕ Quantum Circuit | Canvas 2D API | Circuit params (n=6, reps=2) | ctx.fillRect + ctx.strokeStyle gates |
| ▦ Model Comparison | Chart.js 4.4.1 | Model eval metrics | Bar + Scatter + Ablation charts |

---

## Further Reading

### Knowledge Graph Embeddings
- PyKEEN library: https://pykeen.readthedocs.io/
- RotatE paper: "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space" (Sun et al., ICLR 2019) — https://arxiv.org/abs/1902.10197
- Hetionet: https://het.io/

### Quantum Machine Learning
- Qiskit Machine Learning: https://qiskit-community.github.io/qiskit-machine-learning/
- PennyLane QML tutorials: https://pennylane.ai/qml/
- QSVC paper: "Quantum kernel methods" (Schuld & Killoran, 2019) — https://arxiv.org/abs/1803.07128
- Pauli Feature Map: https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.PauliFeatureMap

### Drug Repurposing with KG Link Prediction
- Hetionet drug repurposing: https://github.com/hetio/hetionet
- OpenBioLink benchmark: https://github.com/OpenBioLink/OpenBioLink
- Drug repurposing survey: "Knowledge Graph-based Drug Repurposing" (Zeng et al., 2022) — https://doi.org/10.1016/j.media.2021.102266

### 3D Visualization
- Three.js fundamentals: https://threejsfundamentals.org/
- D3 Observable notebooks: https://observablehq.com/@d3
- Chart.js samples: https://www.chartjs.org/docs/latest/samples/

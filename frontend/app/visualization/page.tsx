"use client";

import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import {
  getApiBaseUrl,
  fetchVizRunPredictions,
  fetchVizPredictions,
  fetchVizMolecule,
  fetchVizKGSubgraph,
  fetchVizModelMetrics,
  fetchVizCircuitParams,
  fetchVizEmbeddingVector,
  predictLink,
  searchKGEntities,
} from "@/lib/api";
import type {
  VizRunPrediction,
  VizAtom,
  VizBond,
  VizKGNode,
  VizKGLink,
  KGSearchResult,
  VizModelMetric,
  VizCircuitResponse,
  EmbeddingVectorResponse,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type TabId =
  | "predictions"
  | "mol3d"
  | "kggraph"
  | "embedding"
  | "circuit"
  | "comparison";

interface TabDef {
  id: TabId;
  label: string;
  icon: string;
}

const TABS: TabDef[] = [
  { id: "predictions", label: "Predictions", icon: "leaderboard" },
  { id: "mol3d", label: "Molecular 3D", icon: "science" },
  { id: "kggraph", label: "KG Graph", icon: "hub" },
  { id: "embedding", label: "Feature Vectors", icon: "scatter_plot" },
  { id: "circuit", label: "Quantum Circuit", icon: "memory" },
  { id: "comparison", label: "Model Comparison", icon: "bar_chart" },
];

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function VisualizationPage() {
  const [tab, setTab] = useState<TabId>("predictions");
  // Shared state: compound selected from predictions → used by Mol3D tab
  const [selectedCompound, setSelectedCompound] = useState("Cyclosporine");

  const viewMolecule = useCallback((name: string) => {
    setSelectedCompound(name);
    setTab("mol3d");
  }, []);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      const idx = parseInt(e.key) - 1;
      if (idx >= 0 && idx < TABS.length) setTab(TABS[idx].id);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="space-y-4">
      <header>
        <h1 className="font-headline text-2xl font-semibold tracking-tight text-on-surface">
          Pipeline Visualizer
        </h1>
        <p className="mt-1 text-sm text-on-surface-variant">
          Interactive dashboard — predictions from pipeline runs, KG graph,
          embeddings, quantum circuit, model comparison.
        </p>
      </header>

      {/* Tab bar */}
      <nav className="flex gap-1 overflow-x-auto rounded-lg bg-surface-container-high/60 p-1">
        {TABS.map((t, i) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center gap-1.5 whitespace-nowrap rounded-md px-3 py-2 text-sm font-medium transition-colors ${
              tab === t.id
                ? "bg-primary/15 text-primary"
                : "text-on-surface-variant hover:bg-surface-container-lowest/40 hover:text-on-surface"
            }`}
          >
            <span className="material-symbols-outlined text-[18px]">
              {t.icon}
            </span>
            <span className="hidden sm:inline">{t.label}</span>
            <kbd className="ml-1 hidden rounded bg-surface-container-lowest/60 px-1 text-[10px] text-on-surface-variant sm:inline">
              {i + 1}
            </kbd>
          </button>
        ))}
      </nav>

      {/* Tab panels */}
      <div className="min-h-[520px]">
        {tab === "predictions" && <PredictionsTab onViewMolecule={viewMolecule} />}
        {tab === "mol3d" && <Mol3DTab initialCompound={selectedCompound} />}
        {tab === "kggraph" && <KGGraphTab />}
        {tab === "embedding" && <EmbeddingTab />}
        {tab === "circuit" && <CircuitTab />}
        {tab === "comparison" && <ComparisonTab />}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function LoadingMsg({ text = "Loading\u2026" }: { text?: string }) {
  return (
    <p className="py-12 text-center text-sm text-on-surface-variant" role="status">
      {text}
    </p>
  );
}

function ErrorMsg({ msg }: { msg: string }) {
  return (
    <div className="rounded-lg border border-error/40 bg-error-container/20 p-4">
      <p className="text-sm font-medium text-error">Visualization error</p>
      <p className="mt-1 text-xs text-on-surface-variant">{msg}</p>
      <p className="mt-3 text-xs text-on-surface-variant">
        Base URL: <code className="text-on-surface">{getApiBaseUrl()}</code>
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 1. Predictions table
// ---------------------------------------------------------------------------

function PredictionsTab({ onViewMolecule }: { onViewMolecule: (name: string) => void }) {
  const [preds, setPreds] = useState<VizRunPrediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<"All" | "High" | "Medium" | "Low">("All");
  const [runTs, setRunTs] = useState<string | null>(null);
  const [sourceFile, setSourceFile] = useState<string | null>(null);
  const [availableRuns, setAvailableRuns] = useState<string[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>("");

  const load = useCallback((run?: string) => {
    setLoading(true);
    setError(null);
    fetchVizRunPredictions(50, run || undefined)
      .then((r) => {
        if (r.status !== "ok") throw new Error(r.message ?? "Failed");
        const list = (r.predictions ?? []).filter(
          (p): p is VizRunPrediction =>
            p != null &&
            typeof p.compound_id === "string" &&
            typeof p.disease_id === "string",
        );
        setPreds(list);
        setRunTs(r.run_timestamp ?? null);
        setSourceFile(r.source_file ?? null);
        setAvailableRuns(r.available_runs);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  if (loading) return <LoadingMsg text="Loading run predictions\u2026" />;
  if (error) return <ErrorMsg msg={error} />;
  if (!preds.length)
    return (
      <div className="space-y-3">
        <p className="py-8 text-center text-sm text-on-surface-variant">
          No predictions found. Run the pipeline first:
        </p>
        <pre className="mx-auto max-w-xl rounded bg-surface-container-lowest px-3 py-2 text-xs text-on-surface">
          python scripts/run_optimized_pipeline.py --relation CtD
        </pre>
      </div>
    );

  const safePreds = preds.filter(
    (p) => p != null && p.compound_id != null && p.disease_id != null,
  );
  const filtered =
    filter === "All" ? safePreds : safePreds.filter((p) => p.confidence === filter);

  return (
    <div className="space-y-3">
      {/* Run selector + metadata */}
      <div className="flex flex-wrap items-center gap-3 text-xs text-on-surface-variant">
        {availableRuns.length > 1 && (
          <label className="flex items-center gap-1.5">
            Run:
            <select
              value={selectedRun}
              onChange={(e) => { setSelectedRun(e.target.value); load(e.target.value); }}
              className="rounded bg-surface-container-lowest px-2 py-1 text-xs text-on-surface outline-none ring-1 ring-outline-variant/30"
            >
              <option value="">Latest</option>
              {availableRuns.map((ts) => (
                <option key={ts} value={ts}>{ts}</option>
              ))}
            </select>
          </label>
        )}
        {runTs && <span>Timestamp: <strong className="text-on-surface">{runTs}</strong></span>}
        {sourceFile && <span>Source: <strong className="text-on-surface">{sourceFile}</strong></span>}
        <span>{preds.length} predictions</span>
      </div>

      {/* Filters */}
      <div className="flex gap-2">
        {(["All", "High", "Medium", "Low"] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
              filter === f
                ? "bg-primary/20 text-primary"
                : "bg-surface-container-high/60 text-on-surface-variant hover:text-on-surface"
            }`}
          >
            {f}
          </button>
        ))}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-surface-container-high text-left text-xs uppercase tracking-wide text-on-surface-variant">
              <th className="px-3 py-2">#</th>
              <th className="px-3 py-2">Compound</th>
              <th className="px-3 py-2">Disease</th>
              <th className="px-3 py-2 text-right">Classical</th>
              <th className="px-3 py-2 text-right">Quantum</th>
              <th className="px-3 py-2 text-center">True</th>
              <th className="px-3 py-2 text-center">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((p, i) => (
              <tr
                key={`${p.compound_id}-${p.disease_id}-${i}`}
                className="border-b border-outline-variant/10 hover:bg-surface-container-lowest/50"
              >
                <td className="px-3 py-2 font-mono text-on-surface-variant">
                  {i + 1}
                </td>
                <td className="px-3 py-2">
                  {p.compound_id?.startsWith("Compound::") ? (
                    <button
                      onClick={() => onViewMolecule(p.compound_name)}
                      className="font-medium text-primary hover:underline"
                      title="View 3D structure"
                    >
                      {p.compound_name}
                      <span className="material-symbols-outlined ml-1 align-middle text-[14px] opacity-60">view_in_ar</span>
                    </button>
                  ) : (
                    <span className="font-medium text-on-surface">{p.compound_name}</span>
                  )}
                </td>
                <td className="px-3 py-2 text-on-surface-variant">
                  {p.disease_name}
                </td>
                <td className="px-3 py-2 text-right font-mono text-tertiary">
                  {p.score_classical != null ? p.score_classical.toFixed(4) : "\u2014"}
                </td>
                <td className="px-3 py-2 text-right font-mono text-secondary">
                  {p.score_quantum != null ? p.score_quantum.toFixed(4) : "\u2014"}
                </td>
                <td className="px-3 py-2 text-center">
                  <span className={p.y_true ? "text-tertiary" : "text-on-surface-variant"}>
                    {p.y_true ? "CtD" : "\u00d7"}
                  </span>
                </td>
                <td className="px-3 py-2 text-center">
                  <span
                    className={`inline-block rounded-md px-2 py-0.5 text-xs font-semibold ${
                      p.confidence === "High"
                        ? "bg-tertiary/20 text-tertiary"
                        : p.confidence === "Medium"
                          ? "bg-primary/20 text-primary"
                          : "bg-outline-variant/20 text-on-surface-variant"
                    }`}
                  >
                    {p.confidence}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 2. Molecular 3D (Three.js via PubChem)
// ---------------------------------------------------------------------------

const CPK: Record<string, string> = {
  C: "#888888",
  O: "#ee3333",
  N: "#4444ff",
  S: "#eeee00",
  F: "#22dd22",
  H: "#ffffff",
  Cl: "#33cc33",
  Br: "#993300",
  P: "#ff8800",
  I: "#6600bb",
};

const VDW: Record<string, number> = {
  C: 0.35, O: 0.3, N: 0.32, S: 0.4, F: 0.28, H: 0.2,
  Cl: 0.38, Br: 0.42, P: 0.38, I: 0.44,
};

interface CompoundListItem {
  id: string;
  name: string;
}

/** Compounds from latest run predictions (deduped), with CtD-sample fallback — shared by Mol3D + KG tabs. */
function usePipelineCompoundList() {
  const [rows, setRows] = useState<CompoundListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [hint, setHint] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setHint(null);
      try {
        const run = await fetchVizRunPredictions(400);
        if (cancelled) return;
        if (run.status === "ok" && run.predictions.length) {
          const map = new Map<string, CompoundListItem>();
          for (const p of run.predictions) {
            if (!p?.compound_id) continue;
            if (!map.has(p.compound_id)) {
              map.set(p.compound_id, {
                id: p.compound_id,
                name: p.compound_name ?? p.compound_id,
              });
            }
          }
          setRows(Array.from(map.values()));
          setHint(`${run.predictions.length} rows → ${map.size} unique compounds (pipeline run)`);
          return;
        }
        const live = await fetchVizPredictions(80);
        if (cancelled) return;
        if (live.status === "ok" && live.predictions.length) {
          const map = new Map<string, CompoundListItem>();
          for (const p of live.predictions) {
            if (!p?.compound_id) continue;
            if (!map.has(p.compound_id)) {
              map.set(p.compound_id, {
                id: p.compound_id,
                name: p.compound_name ?? p.compound_id,
              });
            }
          }
          setRows(Array.from(map.values()));
          setHint(`${map.size} compounds (scored CtD sample)`);
          return;
        }
        setHint("No compound list from API — use manual entry or run the pipeline.");
      } catch {
        if (!cancelled) setHint("Could not load compound list.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  return { rows, loading, hint };
}

function Mol3DTab({ initialCompound }: { initialCompound: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [compound, setCompound] = useState(initialCompound || "Cyclosporine");
  const [listQuery, setListQuery] = useState("");
  const { rows: compoundRows, loading: listLoading, hint: listHint } = usePipelineCompoundList();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const lastInitialRef = useRef(initialCompound);

  const load = useCallback((query: string) => {
    setLoading(true);
    setError(null);
    setInfo(null);
    cleanupRef.current?.();
    cleanupRef.current = null;

    (async () => {
      try {
        const [mol, THREE] = await Promise.all([
          fetchVizMolecule(query),
          import("three"),
        ]);
        if (mol.status !== "ok") throw new Error(mol.message ?? "Lookup failed");
        if (!mol.atoms.length) throw new Error("No atoms returned");
        if (!containerRef.current) return;

        setInfo(`${mol.compound_name} — ${mol.atoms.length} atoms, ${mol.bonds.length} bonds`);
        cleanupRef.current = renderMolecule3D(
          containerRef.current,
          THREE,
          mol.atoms,
          mol.bonds,
        );
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    load(compound);
    return () => { cleanupRef.current?.(); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Parent navigation (Predictions tab → view 3D)
  useEffect(() => {
    if (initialCompound && initialCompound !== lastInitialRef.current) {
      lastInitialRef.current = initialCompound;
      setCompound(initialCompound);
      load(initialCompound);
    }
  }, [initialCompound, load]);

  const filteredRows = compoundRows.filter((c) => {
    if (!c?.id) return false;
    const q = listQuery.trim().toLowerCase();
    if (!q) return true;
    return (
      (c.name ?? "").toLowerCase().includes(q) ||
      c.id.toLowerCase().includes(q)
    );
  });

  function selectCompound(row: CompoundListItem) {
    if (!row?.id) return;
    const q = row.id.startsWith("Compound::") ? row.id : (row.name ?? row.id);
    setCompound(q);
    load(q);
  }

  return (
    <div className="flex flex-col gap-4 lg:flex-row lg:items-stretch">
      <aside className="flex w-full shrink-0 flex-col rounded-lg border border-outline-variant/15 bg-surface-container-high/40 lg:w-72">
        <div className="border-b border-outline-variant/10 px-3 py-2">
          <p className="text-xs font-medium uppercase tracking-wide text-on-surface-variant">
            Compounds
          </p>
          <p className="mt-0.5 text-[11px] text-on-surface-variant/80">
            Click a row to load 3D (PubChem)
          </p>
        </div>
        <div className="px-2 py-2">
          <input
            type="search"
            value={listQuery}
            onChange={(e) => setListQuery(e.target.value)}
            placeholder="Filter list…"
            className="w-full rounded-md bg-surface-container-lowest px-2 py-1.5 text-xs text-on-surface outline-none ring-1 ring-outline-variant/30 focus:ring-primary/60"
            aria-label="Filter compounds"
          />
        </div>
        <div
          className="max-h-[min(520px,40vh)] flex-1 overflow-y-auto px-1 pb-2 lg:max-h-[520px]"
          role="list"
        >
          {listLoading ? (
            <p className="px-2 py-4 text-center text-xs text-on-surface-variant">Loading list…</p>
          ) : filteredRows.length === 0 ? (
            <p className="px-2 py-4 text-center text-xs text-on-surface-variant">
              No matches. Try clearing the filter or run the pipeline for a larger set.
            </p>
          ) : (
            <ul className="space-y-0.5">
              {filteredRows.map((row) => {
                const active =
                  compound === row.id ||
                  compound === row.name ||
                  compound.toLowerCase() === row.name.toLowerCase();
                return (
                  <li key={row.id}>
                    <button
                      type="button"
                      onClick={() => selectCompound(row)}
                      className={`w-full rounded-md px-2 py-2 text-left text-sm transition-colors ${
                        active
                          ? "bg-primary/20 text-primary"
                          : "text-on-surface hover:bg-surface-container-lowest/80"
                      }`}
                    >
                      <span className="line-clamp-2 font-medium">{row.name}</span>
                      <span className="mt-0.5 block font-mono text-[10px] text-on-surface-variant">
                        {row.id}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
        {listHint && !listLoading && (
          <p className="border-t border-outline-variant/10 px-3 py-2 text-[10px] text-on-surface-variant/90">
            {listHint}
          </p>
        )}
      </aside>

      <div className="min-w-0 flex-1 space-y-3">
        <form
          className="flex flex-wrap items-end gap-2"
          onSubmit={(e) => { e.preventDefault(); load(compound); }}
        >
          <div className="min-w-[200px] flex-1">
            <label className="mb-1 block text-xs text-on-surface-variant">
              Compound name or Hetionet ID
            </label>
            <input
              value={compound}
              onChange={(e) => setCompound(e.target.value)}
              className="w-full rounded-md bg-surface-container-lowest px-3 py-2 text-sm text-on-surface outline-none ring-1 ring-outline-variant/30 focus:ring-primary/60"
              placeholder="Aspirin, Cyclosporine, Compound::DB00091 …"
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="rounded-md bg-primary/15 px-4 py-2 text-sm font-medium text-primary hover:bg-primary/25 disabled:opacity-50"
          >
            {loading ? "Loading\u2026" : "Render"}
          </button>
        </form>

        {error && <ErrorMsg msg={error} />}
        {info && (
          <p className="text-xs text-on-surface-variant">{info} (via PubChem 3D conformer)</p>
        )}

        <div
          ref={containerRef}
          className="relative h-[min(520px,55vh)] w-full overflow-hidden rounded-lg border border-outline-variant/15 bg-surface-container-lowest/40 lg:h-[520px]"
          style={{ display: error && !loading ? "none" : undefined }}
        />

        <div className="flex flex-wrap gap-3 text-xs text-on-surface-variant">
          {Object.entries(CPK).slice(0, 8).map(([el, color]) => (
            <span key={el} className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: color }} />
              {el}
            </span>
          ))}
          <span className="text-on-surface-variant/60">
            Drag to rotate &middot; Scroll to zoom &middot; Auto-rotates when idle
          </span>
        </div>
      </div>
    </div>
  );
}

function renderMolecule3D(
  container: HTMLElement,
  T: typeof import("three"),
  atoms: VizAtom[],
  bonds: VizBond[],
): () => void {
  const THREE = T;
  const width = container.clientWidth;
  const height = container.clientHeight || 520;

  // Clear any previous canvas
  while (container.firstChild) container.removeChild(container.firstChild);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x060e20);
  const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 200);

  // Lighting — warm key, cool fill, purple accent
  scene.add(new THREE.AmbientLight(0x404040, 1.0));
  const keyLight = new THREE.DirectionalLight(0xffeedd, 0.9);
  keyLight.position.set(5, 5, 5);
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight(0x6644aa, 0.4);
  fillLight.position.set(-5, -3, 2);
  scene.add(fillLight);
  const accentLight = new THREE.PointLight(0x00ccff, 0.3, 20);
  accentLight.position.set(0, 5, -5);
  scene.add(accentLight);

  const group = new THREE.Group();
  scene.add(group);

  // Centre molecule
  const cx = atoms.reduce((s, a) => s + a.x, 0) / atoms.length;
  const cy = atoms.reduce((s, a) => s + a.y, 0) / atoms.length;
  const cz = atoms.reduce((s, a) => s + a.z, 0) / atoms.length;

  // Atom spheres
  const positions: import("three").Vector3[] = [];
  atoms.forEach((a) => {
    const r = VDW[a.element] ?? 0.3;
    const color = CPK[a.element] ?? "#cccccc";
    const geo = new THREE.SphereGeometry(r, 24, 24);
    const mat = new THREE.MeshPhongMaterial({ color, shininess: 80 });
    const mesh = new THREE.Mesh(geo, mat);
    const pos = new THREE.Vector3(a.x - cx, a.y - cy, a.z - cz);
    mesh.position.copy(pos);
    group.add(mesh);
    positions.push(pos);
  });

  // Bond cylinders
  const bondMat = new THREE.MeshPhongMaterial({ color: 0x999999, shininess: 40 });
  bonds.forEach((b) => {
    if (b.source >= positions.length || b.target >= positions.length) return;
    const posA = positions[b.source];
    const posB = positions[b.target];
    const dir = new THREE.Vector3().subVectors(posB, posA);
    const len = dir.length();
    if (len < 0.01) return;
    dir.normalize();
    const mid = new THREE.Vector3().addVectors(posA, posB).multiplyScalar(0.5);

    const geo = new THREE.CylinderGeometry(0.08, 0.08, len, 8);
    const mesh = new THREE.Mesh(geo, bondMat);
    mesh.position.copy(mid);
    mesh.setRotationFromQuaternion(
      new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        dir,
      ),
    );
    group.add(mesh);
  });

  // Camera distance from bounding sphere
  let maxR = 0;
  positions.forEach((p) => { maxR = Math.max(maxR, p.length()); });
  camera.position.set(0, 0, maxR * 2.5 + 3);

  // Manual orbit
  let isDragging = false;
  let lastX = 0;
  let lastY = 0;
  const onDown = (e: PointerEvent) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; };
  const onMove = (e: PointerEvent) => {
    if (!isDragging) return;
    group.rotation.y += (e.clientX - lastX) * 0.008;
    group.rotation.x += (e.clientY - lastY) * 0.008;
    lastX = e.clientX; lastY = e.clientY;
  };
  const onUp = () => { isDragging = false; };
  const onWheel = (e: WheelEvent) => {
    camera.position.z = Math.max(2, Math.min(50, camera.position.z + e.deltaY * 0.03));
  };
  renderer.domElement.addEventListener("pointerdown", onDown);
  renderer.domElement.addEventListener("pointermove", onMove);
  renderer.domElement.addEventListener("pointerup", onUp);
  renderer.domElement.addEventListener("pointerleave", onUp);
  renderer.domElement.addEventListener("wheel", onWheel);

  // Animate
  let animId: number;
  function animate() {
    animId = requestAnimationFrame(animate);
    if (!isDragging) group.rotation.y += 0.003;
    renderer.render(scene, camera);
  }
  animate();

  // Resize
  const ro = new ResizeObserver(() => {
    const w = container.clientWidth;
    const h = container.clientHeight || 520;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  });
  ro.observe(container);

  return () => {
    cancelAnimationFrame(animId);
    ro.disconnect();
    renderer.domElement.removeEventListener("pointerdown", onDown);
    renderer.domElement.removeEventListener("pointermove", onMove);
    renderer.domElement.removeEventListener("pointerup", onUp);
    renderer.domElement.removeEventListener("pointerleave", onUp);
    renderer.domElement.removeEventListener("wheel", onWheel);
    renderer.dispose();
    if (container.contains(renderer.domElement))
      container.removeChild(renderer.domElement);
  };
}

// ---------------------------------------------------------------------------
// 3. KG Graph (D3 force)
// ---------------------------------------------------------------------------

const KG_COLORS: Record<string, string> = {
  Compound: "#7bd0ff",
  Disease: "#ff8a65",
  Gene: "#d0bcff",
  Pathway: "#3cddc7",
  Anatomy: "#ffd54f",
  "Biological Process": "#81c784",
  "Pharmacologic Class": "#90a4ae",
};

function KGGraphTab() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [nodes, setNodes] = useState<VizKGNode[]>([]);
  const [links, setLinks] = useState<VizKGLink[]>([]);
  const [center, setCenter] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [entityInput, setEntityInput] = useState("Compound::DB00635");
  const [hops, setHops] = useState(1);
  const [maxNodes, setMaxNodes] = useState(120);
  const [searchResults, setSearchResults] = useState<KGSearchResult[]>([]);
  const [showSearch, setShowSearch] = useState(false);
  const searchTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const { rows: compoundOptions, loading: compoundListLoading, hint: compoundListHint } =
    usePipelineCompoundList();

  const sortedCompounds = useMemo(
    () =>
      [...compoundOptions]
        .filter((c) => c != null && c.id != null && c.id !== "")
        .sort((a, b) => (a.name ?? "").localeCompare(b.name ?? "")),
    [compoundOptions],
  );

  const compoundSelectValue = useMemo(
    () => (compoundOptions.some((c) => c?.id === entityInput) ? entityInput : ""),
    [compoundOptions, entityInput],
  );

  const load = useCallback((entity: string, h?: number, mn?: number) => {
    setLoading(true);
    setError(null);
    setShowSearch(false);
    fetchVizKGSubgraph(entity, mn ?? maxNodes, h ?? hops)
      .then((r) => {
        if (r.status !== "ok") throw new Error(r.message ?? "Failed");
        setNodes(
          (r.nodes ?? []).filter(
            (n): n is VizKGNode =>
              n != null && typeof n.id === "string" && n.id.length > 0,
          ),
        );
        setLinks(
          (r.links ?? []).filter(
            (l) =>
              l != null &&
              l.source != null &&
              l.target != null &&
              String(l.source).length > 0 &&
              String(l.target).length > 0,
          ),
        );
        setCenter(r.center_entity ?? null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, [hops, maxNodes]);

  // Autocomplete search
  const onSearchInput = useCallback((val: string) => {
    setEntityInput(val);
    if (searchTimeout.current) clearTimeout(searchTimeout.current);
    if (val.length < 2) { setSearchResults([]); setShowSearch(false); return; }
    searchTimeout.current = setTimeout(() => {
      searchKGEntities(val, 12)
        .then((r) => { setSearchResults(r.results); setShowSearch(r.results.length > 0); })
        .catch(() => setShowSearch(false));
    }, 250);
  }, []);

  useEffect(() => {
    load(entityInput);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // D3 render — on double-click a node, re-center on it
  const loadRef = useRef(load);
  loadRef.current = load;

  useEffect(() => {
    if (!nodes.length || !svgRef.current) return;
    let destroyed = false;

    import("d3")
      .then((d3) => {
        if (destroyed || !svgRef.current) return;
        try {

      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();
      const width = svgRef.current.clientWidth || 800;
      const height = 520;
      svg.attr("viewBox", `0 0 ${width} ${height}`);

      // Arrow marker - adjusted for better visibility
      svg
        .append("defs")
        .append("marker")
        .attr("id", "arrow")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 26)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("fill", "#666");

      const g = svg.append("g");

      // Zoom with smoother transitions
      // eslint-disable-next-line
      svg.call(
        d3
          .zoom<SVGSVGElement, unknown>()
          .scaleExtent([0.1, 4])
          .on("zoom", (e) => g.attr("transform", e.transform)) as never,
      );

      // Build sim data with proper node mapping (drop null/invalid nodes — avoids D3 id() / null .id crashes)
      type SimNode = VizKGNode & d3.SimulationNodeDatum;
      const validNodes = nodes.filter(
        (n): n is VizKGNode => n != null && typeof n.id === "string" && n.id.length > 0,
      );
      // One simulation node per id (duplicate ids break forceLink id() lookups)
      const seenIds = new Set<string>();
      const simNodes: SimNode[] = [];
      for (const n of validNodes) {
        if (seenIds.has(n.id)) continue;
        seenIds.add(n.id);
        simNodes.push({ ...n });
      }
      const nodeIds = new Set(simNodes.map((n) => n.id));
      
      // Filter links to only include edges where both nodes exist
      const simLinks = links
        .filter(
          (l) =>
            l != null &&
            l.source != null &&
            l.target != null &&
            nodeIds.has(String(l.source)) &&
            nodeIds.has(String(l.target)),
        )
        .map((l) => ({
          source: l.source,
          target: l.target,
          relation: l.relation ?? "",
        }));

      // Adaptive force parameters based on graph size
      const nodeCount = simNodes.length;
      const chargeStrength = nodeCount > 100 ? -80 : nodeCount > 60 ? -120 : -180;
      const linkDist = nodeCount > 100 ? 50 : nodeCount > 60 ? 70 : 90;
      const collideRadius = nodeCount > 100 ? 15 : nodeCount > 60 ? 18 : 22;

      const sim = d3
        .forceSimulation<SimNode>(simNodes)
        .force(
          "link",
          d3
            .forceLink<SimNode, (typeof simLinks)[0]>(simLinks)
            .id((d) => String(d?.id ?? ""))
            .distance(linkDist)
            .strength((d: any) => {
              const priorityRels = ["CtD", "CbG", "CpD"];
              return priorityRels.includes(d?.relation) ? 0.7 : 0.4;
            }),
        )
        .force("charge", d3.forceManyBody().strength(chargeStrength))
        .force("center", d3.forceCenter(width / 2, height / 2).strength(0.05))
        .force("collide", d3.forceCollide(collideRadius).strength(0.7))
        .force("x", d3.forceX(width / 2).strength(0.02))
        .force("y", d3.forceY(height / 2).strength(0.02));

      // Pin center entity
      const centerNode = simNodes.find((n) => n.id === center);
      if (centerNode) {
        centerNode.fx = width / 2;
        centerNode.fy = height / 2;
      }

      // Edges with relation-based coloring
      const linkG = g.append("g");
      const link = linkG
        .selectAll("line")
        .data(simLinks)
        .enter()
        .append("line")
        .attr("stroke", (d) => {
          const relColors: Record<string, string> = {
            CtD: "#7bd0ff", CbG: "#a78bfa", CpD: "#ff8a65",
            DaG: "#f9a825", CuG: "#4db6ac", CdG: "#ef5350",
            DuG: "#90a4ae", DdG: "#81c784", DpS: "#ffd54f",
          };
          return relColors[d?.relation ?? ""] ?? "#666";
        })
        .attr("stroke-width", (d) => {
          const priorityRels = ["CtD", "CbG", "CpD"];
          return priorityRels.includes(d?.relation ?? "") ? 1.8 : 1.2;
        })
        .attr("stroke-opacity", (d) => {
          const priorityRels = ["CtD", "CbG", "CpD"];
          return priorityRels.includes(d?.relation ?? "") ? 0.8 : 0.5;
        })
        .attr("marker-end", "url(#arrow)");

      // Nodes with improved visual hierarchy
      const node = g
        .append("g")
        .selectAll<SVGGElement, SimNode>("g")
        .data(simNodes)
        .enter()
        .append("g")
        .style("cursor", "pointer")
        .call(
          d3
            .drag<SVGGElement, SimNode>()
            .on("start", (e, d) => {
              if (!d) return;
              if (!e.active) sim.alphaTarget(0.3).restart();
              d.fx = d.x;
              d.fy = d.y;
            })
            .on("drag", (e, d) => {
              if (!d) return;
              d.fx = e.x;
              d.fy = e.y;
            })
            .on("end", (e, d) => {
              if (!d) return;
              if (!e.active) sim.alphaTarget(0);
              if (d.id !== center) {
                d.fx = null;
                d.fy = null;
              }
            }),
        );

      // Double-click to re-center on a node
      node.on("dblclick", (_e, d) => {
        if (d?.id) {
          setEntityInput(d.id);
          loadRef.current(d.id);
        }
      });

      // Node circles with glow effect for center
      node
        .append("circle")
        .attr("r", (d) => (d?.id === center ? 16 : 11))
        .attr("fill", (d) => KG_COLORS[d?.entity_type ?? ""] ?? "#909097")
        .attr("stroke", (d) => (d?.id === center ? "#fff" : "#0b1326"))
        .attr("stroke-width", (d) => (d?.id === center ? 2.5 : 1.5))
        .attr("filter", (d) => (d?.id === center ? "drop-shadow(0 0 4px rgba(255,255,255,0.6))" : "none"));

      // Node labels with better text truncation
      node
        .append("text")
        .text((d) => {
          const lab = d?.label ?? "";
          const maxLen = d?.id === center ? 20 : 14;
          return lab.length > maxLen ? lab.slice(0, maxLen - 1) + "\u2026" : lab;
        })
        .attr("dy", (d) => (d?.id === center ? -20 : -16))
        .attr("text-anchor", "middle")
        .attr("fill", "#c6c6cd")
        .attr("font-size", (d) => (d?.id === center ? 12 : 10))
        .attr("font-weight", (d) => (d?.id === center ? "600" : "400"))
        .attr("pointer-events", "none")
        .attr("text-shadow", "1px 1px 2px rgba(0,0,0,0.8)");

      // Tooltip on hover with more details
      node
        .append("title")
        .text((d) => 
          d?.id ? `${d.label}\nType: ${d.entity_type}\nID: ${d.id}\n\nDouble-click to explore` : ""
        );
      
      link
        .append("title")
        .text((d) => (d && d.relation ? d.relation : ""));

      // Animation on tick with null safety
      sim.on("tick", () => {
        link
          // eslint-disable-next-line
          .attr("x1", (d: any) => d.source?.x ?? 0)
          // eslint-disable-next-line
          .attr("y1", (d: any) => d.source?.y ?? 0)
          // eslint-disable-next-line
          .attr("x2", (d: any) => d.target?.x ?? 0)
          // eslint-disable-next-line
          .attr("y2", (d: any) => d.target?.y ?? 0);

        node.attr("transform", (d) => d?.x != null && d?.y != null ? `translate(${d.x},${d.y})` : "");
      });

      // Stop simulation after stabilization
      setTimeout(() => {
        if (!destroyed) sim.alpha(0).stop();
      }, 3000);
        } catch (err) {
          setError(err instanceof Error ? err.message : String(err));
        }
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : String(err));
      });

    return () => {
      destroyed = true;
    };
  }, [nodes, links, center]);

  return (
    <div className="space-y-3">
      {/* Controls row */}
      <form
        className="flex flex-wrap items-end gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          load(entityInput);
        }}
      >
        <div className="w-full min-w-[200px] sm:w-auto sm:min-w-[220px] sm:max-w-[min(100%,320px)]">
          <label className="mb-1 block text-xs text-on-surface-variant">
            Compound (from pipeline)
          </label>
          <select
            value={compoundSelectValue}
            disabled={compoundListLoading || sortedCompounds.length === 0}
            onChange={(e) => {
              const v = e.target.value;
              if (!v) return;
              setEntityInput(v);
              load(v);
            }}
            className="w-full rounded-md bg-surface-container-lowest px-2 py-2 text-sm text-on-surface outline-none ring-1 ring-outline-variant/30 focus:ring-primary/60 disabled:cursor-not-allowed disabled:opacity-60"
            aria-label="Select a compound to center the knowledge graph"
          >
            <option value="">
              {compoundListLoading
                ? "Loading compounds…"
                : sortedCompounds.length === 0
                  ? "No compounds in list"
                  : "— Choose compound —"}
            </option>
            {sortedCompounds.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name ?? c.id}
              </option>
            ))}
          </select>
          {compoundListHint && !compoundListLoading && (
            <p className="mt-1 text-[10px] text-on-surface-variant/90">{compoundListHint}</p>
          )}
        </div>

        <div className="relative min-w-[200px] flex-1">
          <label className="mb-1 block text-xs text-on-surface-variant">
            Center entity (search by name or ID)
          </label>
          <input
            value={entityInput}
            onChange={(e) => onSearchInput(e.target.value)}
            onFocus={() => searchResults.length > 0 && setShowSearch(true)}
            onBlur={() => setTimeout(() => setShowSearch(false), 200)}
            className="w-full rounded-md bg-surface-container-lowest px-3 py-2 text-sm text-on-surface outline-none ring-1 ring-outline-variant/30 focus:ring-primary/60"
            placeholder="Type a name… e.g. Metformin, Alzheimer"
          />
          {showSearch && (
            <div className="absolute left-0 right-0 top-full z-20 mt-1 max-h-48 overflow-y-auto rounded-md border border-outline-variant/20 bg-surface-container shadow-lg">
              {searchResults.filter((r) => r != null && r.id).map((r) => (
                <button
                  key={r.id}
                  type="button"
                  className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm hover:bg-primary/10"
                  onMouseDown={() => { setEntityInput(r.id); setShowSearch(false); load(r.id); }}
                >
                  <span
                    className="inline-block h-2 w-2 rounded-full flex-shrink-0"
                    style={{ background: KG_COLORS[r.kind] ?? "#909097" }}
                  />
                  <span className="text-on-surface truncate">{r.name}</span>
                  <span className="ml-auto text-[10px] text-on-surface-variant">{r.kind}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="w-20">
          <label className="mb-1 block text-xs text-on-surface-variant">Hops</label>
          <select
            value={hops}
            onChange={(e) => setHops(Number(e.target.value))}
            className="w-full rounded-md bg-surface-container-lowest px-2 py-2 text-sm text-on-surface outline-none ring-1 ring-outline-variant/30 focus:ring-primary/60"
          >
            <option value={1}>1</option>
            <option value={2}>2</option>
            <option value={3}>3</option>
          </select>
        </div>

        <div className="w-24">
          <label className="mb-1 block text-xs text-on-surface-variant">Max nodes</label>
          <select
            value={maxNodes}
            onChange={(e) => setMaxNodes(Number(e.target.value))}
            className="w-full rounded-md bg-surface-container-lowest px-2 py-2 text-sm text-on-surface outline-none ring-1 ring-outline-variant/30 focus:ring-primary/60"
          >
            <option value={50}>50</option>
            <option value={80}>80</option>
            <option value={120}>120</option>
            <option value={200}>200</option>
          </select>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="rounded-md bg-primary/15 px-4 py-2 text-sm font-medium text-primary hover:bg-primary/25 disabled:opacity-50"
        >
          {loading ? "Loading\u2026" : "Explore"}
        </button>
      </form>

      {loading ? (
        <LoadingMsg text="Building subgraph\u2026" />
      ) : error ? (
        <ErrorMsg msg={error} />
      ) : (
        <>
          <p className="text-xs text-on-surface-variant">
            {nodes.length} nodes &middot; {links.length} edges &middot;
            centered on <strong className="text-on-surface">{center}</strong>
            {hops > 1 && ` (${hops}-hop)`}
            <span className="ml-3 text-on-surface-variant/60">Double-click a node to re-center</span>
          </p>
          <div className="overflow-hidden rounded-lg border border-outline-variant/15 bg-surface-container-lowest/40">
            <svg ref={svgRef} className="h-[520px] w-full" />
            <div className="flex flex-wrap gap-3 border-t border-outline-variant/10 px-3 py-2">
              {Object.entries(KG_COLORS).map(([kind, color]) => (
                <span key={kind} className="flex items-center gap-1.5 text-xs text-on-surface-variant">
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-full"
                    style={{ background: color }}
                  />
                  {kind}
                </span>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// 3. Feature Vector Comparison
// ---------------------------------------------------------------------------

// Mini bar chart for a single embedding dimension group
function EmbedBars({
  values,
  color,
  label,
  note,
}: {
  values: number[];
  color: string;
  label: string;
  note?: string;
}) {
  const max = Math.max(...values.map(Math.abs), 1e-9);
  return (
    <div className="space-y-1">
      <div className="flex items-baseline gap-2">
        <span className="text-xs font-semibold text-on-surface">{label}</span>
        {note && <span className="text-[10px] text-on-surface-variant/70">{note}</span>}
      </div>
      <div className="flex gap-px items-end h-10">
        {values.map((v, i) => {
          const pct = Math.abs(v) / max;
          const isNeg = v < 0;
          return (
            <div
              key={i}
              title={`dim ${i + 1}: ${v.toFixed(4)}`}
              className="flex-1 rounded-t-sm transition-all"
              style={{
                height: `${Math.max(pct * 100, 4)}%`,
                backgroundColor: isNeg ? `${color}88` : color,
                opacity: 0.85 + 0.15 * pct,
              }}
            />
          );
        })}
      </div>
      <div className="flex justify-between text-[9px] text-on-surface-variant/50">
        <span>dim 1</span>
        <span>dim {values.length}</span>
      </div>
    </div>
  );
}

function EmbeddingTab() {
  const [drug, setDrug] = useState("pindolol");
  const [disease, setDisease] = useState("hypertension");
  const [data, setData] = useState<EmbeddingVectorResponse | null>(null);
  const [probability, setProbability] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function compare(e?: React.FormEvent) {
    e?.preventDefault();
    setLoading(true);
    setError(null);
    setData(null);
    setProbability(null);
    try {
      const [vec, pred] = await Promise.all([
        fetchVizEmbeddingVector(drug.trim(), disease.trim()),
        predictLink({ drug: drug.trim(), disease: disease.trim(), method: "classical" }),
      ]);
      if (vec.status === "error") throw new Error(vec.message ?? "Failed");
      setData(vec);
      if (pred.status === "success") setProbability(pred.link_probability);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  // Load default pair on mount
  useEffect(() => { compare(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const scoreColor =
    probability == null ? "text-on-surface-variant"
    : probability >= 0.70 ? "text-tertiary"
    : probability >= 0.40 ? "text-secondary"
    : "text-error";

  const GROUPS = data
    ? [
        { values: data.drug_embedding,     color: "#7bd0ff", label: `h  (${data.drug_name})`,     note: "drug RotatE vector" },
        { values: data.disease_embedding,  color: "#ff8a65", label: `t  (${data.disease_name})`,  note: "disease RotatE vector" },
        { values: data.abs_diff,           color: "#ffd54f", label: "|h − t|",                    note: "absolute difference" },
        { values: data.hadamard_product,   color: "#69f0ae", label: "h ⊙ t",                      note: "element-wise product" },
      ]
    : [];

  const inTraining = data?.in_training_set;

  return (
    <div className="space-y-5">
      {/* Input form */}
      <form onSubmit={compare} className="flex flex-wrap items-end gap-3">
        <div>
          <label className="mb-1 block text-xs text-on-surface-variant">Drug / compound</label>
          <input
            value={drug}
            onChange={(e) => setDrug(e.target.value)}
            placeholder="e.g. pindolol"
            className="rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface placeholder:text-on-surface-variant/50 focus:border-primary focus:outline-none"
          />
        </div>
        <div>
          <label className="mb-1 block text-xs text-on-surface-variant">Disease</label>
          <input
            value={disease}
            onChange={(e) => setDisease(e.target.value)}
            placeholder="e.g. hypertension"
            className="rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface placeholder:text-on-surface-variant/50 focus:border-primary focus:outline-none"
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          className="primary-gradient rounded-lg px-5 py-2 text-sm font-semibold text-on-primary shadow-glow disabled:opacity-50"
        >
          {loading ? "Computing…" : "Compare"}
        </button>
      </form>

      {error && <ErrorMsg msg={error} />}
      {loading && <LoadingMsg text="Fetching embedding vectors…" />}

      {data && !loading && (
        <div className="space-y-6">

          {/* Training-set coverage warning */}
          {(inTraining && (!inTraining.drug || !inTraining.disease)) && (
            <div className="rounded-md border border-amber-400/30 bg-amber-400/10 px-3 py-2 text-xs text-amber-200">
              {!inTraining.drug && <span className="block">⚠ <strong>{data.drug_name}</strong> is not in the current embedding matrix — using a deterministic fallback vector. Run the full-graph pipeline for coverage.</span>}
              {!inTraining.disease && <span className="block">⚠ <strong>{data.disease_name}</strong> is not in the current embedding matrix — using a deterministic fallback vector.</span>}
            </div>
          )}

          {/* Score */}
          {probability != null && (
            <div className="flex items-center gap-4 rounded-xl border border-outline/10 bg-surface-container-lowest/60 px-5 py-4">
              <div>
                <p className="text-xs uppercase tracking-wide text-on-surface-variant">Link probability</p>
                <p className={`text-3xl font-semibold ${scoreColor}`}>
                  {(probability * 100).toFixed(1)}%
                </p>
              </div>
              <div className="text-xs text-on-surface-variant space-y-0.5">
                <p><span className="text-tertiary font-medium">≥ 70%</span> — strong structural evidence</p>
                <p><span className="text-secondary font-medium">40–70%</span> — moderate signal</p>
                <p><span className="text-on-surface-variant/70">&lt; 40%</span> — weak / no known relation</p>
              </div>
            </div>
          )}

          {/* Feature vector breakdown */}
          <div>
            <h3 className="mb-1 font-headline text-sm font-semibold text-on-surface">
              Feature vector breakdown — {data.qml_dim}-dim RotatE embeddings → {data.qml_dim * 4} features
            </h3>
            <p className="mb-4 text-xs text-on-surface-variant">
              The model receives the concatenation <code className="rounded bg-surface-container px-1 py-0.5 text-[10px] text-primary">[h, t, |h−t|, h⊙t]</code> as its input.
              Each bar is one embedding dimension. Height = magnitude; muted shade = negative value.
            </p>
            <div className="grid gap-5 sm:grid-cols-2">
              {GROUPS.map((g) => (
                <EmbedBars key={g.label} values={g.values} color={g.color} label={g.label} note={g.note} />
              ))}
            </div>
          </div>

          {/* Resolved IDs */}
          <p className="text-[10px] text-on-surface-variant/60">
            Resolved: <code className="text-on-surface/70">{data.drug_id}</code> → <code className="text-on-surface/70">{data.disease_id}</code>
          </p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// 4. Quantum Circuit (Canvas 2D)
// ---------------------------------------------------------------------------

function CircuitTab() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [config, setConfig] = useState<VizCircuitResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let c = false;
    fetchVizCircuitParams()
      .then((r) => {
        if (!c) setConfig(r);
      })
      .catch((e) => {
        if (!c) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!c) setLoading(false);
      });
    return () => { c = true; };
  }, []);

  useEffect(() => {
    if (!config || !canvasRef.current) return;
    drawCircuit(canvasRef.current, config);
  }, [config]);

  if (loading) return <LoadingMsg text="Loading circuit params\u2026" />;
  if (error) return <ErrorMsg msg={error} />;

  const nq = config?.n_qubits ?? 12;
  const nReps = config?.n_reps ?? 2;

  return (
    <div className="space-y-6">

      {/* ── How the circuit compares a drug to a disease ── */}
      <section className="space-y-3">
        <h2 className="font-headline text-base font-semibold text-on-surface">
          How the quantum circuit compares a compound to a disease
        </h2>

        {/* Data-flow pipeline */}
        <ol className="grid gap-3 sm:grid-cols-4">
          {[
            {
              step: "01",
              title: "PCA projection",
              body: `The 128-dim RotatE embedding of the drug (or disease) is reduced to ${nq} principal components. Each component becomes one qubit's rotation angle φᵢ.`,
              color: "text-[#7bd0ff]",
            },
            {
              step: "02",
              title: "Qubit encoding",
              body: `A Hadamard gate puts each qubit into superposition. Then Rz(2φᵢ) rotates it by twice the embedding value — encoding the drug's position in graph space into a quantum state |ψ(drug)⟩.`,
              color: "text-[#9b59b6]",
            },
            {
              step: "03",
              title: "ZZ entanglement",
              body: `ZZ-interaction gates between qubit pairs compute cross-products φᵢ·φⱼ. This captures correlations between embedding dimensions that a simple dot-product misses — the circuit runs ${nReps} rep${nReps !== 1 ? "s" : ""}.`,
              color: "text-[#3cddc7]",
            },
            {
              step: "04",
              title: "Quantum kernel",
              body: `The circuit is run twice — once for the drug, once for the disease — producing |ψ(drug)⟩ and |ψ(disease)⟩. The kernel K = |⟨ψ(drug)|ψ(disease)⟩|² measures similarity in quantum feature space. QSVC uses this kernel as its decision boundary.`,
              color: "text-[#69f0ae]",
            },
          ].map(({ step, title, body, color }) => (
            <li key={step} className="rounded-xl border border-outline/10 bg-surface-container-lowest/60 p-4 space-y-1">
              <p className={`font-mono text-xs font-bold opacity-70 ${color}`}>{step}</p>
              <p className="text-sm font-semibold text-on-surface">{title}</p>
              <p className="text-xs leading-relaxed text-on-surface-variant">{body}</p>
            </li>
          ))}
        </ol>
      </section>

      {/* ── Classical vs Quantum comparison ── */}
      <section className="rounded-xl border border-outline/10 bg-surface-container-lowest/60 p-5 space-y-3">
        <h2 className="font-headline text-base font-semibold text-on-surface">
          Classical vs quantum — what changes and what doesn&apos;t
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-outline/10 text-left text-on-surface-variant">
                <th className="pb-2 pr-4 font-medium">Aspect</th>
                <th className="pb-2 pr-4 font-medium text-secondary">Classical (LR / ExtraTrees)</th>
                <th className="pb-2 font-medium text-[#3cddc7]">Quantum (QSVC)</th>
              </tr>
            </thead>
            <tbody className="text-on-surface-variant">
              {[
                ["Input features", `${nq * 4} values — [h, t, |h−t|, h⊙t]`, `${nq} PCA dims each for drug and disease`],
                ["Similarity measure", "Dot product / tree split", `Quantum kernel K(drug, disease) = |⟨ψ(drug)|ψ(disease)⟩|²`],
                ["Cross-dim correlations", "Captured only by products in feature set", `Captured by ZZ gates — φᵢ·φⱼ for all pairs`],
                ["Decision boundary", "Linear (LR) or piecewise (trees)", "Kernel SVM in Hilbert space — non-linear"],
                ["Current PR-AUC (CtD)", "0.81 (ExtraTrees, 2317 features)", "0.80 (QSVC, ensemble)"],
                ["Inference speed", "Microseconds", "Seconds (simulator) / minutes (hardware)"],
                ["Served in API", "Yes — classical_serving.joblib", "No — not yet saved/loaded"],
              ].map(([aspect, classical, quantum]) => (
                <tr key={aspect} className="border-b border-outline/10">
                  <td className="py-2 pr-4 font-medium text-on-surface">{aspect}</td>
                  <td className="py-2 pr-4">{classical}</td>
                  <td className="py-2 text-[#3cddc7]/90">{quantum}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-[11px] text-on-surface-variant/70 pt-1">
          The quantum advantage is theoretical at this scale — {nq} qubits on a simulator. On real hardware, decoherence limits circuit depth. The value is in the kernel: ZZ entanglement captures embedding-dimension co-activations that linear features approximate but quantum circuits compute exactly.
        </p>
      </section>

      {/* ── Circuit diagram ── */}
      <section className="space-y-2">
        <h2 className="font-headline text-base font-semibold text-on-surface">
          Circuit diagram ({nq} qubits, {nReps} rep{nReps !== 1 ? "s" : ""})
        </h2>

        {/* Parameter badges */}
        <div className="flex flex-wrap gap-2">
          {[
            { label: "Qubits", value: config?.n_qubits },
            { label: "Reps", value: config?.n_reps },
            { label: "Feature map", value: config?.feature_map },
            { label: "Entanglement", value: config?.entanglement },
            { label: "Mode", value: config?.execution_mode ?? "simulator" },
            { label: "Backend", value: config?.backend },
            { label: "Shots", value: config?.shots },
          ]
            .filter((b) => b.value != null)
            .map((b) => (
              <span
                key={b.label}
                className="inline-flex items-center gap-1.5 rounded-md bg-surface-container-high/60 px-2.5 py-1 text-xs"
              >
                <span className="text-on-surface-variant">{b.label}:</span>
                <strong className="text-on-surface">{b.value}</strong>
              </span>
            ))}
        </div>

        <p className="text-xs text-on-surface-variant">
          Each horizontal wire = one qubit (one PCA component of the embedding).
          Blue = Hadamard (superposition), purple = Rz rotation (encodes φᵢ), teal dot = ZZ entanglement (encodes φᵢ·φⱼ), orange = measurement.
        </p>

        <div className="overflow-x-auto rounded-lg border border-outline-variant/15 bg-surface-container-lowest/40 p-4">
          <canvas ref={canvasRef} className="mx-auto" style={{ maxWidth: "100%" }} />
        </div>

        <div className="flex flex-wrap gap-3 text-xs text-on-surface-variant">
          <span className="flex items-center gap-1.5"><span className="inline-block h-3 w-5 rounded-sm bg-[#4a9eff]" /> Hadamard — superposition</span>
          <span className="flex items-center gap-1.5"><span className="inline-block h-3 w-5 rounded-sm bg-[#9b59b6]" /> Rz(2φᵢ) — encodes one embedding dim</span>
          <span className="flex items-center gap-1.5"><span className="inline-block h-3 w-3 rounded-full bg-[#3cddc7]" /> ZZ — captures φᵢ·φⱼ cross-correlation</span>
          <span className="flex items-center gap-1.5"><span className="inline-block h-3 w-5 rounded-sm bg-[#e67e22]" /> Measure — collapses to classical bit</span>
        </div>
      </section>
    </div>
  );
}

function drawCircuit(canvas: HTMLCanvasElement, cfg: VizCircuitResponse) {
  const nq = Math.min(cfg.n_qubits, 16); // cap visual
  const nReps = cfg.n_reps;
  const wireSpacing = 50;
  const leftPad = 70;
  const topPad = 40;
  const gapX = 70;
  const hasEntanglement = cfg.feature_map !== "ZFeatureMap";

  // Width: init|0⟩ + H + (Rz + ZZ?) * nReps + measurement + padding
  const repWidth = hasEntanglement ? gapX * 2 + 20 : gapX + 20;
  const totalW = leftPad + gapX + repWidth * nReps + gapX + 60; // +gapX for measurement
  const totalH = topPad + nq * wireSpacing + 30;

  canvas.width = totalW * 2; // hi-dpi
  canvas.height = totalH * 2;
  canvas.style.width = `${totalW}px`;
  canvas.style.height = `${totalH}px`;
  const ctx = canvas.getContext("2d")!;
  ctx.scale(2, 2);
  ctx.clearRect(0, 0, totalW, totalH);

  // Qubit wires
  for (let q = 0; q < nq; q++) {
    const y = topPad + q * wireSpacing;
    ctx.strokeStyle = "#45464d";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(leftPad, y);
    ctx.lineTo(totalW - 20, y);
    ctx.stroke();

    // |0⟩ label
    ctx.fillStyle = "#c6c6cd";
    ctx.font = "12px monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(`|0\u27E9`, leftPad - 8, y);

    // qubit index
    ctx.fillStyle = "#909097";
    ctx.font = "9px monospace";
    ctx.fillText(`q${q}`, leftPad - 36, y);
  }

  let xPos = leftPad + gapX;

  // H gates
  for (let q = 0; q < nq; q++) {
    drawGateBox(ctx, xPos, topPad + q * wireSpacing, "H", "#4a9eff");
  }
  xPos += gapX;

  for (let rep = 0; rep < nReps; rep++) {
    // Rz gates (rotation layer)
    for (let q = 0; q < nq; q++) {
      const label = nq <= 8 ? `Rz(\u03c6${q})` : "Rz";
      drawGateBox(ctx, xPos, topPad + q * wireSpacing, label, "#9b59b6");
    }
    xPos += gapX;

    // ZZ entanglement (if applicable)
    if (hasEntanglement) {
      const ent = cfg.entanglement || "full";
      if (ent === "linear" || ent === "full") {
        // Linear: nearest neighbour, Full: we draw linear for visual clarity
        for (let q = 0; q < nq - 1; q++) {
          const y1 = topPad + q * wireSpacing;
          const y2 = topPad + (q + 1) * wireSpacing;
          ctx.strokeStyle = "#3cddc7";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(xPos, y1);
          ctx.lineTo(xPos, y2);
          ctx.stroke();
          [y1, y2].forEach((y) => {
            ctx.fillStyle = "#3cddc7";
            ctx.beginPath();
            ctx.arc(xPos, y, 4, 0, Math.PI * 2);
            ctx.fill();
          });
        }
        // Full entanglement: extra cross-connections shown as faint arcs
        if (ent === "full" && nq > 2 && nq <= 8) {
          ctx.strokeStyle = "rgba(60,221,199,0.25)";
          ctx.lineWidth = 0.8;
          for (let q = 0; q < nq - 2; q++) {
            const y1 = topPad + q * wireSpacing;
            const y3 = topPad + (q + 2) * wireSpacing;
            ctx.beginPath();
            ctx.moveTo(xPos + 12, y1);
            ctx.quadraticCurveTo(xPos + 24, (y1 + y3) / 2, xPos + 12, y3);
            ctx.stroke();
          }
        }
      }
      xPos += gapX;
    }

    // Barrier between reps
    if (rep < nReps - 1) {
      ctx.strokeStyle = "#666";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      const bx = xPos - gapX / 2;
      ctx.moveTo(bx, topPad - 15);
      ctx.lineTo(bx, topPad + (nq - 1) * wireSpacing + 15);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = "#909097";
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      ctx.fillText(`rep ${rep + 1}`, bx, topPad - 20);
    }
  }

  // Measurement gates
  for (let q = 0; q < nq; q++) {
    const y = topPad + q * wireSpacing;
    drawGateBox(ctx, xPos, y, "M", "#e67e22");
    // Meter arc inside
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(xPos, y + 2, 6, Math.PI, 0);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(xPos, y + 2);
    ctx.lineTo(xPos + 4, y - 5);
    ctx.stroke();
  }

  // Classical bits double-line below
  const clY = topPad + (nq - 1) * wireSpacing + 25;
  ctx.strokeStyle = "#666";
  ctx.lineWidth = 0.8;
  ctx.beginPath();
  ctx.moveTo(xPos - 5, clY);
  ctx.lineTo(totalW - 20, clY);
  ctx.moveTo(xPos - 5, clY + 3);
  ctx.lineTo(totalW - 20, clY + 3);
  ctx.stroke();
  ctx.fillStyle = "#909097";
  ctx.font = "9px monospace";
  ctx.textAlign = "left";
  ctx.fillText(`c[${nq}]`, totalW - 18, clY + 2);
}

function drawGateBox(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  label: string,
  color: string,
) {
  const w = 48;
  const h = 28;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(x - w / 2, y - h / 2, w, h, 3);
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.3)";
  ctx.lineWidth = 0.5;
  ctx.stroke();
  ctx.fillStyle = "#fff";
  ctx.font = "bold 10px monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x, y);
}

// ---------------------------------------------------------------------------
// 5. Model Comparison (Chart.js)
// ---------------------------------------------------------------------------

function ComparisonTab() {
  const praucRef = useRef<HTMLCanvasElement>(null);
  const scatterRef = useRef<HTMLCanvasElement>(null);
  const ablationRef = useRef<HTMLCanvasElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const chartsRef = useRef<{ destroy(): void }[]>([]);
  useEffect(() => {
    let destroyed = false;

    (async () => {
      try {
        const [metrics, Chart] = await Promise.all([
          fetchVizModelMetrics(),
          import("chart.js/auto").then((m) => m.default),
        ]);
        if (destroyed) return;
        if (metrics.status !== "ok") throw new Error(metrics.message ?? "No data");
        const safeModels = (metrics.models ?? []).filter(
          (m): m is VizModelMetric =>
            m != null && typeof m.name === "string" && m.name.length > 0,
        );
        if (safeModels.length === 0) {
          throw new Error(
            "No model rows in latest pipeline results (empty ranking). Run the pipeline or check results JSON.",
          );
        }

        if (!praucRef.current || !scatterRef.current || !ablationRef.current) {
          throw new Error("Chart canvas not mounted");
        }

        renderCharts(
          Chart,
          safeModels,
          metrics.ablation ?? {},
          praucRef.current,
          scatterRef.current,
          ablationRef.current,
          chartsRef,
        );
      } catch (e) {
        if (!destroyed)
          setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!destroyed) setLoading(false);
      }
    })();

    return () => {
      destroyed = true;
      chartsRef.current.forEach((c) => c?.destroy());
      chartsRef.current = [];
    };
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  return (
    <div className="relative space-y-4">
      {/* Canvases must stay mounted while loading — otherwise refs are null when fetch completes */}
      {loading && (
        <div
          className="absolute inset-0 z-10 flex items-center justify-center rounded-lg bg-background/70 backdrop-blur-sm"
          role="status"
        >
          <LoadingMsg text="Loading model metrics\u2026" />
        </div>
      )}
      {/* Info banner */}
      <div className="rounded-lg border border-outline-variant/20 bg-surface-container-high/40 p-3">
        <p className="text-xs text-on-surface-variant">
          Model comparison from the latest pipeline run. Models are grouped by type: 
          <span className="ml-1 inline-flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full bg-[#7bd0ff]" /> Classical
          </span>
          <span className="ml-2 inline-flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full bg-[#d0bcff]" /> Quantum
          </span>
          <span className="ml-2 inline-flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full bg-[#ff8a65]" /> Ensemble
          </span>
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="flex flex-col rounded-lg border border-outline-variant/15 bg-surface-container-lowest/40 p-4">
          <h3 className="mb-3 text-sm font-medium text-on-surface">
            PR-AUC by Model
          </h3>
          <div className="min-h-[280px] flex-1">
            <canvas ref={praucRef} />
          </div>
        </div>
        <div className="flex flex-col rounded-lg border border-outline-variant/15 bg-surface-container-lowest/40 p-4">
          <h3 className="mb-3 text-sm font-medium text-on-surface">
            Model Performance Map
          </h3>
          <div className="min-h-[280px] flex-1">
            <canvas ref={scatterRef} />
          </div>
        </div>
        <div className="flex flex-col rounded-lg border border-outline-variant/15 bg-surface-container-lowest/40 p-4 lg:col-span-2">
          <h3 className="mb-3 text-sm font-medium text-on-surface">
            Category Comparison
          </h3>
          <div className="min-h-[200px] flex-1">
            <canvas ref={ablationRef} />
          </div>
        </div>
      </div>
    </div>
  );
}

function renderCharts(
  // eslint-disable-next-line
  Chart: any,
  models: VizModelMetric[],
  ablation: Record<string, number>,
  praucCanvas: HTMLCanvasElement,
  scatterCanvas: HTMLCanvasElement,
  ablationCanvas: HTMLCanvasElement,
  chartsRef: React.MutableRefObject<{ destroy(): void }[]>,
) {
  const typeColor = (t: string) =>
    t === "quantum" ? "#d0bcff" : t === "ensemble" ? "#ff8a65" : "#7bd0ff";

  const typeLabel = (t: string) =>
    t === "quantum" ? "Quantum" : t === "ensemble" ? "Ensemble" : "Classical";

  const safeModels = models.filter(
    (m): m is VizModelMetric =>
      m != null &&
      typeof m.name === "string" &&
      typeof m.type === "string" &&
      typeof m.pr_auc === "number" &&
      typeof m.accuracy === "number",
  );

  // Sort models by PR-AUC for better visualization
  const sortedModels = [...safeModels].sort((a, b) => b.pr_auc - a.pr_auc);

  // 1. PR-AUC bar chart with improved styling
  chartsRef.current.push(
    new Chart(praucCanvas, {
      type: "bar",
      data: {
        labels: sortedModels.map((m) => m.name),
        datasets: [
          {
            label: "PR-AUC",
            data: sortedModels.map((m) => m.pr_auc),
            backgroundColor: sortedModels.map((m) => typeColor(m.type)),
            borderRadius: 4,
            borderSkipped: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx: any) => {
                const model = sortedModels[ctx.dataIndex];
                if (!model) return [];
                return [
                  `PR-AUC: ${model.pr_auc.toFixed(4)}`,
                  `Accuracy: ${model.accuracy.toFixed(4)}`,
                  `Type: ${typeLabel(model.type)}`,
                  `Fit time: ${model.fit_time.toFixed(3)}s`,
                ];
              },
            },
          },
        },
        scales: {
          y: {
            min: 0,
            max: 1,
            title: { display: true, text: "PR-AUC (higher is better)", color: "#909097" },
            ticks: { 
              color: "#909097",
              callback: (val: any) => `${(val * 100).toFixed(0)}%`,
            },
            grid: { color: "rgba(144,144,151,0.15)" },
          },
          x: {
            ticks: { 
              color: "#c6c6cd", 
              maxRotation: 45,
              minRotation: 45,
              font: { size: 10 },
            },
            grid: { display: false },
          },
        },
      },
    }),
  );

  // 2. Model comparison scatter plot (PR-AUC vs Accuracy)
  // Group by type for better legend
  const modelTypes = Array.from(new Set(safeModels.map((m) => m.type)));
  const datasets = modelTypes.map((type) => ({
    label: typeLabel(type),
    data: safeModels
      .filter((m) => m.type === type)
      .map((m) => ({ x: m.pr_auc, y: m.accuracy, name: m.name })),
    pointRadius: 9,
    pointHoverRadius: 13,
    backgroundColor: typeColor(type),
    borderColor: typeColor(type),
    pointBorderWidth: 2,
    pointBorderColor: "#0b1326",
  }));

  chartsRef.current.push(
    new Chart(scatterCanvas, {
      type: "scatter",
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { 
            labels: { 
              color: "#c6c6cd", 
              boxWidth: 12,
              font: { size: 11 },
            },
          },
          tooltip: {
            callbacks: {
              title: (items: any[]) => {
                const point = items?.[0]?.raw as { name?: string } | undefined;
                return point?.name ?? "";
              },
              label: (ctx: any) => {
                const point = ctx?.raw as { x?: number; y?: number } | undefined;
                if (point == null || point.x == null || point.y == null) return "";
                return [
                  `PR-AUC: ${point.x.toFixed(4)}`,
                  `Accuracy: ${point.y.toFixed(4)}`,
                ];
              },
            },
          },
        },
        scales: {
          x: {
            min: 0,
            max: 1,
            title: { display: true, text: "PR-AUC", color: "#909097" },
            ticks: { 
              color: "#909097",
              callback: (val: any) => `${(val * 100).toFixed(0)}%`,
            },
            grid: { color: "rgba(144,144,151,0.15)" },
          },
          y: {
            min: 0,
            max: 1,
            title: { display: true, text: "Accuracy", color: "#909097" },
            ticks: { 
              color: "#909097",
              callback: (val: any) => `${(val * 100).toFixed(0)}%`,
            },
            grid: { color: "rgba(144,144,151,0.15)" },
          },
        },
      },
    }),
  );

  // 3. Ablation / Category comparison
  const ablLabels = Object.keys(ablation);
  const ablValues = Object.values(ablation);
  if (ablLabels.length > 0) {
    chartsRef.current.push(
      new Chart(ablationCanvas, {
        type: "bar",
        data: {
          labels: ablLabels,
          datasets: [
            {
              label: "PR-AUC",
              data: ablValues,
              backgroundColor: ablLabels.map((label, i) => {
                if (label.includes("Full") || label.includes("full")) return "#3cddc7";
                if (label.includes("classical") || label.includes("Classical")) return "#7bd0ff";
                if (label.includes("quantum") || label.includes("Quantum")) return "#d0bcff";
                return i === 0 ? "#3cddc7" : "#ff8a65";
              }),
              borderRadius: 4,
              borderSkipped: false,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          indexAxis: "y" as const,
          plugins: { 
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx: any) => `PR-AUC: ${(ctx.raw * 100).toFixed(2)}%`,
              },
            },
          },
          scales: {
            x: {
              min: 0,
              max: 1,
              title: { display: true, text: "PR-AUC (higher is better)", color: "#909097" },
              ticks: { 
                color: "#909097",
                callback: (val: any) => `${(val * 100).toFixed(0)}%`,
              },
              grid: { color: "rgba(144,144,151,0.15)" },
            },
            y: {
              ticks: { 
                color: "#c6c6cd",
                font: { size: 10 },
              },
              grid: { display: false },
            },
          },
        },
      }),
    );
  }
}

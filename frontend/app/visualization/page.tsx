"use client";

import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import {
  getApiBaseUrl,
  fetchVizRunPredictions,
  fetchVizPredictions,
  fetchVizMolecule,
  fetchVizEmbeddings,
  fetchVizKGSubgraph,
  fetchVizModelMetrics,
  fetchVizCircuitParams,
  searchKGEntities,
} from "@/lib/api";
import type {
  VizRunPrediction,
  VizAtom,
  VizBond,
  VizEmbNode,
  VizEmbEdge,
  VizEmbeddingsResponse,
  EmbeddingProjection,
  VizKGNode,
  VizKGLink,
  KGSearchResult,
  VizModelMetric,
  VizCircuitResponse,
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
  { id: "embedding", label: "Embedding Space", icon: "scatter_plot" },
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
// 3. Embedding Space (Three.js 3D scatter)
// ---------------------------------------------------------------------------

function EmbeddingTab() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState<string>(""); // empty = auto
  const [projection, setProjection] = useState<EmbeddingProjection>("pca_stretch");
  const [showLabels, setShowLabels] = useState(false);
  const [meta, setMeta] = useState<VizEmbeddingsResponse | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);

  const showLabelsRef = useRef(showLabels);
  showLabelsRef.current = showLabels;

  const load = useCallback(
    (selectedModel?: string, proj?: EmbeddingProjection) => {
      const m = selectedModel ?? (model || undefined);
      const p = proj ?? projection;
      setLoading(true);
      setError(null);
      cleanupRef.current?.();
      cleanupRef.current = null;

      (async () => {
        try {
          const [data, THREE] = await Promise.all([
            fetchVizEmbeddings(m, p),
            import("three"),
          ]);
          if (!containerRef.current) return;
          if (data.status !== "ok") throw new Error(data.message ?? "Failed");

          setMeta(data);
          if (!model && data.model_name) setModel(data.model_name);

          cleanupRef.current = renderEmbedding3D(
            containerRef.current,
            THREE,
            data.nodes,
            data.edges,
            { showLabels: showLabelsRef.current, maxLabels: 36 },
          );
        } catch (e) {
          setError(e instanceof Error ? e.message : String(e));
        } finally {
          setLoading(false);
        }
      })();
    },
    [model, projection],
  );

  useEffect(() => {
    load(undefined, "pca_stretch");
    return () => { cleanupRef.current?.(); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Re-render Three.js when only label toggle changes (same data)
  useEffect(() => {
    if (loading || error || !meta || !containerRef.current) return;
    let cancelled = false;
    (async () => {
      try {
        cleanupRef.current?.();
        cleanupRef.current = null;
        const THREE = await import("three");
        if (cancelled || !containerRef.current) return;
        cleanupRef.current = renderEmbedding3D(
          containerRef.current,
          THREE,
          meta.nodes ?? [],
          meta.edges ?? [],
          { showLabels, maxLabels: 36 },
        );
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => { cancelled = true; };
  }, [showLabels]); // eslint-disable-line react-hooks/exhaustive-deps

  const onModelChange = (m: string) => {
    setModel(m);
    load(m, projection);
  };

  const onProjectionChange = (p: EmbeddingProjection) => {
    setProjection(p);
    load(model || undefined, p);
  };

  const lowLinearVariance =
    meta?.variance_explained &&
    meta.projection === "pca" &&
    meta.variance_explained.reduce((a, b) => a + b, 0) < 0.15;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end gap-3">
        {meta?.available_models && meta.available_models.length > 1 && (
          <div>
            <label className="mb-1 block text-xs text-on-surface-variant">Embedding model</label>
            <select
              value={model}
              onChange={(e) => onModelChange(e.target.value)}
              disabled={loading}
              className="rounded-md bg-surface-container-lowest px-3 py-2 text-sm text-on-surface outline-none ring-1 ring-outline-variant/30 focus:ring-primary/60 disabled:opacity-50"
            >
              {meta.available_models.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
        )}
        <div>
          <label className="mb-1 block text-xs text-on-surface-variant">3D projection</label>
          <select
            value={projection}
            onChange={(e) => onProjectionChange(e.target.value as EmbeddingProjection)}
            disabled={loading}
            className="rounded-md bg-surface-container-lowest px-3 py-2 text-sm text-on-surface outline-none ring-1 ring-outline-variant/30 focus:ring-primary/60 disabled:opacity-50"
            title="Stretched PCA spreads axes for a clearer view; t-SNE can separate clusters."
          >
            <option value="pca_stretch">PCA (stretched axes) — recommended</option>
            <option value="pca">PCA (raw scale)</option>
            <option value="tsne">t-SNE (slow, first load)</option>
          </select>
        </div>
        <label className="flex cursor-pointer items-center gap-2 pb-2 text-xs text-on-surface-variant">
          <input
            type="checkbox"
            checked={showLabels}
            onChange={(e) => setShowLabels(e.target.checked)}
            className="rounded border-outline-variant/40"
          />
          Show compound labels (sample)
        </label>
        {meta && !loading && (
          <div className="flex flex-wrap gap-4 text-xs text-on-surface-variant py-2">
            <span>{meta.compound_count} compounds</span>
            <span>{meta.disease_count} diseases</span>
            <span>{meta.edge_count} CtD edges</span>
            {meta.variance_explained && meta.projection !== "tsne" && (
              <span className="flex items-center gap-1">
                <span>
                  Linear PCA variance:{" "}
                  {meta.variance_explained.map((v) => `${(v * 100).toFixed(1)}%`).join(" / ")}
                </span>
                {lowLinearVariance && (
                  <span
                    className="rounded bg-amber-500/20 px-1.5 py-0.5 text-[10px] text-amber-200"
                    title="First 3 PCs capture little of total variance — try stretched PCA or t-SNE"
                  >
                    ⚠️ Low variance
                  </span>
                )}
              </span>
            )}
            {meta.projection === "tsne" && (
              <span className="text-on-surface-variant/80">t-SNE (nonlinear; no PCA %)</span>
            )}
          </div>
        )}
      </div>

      {meta?.projection_note && (
        <p className="rounded-md border border-outline-variant/20 bg-surface-container-high/50 px-3 py-2 text-[11px] leading-relaxed text-on-surface-variant">
          {meta.projection_note}
        </p>
      )}

      {loading && <LoadingMsg text="Projecting embeddings\u2026" />}
      {error && <ErrorMsg msg={error} />}
      <div
        ref={containerRef}
        className="relative h-[520px] w-full overflow-hidden rounded-lg border border-outline-variant/15 bg-surface-container-lowest/40"
        style={{ display: loading || error ? "none" : undefined }}
      />
      {!loading && !error && (
        <div className="space-y-2">
          <div className="flex flex-wrap gap-4 text-xs text-on-surface-variant">
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-[#7bd0ff]" />
              Compound
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-[#ff8a65]" />
              Disease
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-1 w-4 bg-[#3cddc7]" />
              CtD edge (sampled)
            </span>
            <span className="text-on-surface-variant/60">
              Drag to rotate &middot; Scroll to zoom
            </span>
          </div>
          {meta?.variance_explained && meta.projection !== "tsne" && (
            <p className="text-[11px] text-on-surface-variant/70">
              First three linear components explain{" "}
              {(meta.variance_explained.reduce((a, b) => a + b, 0) * 100).toFixed(1)}% of total
              variance — typical for 128D RotatE. Use <strong>stretched PCA</strong> or{" "}
              <strong>t-SNE</strong> for a clearer spatial layout.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function renderEmbedding3D(
  container: HTMLElement,
  T: typeof import("three"),
  nodes: VizEmbNode[],
  edges: VizEmbEdge[],
  opts?: { showLabels?: boolean; maxLabels?: number },
): () => void {
  const THREE = T; // local alias
  const showLabels = opts?.showLabels === true;
  const maxLabels = opts?.maxLabels ?? 32;
  const cleanNodes = (nodes ?? []).filter(
    (n): n is VizEmbNode => n != null && n.id != null && n.id !== "",
  );
  const edgeSet = new Set(cleanNodes.map((n) => n.id));
  const cleanEdges = (edges ?? []).filter(
    (e) =>
      e != null &&
      e.source != null &&
      e.target != null &&
      edgeSet.has(e.source) &&
      edgeSet.has(e.target),
  );
  const width = container.clientWidth;
  const height = container.clientHeight || 520;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x060e20);
  const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 200);

  // Lights
  scene.add(new THREE.AmbientLight(0x404060, 1.5));
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(5, 5, 5);
  scene.add(dir);

  const group = new THREE.Group();
  scene.add(group);

  // Calculate bounding box for better camera positioning
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
  cleanNodes.forEach((n) => {
    minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x);
    minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y);
    minZ = Math.min(minZ, n.z); maxZ = Math.max(maxZ, n.z);
  });
  if (cleanNodes.length === 0 || !Number.isFinite(minX)) {
    minX = maxX = minY = maxY = minZ = maxZ = 0;
  }

  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const centerZ = (minZ + maxZ) / 2;
  const sizeX = maxX - minX;
  const sizeY = maxY - minY;
  const sizeZ = maxZ - minZ;
  const maxDim = Math.max(sizeX, sizeY, sizeZ, 1);

  // Position camera based on data extent
  const cameraDistance = maxDim * 1.8;
  camera.position.set(0, 0, cameraDistance);

  // Nodes
  const nodeMap = new Map<string, import("three").Vector3>();
  const compColor = new THREE.Color(0x7bd0ff);
  const disColor = new THREE.Color(0xff8a65);

  // Adjust node size based on data density
  const nodeRadius = Math.max(0.08, maxDim * 0.03);

  let labelCount = 0;
  cleanNodes.forEach((n) => {
    const geo = new THREE.SphereGeometry(nodeRadius, 16, 16);
    const mat = new THREE.MeshPhongMaterial({
      color: n.type === "compound" ? compColor : disColor,
      shininess: 60,
    });
    const mesh = new THREE.Mesh(geo, mat);
    // Center the data
    mesh.position.set(n.x - centerX, n.y - centerY, n.z - centerZ);
    group.add(mesh);
    nodeMap.set(n.id, mesh.position);

    if (
      showLabels &&
      n.type === "compound" &&
      labelCount < maxLabels
    ) {
      labelCount += 1;
      const canvas = document.createElement("canvas");
      canvas.width = 256;
      canvas.height = 64;
      const ctx = canvas.getContext("2d")!;
      ctx.fillStyle = "#7bd0ff";
      ctx.font = "bold 22px monospace";
      const rawLabel = n.label ?? "";
      const label = rawLabel.length > 18 ? rawLabel.slice(0, 17) + "\u2026" : rawLabel;
      ctx.fillText(label, 4, 40);
      const tex = new THREE.CanvasTexture(canvas);
      const spriteMat = new THREE.SpriteMaterial({
        map: tex,
        transparent: true,
        opacity: 0.8,
      });
      const sprite = new THREE.Sprite(spriteMat);
      sprite.scale.set(Math.max(1.5, maxDim * 0.08), Math.max(0.4, maxDim * 0.02), 1);
      sprite.position.set(n.x - centerX, n.y - centerY + nodeRadius + 0.2, n.z - centerZ);
      group.add(sprite);
    }
  });

  // Edges - only draw a subset to avoid visual clutter
  const maxEdges = 200;
  const edgeStep = Math.max(1, Math.ceil(cleanEdges.length / maxEdges));
  const edgesToDraw = cleanEdges.filter((_, i) => i % edgeStep === 0);

  edgesToDraw.forEach((e) => {
    const src = nodeMap.get(e.source);
    const tgt = nodeMap.get(e.target);
    if (!src || !tgt) return;
    const pts = new Float32Array([src.x, src.y, src.z, tgt.x, tgt.y, tgt.z]);
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute("position", new THREE.BufferAttribute(pts, 3));
    const line = new THREE.Line(
      lineGeo,
      new THREE.LineBasicMaterial({ color: 0x3cddc7, transparent: true, opacity: 0.45 }),
    );
    group.add(line);
  });

  // Subtle grid floor
  const gridHelper = new THREE.GridHelper(maxDim * 1.5, 10, 0x333355, 0x222244);
  gridHelper.position.set(0, -maxDim * 0.6, 0);
  gridHelper.rotation.x = 0;
  scene.add(gridHelper);

  // Orbit controls (manual)
  let isDragging = false;
  let lastX = 0;
  let lastY = 0;
  let rotationX = 0;
  let rotationY = 0;

  const onDown = (e: PointerEvent) => {
    isDragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
  };
  const onMove = (e: PointerEvent) => {
    if (!isDragging) return;
    rotationY += (e.clientX - lastX) * 0.008;
    rotationX += (e.clientY - lastY) * 0.008;
    rotationX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotationX));
    group.rotation.x = rotationX;
    group.rotation.y = rotationY;
    lastX = e.clientX;
    lastY = e.clientY;
  };
  const onUp = () => {
    isDragging = false;
  };
  const onWheel = (e: WheelEvent) => {
    camera.position.z = Math.max(maxDim * 0.5, Math.min(maxDim * 5, camera.position.z + e.deltaY * 0.02));
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
    if (!isDragging) {
      group.rotation.y += 0.002;
    }
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

  return (
    <div className="space-y-3">
      {/* Parameter badges */}
      <div className="flex flex-wrap gap-3">
        {[
          { label: "Qubits", value: config?.n_qubits },
          { label: "Reps", value: config?.n_reps },
          { label: "Feature map", value: config?.feature_map },
          { label: "Entanglement", value: config?.entanglement },
          { label: "Mode", value: config?.execution_mode },
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

      {/* Circuit diagram description */}
      <p className="text-xs text-on-surface-variant">
        {config?.feature_map === "ZZFeatureMap" || config?.feature_map === "Pauli"
          ? "ZZFeatureMap: Hadamard layer \u2192 Rz(\u03c6) rotations \u2192 ZZ entangling \u2192 repeat \u2192 measurement"
          : config?.feature_map === "ZFeatureMap"
            ? "ZFeatureMap: Hadamard layer \u2192 Rz(\u03c6) rotations \u2192 repeat \u2192 measurement (no entanglement)"
            : `${config?.feature_map} feature map: encoding + variational layers \u2192 measurement`}
      </p>

      <div className="overflow-x-auto rounded-lg border border-outline-variant/15 bg-surface-container-lowest/40 p-4">
        <canvas
          ref={canvasRef}
          className="mx-auto"
          style={{ maxWidth: "100%" }}
        />
      </div>

      {/* Gate colour legend */}
      <div className="flex flex-wrap gap-3 text-xs text-on-surface-variant">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-5 rounded-sm bg-[#4a9eff]" /> Hadamard
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-5 rounded-sm bg-[#9b59b6]" /> Rotation (Rz)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-full bg-[#3cddc7]" /> Entangling (ZZ)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-5 rounded-sm bg-[#e67e22]" /> Measurement
        </span>
      </div>
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

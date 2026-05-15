"use client";

import { useEffect, useRef, useState } from "react";
import type { KGSearchResult, PredictionResponse } from "@/lib/api";
import { predictLink, searchKGEntities } from "@/lib/api";
import { Spinner } from "@/components/spinner";

export function PredictForm() {
  const [drug, setDrug] = useState("");
  const [disease, setDisease] = useState("");
  const [drugResolved, setDrugResolved] = useState<KGSearchResult | null>(null);
  const [diseaseResolved, setDiseaseResolved] = useState<KGSearchResult | null>(
    null,
  );
  const [drugWarning, setDrugWarning] = useState<string | null>(null);
  const [diseaseWarning, setDiseaseWarning] = useState<string | null>(null);
  const [method, setMethod] = useState("auto");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const drugParam = params.get("drug");
    const diseaseParam = params.get("disease");
    if (drugParam) setDrug(drugParam);
    if (diseaseParam) setDisease(diseaseParam);
  }, []);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const out = await predictLink({ drug, disease, method });
      setResult(out);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  const canSubmit =
    !loading && drug.trim().length > 0 && disease.trim().length > 0;

  return (
    <div className="space-y-6">
      <form onSubmit={onSubmit} className="space-y-4">
        <EntityAutocomplete
          label="Drug / compound"
          placeholder="e.g. Aspirin or DB00945"
          value={drug}
          onChange={(v) => {
            setDrug(v);
            setDrugResolved(null);
          }}
          onResolved={setDrugResolved}
          onWarning={setDrugWarning}
          kindFilter="compound"
          idPattern={/^DB\d{5,7}$/i}
          idHint="DrugBank IDs look like DB00945"
          resolved={drugResolved}
          warning={drugWarning}
        />
        <EntityAutocomplete
          label="Disease"
          placeholder="e.g. Diabetes or DOID_9352"
          value={disease}
          onChange={(v) => {
            setDisease(v);
            setDiseaseResolved(null);
          }}
          onResolved={setDiseaseResolved}
          onWarning={setDiseaseWarning}
          kindFilter="disease"
          idPattern={/^DOID[_:]?\d{3,7}$/i}
          idHint="Disease Ontology IDs look like DOID_9352"
          resolved={diseaseResolved}
          warning={diseaseWarning}
        />
        <div>
          <label className="block text-xs font-medium uppercase tracking-wide text-on-surface-variant">
            Method
          </label>
          <select
            className="mt-1 w-full rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
            value={method}
            onChange={(e) => setMethod(e.target.value)}
          >
            <option value="auto">auto</option>
            <option value="classical">classical</option>
            <option value="quantum">quantum</option>
          </select>
        </div>
        <button
          type="submit"
          disabled={!canSubmit}
          className="primary-gradient inline-flex items-center gap-2 rounded-lg px-5 py-2.5 text-sm font-semibold text-on-primary shadow-glow disabled:opacity-50"
        >
          {loading ? (
            <>
              <Spinner />
              <span>Resolving probability…</span>
            </>
          ) : (
            "Predict link"
          )}
        </button>
      </form>

      {error ? (
        <p className="text-sm text-error" role="alert">
          {error}
        </p>
      ) : null}

      {result && result.status === "success" ? (
        <div className="rounded-lg border border-tertiary/30 bg-surface-container-lowest/50 p-4">
          <p className="text-xs uppercase text-on-surface-variant">Link probability</p>
          <p className="text-2xl font-semibold text-tertiary">
            {(result.link_probability * 100).toFixed(2)}%
          </p>
          <p className="mt-2 text-xs text-on-surface-variant">
            Model: {result.model_used} · {result.drug_id} → {result.disease_id}
          </p>
        </div>
      ) : null}
    </div>
  );
}

interface EntityAutocompleteProps {
  label: string;
  placeholder: string;
  value: string;
  onChange: (v: string) => void;
  onResolved?: (r: KGSearchResult | null) => void;
  onWarning?: (w: string | null) => void;
  resolved?: KGSearchResult | null;
  warning?: string | null;
  kindFilter?: string;
  idPattern?: RegExp;
  idHint?: string;
}

function EntityAutocomplete({
  label,
  placeholder,
  value,
  onChange,
  onResolved,
  onWarning,
  resolved,
  warning,
  kindFilter,
  idPattern,
  idHint,
}: EntityAutocompleteProps) {
  const [suggestions, setSuggestions] = useState<KGSearchResult[]>([]);
  const [open, setOpen] = useState(false);
  const [activeIdx, setActiveIdx] = useState(-1);
  const [fetching, setFetching] = useState(false);
  const [searched, setSearched] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reqIdRef = useRef(0);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    const trimmed = value.trim();
    if (trimmed.length < 2) {
      setSuggestions([]);
      setSearched(false);
      onWarning?.(null);
      return;
    }

    // Format hint: if it looks like an ID but the format is wrong
    const looksLikeId = /^[A-Z]{2,}[_:]?\d+$/i.test(trimmed);
    if (looksLikeId && idPattern && !idPattern.test(trimmed)) {
      onWarning?.(idHint ?? "Unexpected ID format");
    } else {
      onWarning?.(null);
    }

    debounceRef.current = setTimeout(async () => {
      const reqId = ++reqIdRef.current;
      setFetching(true);
      try {
        const { results } = await searchKGEntities(trimmed, 8);
        if (reqId !== reqIdRef.current) return;
        const filtered = kindFilter
          ? results.filter(
              (r) => r.kind.toLowerCase() === kindFilter.toLowerCase(),
            )
          : results;
        const list = filtered.length > 0 ? filtered : results;
        setSuggestions(list);
        setSearched(true);

        // Auto-resolve exact match
        const lower = trimmed.toLowerCase();
        const exact = list.find(
          (r) =>
            r.id.toLowerCase() === lower || r.name.toLowerCase() === lower,
        );
        if (exact) {
          onResolved?.(exact);
          onWarning?.(null);
        } else {
          onResolved?.(null);
          if (list.length === 0) {
            onWarning?.(
              kindFilter
                ? `No matching ${kindFilter} found in the knowledge graph`
                : "No matching entity found",
            );
          }
        }
      } catch {
        setSuggestions([]);
      } finally {
        if (reqId === reqIdRef.current) setFetching(false);
      }
    }, 200);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [value, kindFilter, idPattern, idHint, onResolved, onWarning]);

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  function pick(item: KGSearchResult) {
    onChange(item.name || item.id);
    onResolved?.(item);
    onWarning?.(null);
    setOpen(false);
    setActiveIdx(-1);
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!open || suggestions.length === 0) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIdx((i) => Math.min(i + 1, suggestions.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && activeIdx >= 0) {
      e.preventDefault();
      pick(suggestions[activeIdx]);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  const showValid = resolved != null && !warning;
  const showWarning = warning != null && !fetching;
  const borderColor = showWarning
    ? "border-error/60 focus:border-error focus:ring-error"
    : showValid
      ? "border-tertiary/60 focus:border-tertiary focus:ring-tertiary"
      : "border-outline/20 focus:border-primary focus:ring-primary";

  return (
    <div ref={containerRef} className="relative">
      <label className="block text-xs font-medium uppercase tracking-wide text-on-surface-variant">
        {label}
      </label>
      <div className="relative mt-1">
        <input
          className={`w-full rounded-lg border bg-surface-container-lowest px-3 py-2 pr-9 text-sm text-on-surface placeholder:text-on-surface-variant/60 focus:outline-none focus:ring-1 ${borderColor}`}
          value={value}
          onChange={(e) => {
            onChange(e.target.value);
            setOpen(true);
            setActiveIdx(-1);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
          autoComplete="off"
          role="combobox"
          aria-expanded={open && suggestions.length > 0}
          aria-autocomplete="list"
          aria-invalid={showWarning}
          required
        />
        <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2">
          {fetching ? (
            <Spinner className="h-4 w-4 text-on-surface-variant" />
          ) : showValid ? (
            <span aria-label="Resolved" className="text-tertiary">
              ✓
            </span>
          ) : showWarning ? (
            <span aria-label="Warning" className="text-error">
              ⚠
            </span>
          ) : null}
        </span>
      </div>

      {showValid ? (
        <p className="mt-1 text-xs text-tertiary">
          {resolved!.kind}: <span className="font-mono">{resolved!.id}</span>
        </p>
      ) : showWarning ? (
        <p className="mt-1 text-xs text-error" role="status">
          {warning}
        </p>
      ) : !searched && value.trim().length >= 2 && fetching ? (
        <p className="mt-1 text-xs text-on-surface-variant">Checking…</p>
      ) : null}

      {open && (suggestions.length > 0 || fetching) ? (
        <ul
          className="absolute left-0 right-0 z-20 mt-1 max-h-60 overflow-auto rounded-lg border border-outline/20 bg-surface-container-high shadow-lg"
          role="listbox"
        >
          {fetching && suggestions.length === 0 ? (
            <li className="px-3 py-2 text-xs text-on-surface-variant">
              Searching…
            </li>
          ) : null}
          {suggestions.map((item, i) => (
            <li
              key={`${item.id}-${i}`}
              role="option"
              aria-selected={i === activeIdx}
              onMouseDown={(e) => {
                e.preventDefault();
                pick(item);
              }}
              onMouseEnter={() => setActiveIdx(i)}
              className={`cursor-pointer px-3 py-2 text-sm ${
                i === activeIdx
                  ? "bg-primary/20 text-on-surface"
                  : "text-on-surface hover:bg-surface-container-highest"
              }`}
            >
              <div className="font-medium">{item.name || item.id}</div>
              <div className="text-xs text-on-surface-variant">
                <span className="font-mono">{item.id}</span>
                <span className="ml-2 uppercase">{item.kind}</span>
              </div>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

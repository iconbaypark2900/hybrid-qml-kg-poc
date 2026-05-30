import type { Metaedge } from './types';

// All 24 Hetionet v1.0 metaedges with official edge counts (sum to 2,250,197)
export const HETIONET_METAEDGES: Metaedge[] = [
  { code: 'AdG',  name: 'Anatomy–downregulates–Gene',           src: 'Anatomy',  tgt: 'Gene',              count: 102240, dir: 'undirected' },
  { code: 'AeG',  name: 'Anatomy–expresses–Gene',               src: 'Anatomy',  tgt: 'Gene',              count: 526407, dir: 'undirected' },
  { code: 'AuG',  name: 'Anatomy–upregulates–Gene',             src: 'Anatomy',  tgt: 'Gene',              count: 97848,  dir: 'undirected' },
  { code: 'CbG',  name: 'Compound–binds–Gene',                  src: 'Compound', tgt: 'Gene',              count: 11571,  dir: 'undirected' },
  { code: 'CcSE', name: 'Compound–causes–SideEffect',           src: 'Compound', tgt: 'SideEffect',        count: 138944, dir: 'undirected' },
  { code: 'CdG',  name: 'Compound–downregulates–Gene',          src: 'Compound', tgt: 'Gene',              count: 21102,  dir: 'undirected' },
  { code: 'CpD',  name: 'Compound–palliates–Disease',           src: 'Compound', tgt: 'Disease',           count: 390,    dir: 'undirected' },
  { code: 'CrC',  name: 'Compound–resembles–Compound',          src: 'Compound', tgt: 'Compound',          count: 6486,   dir: 'undirected' },
  { code: 'CtD',  name: 'Compound–treats–Disease',              src: 'Compound', tgt: 'Disease',           count: 755,    dir: 'undirected' },
  { code: 'CuG',  name: 'Compound–upregulates–Gene',            src: 'Compound', tgt: 'Gene',              count: 18756,  dir: 'undirected' },
  { code: 'DaG',  name: 'Disease–associates–Gene',              src: 'Disease',  tgt: 'Gene',              count: 12623,  dir: 'undirected' },
  { code: 'DdG',  name: 'Disease–downregulates–Gene',           src: 'Disease',  tgt: 'Gene',              count: 7623,   dir: 'undirected' },
  { code: 'DlA',  name: 'Disease–localizes–Anatomy',            src: 'Disease',  tgt: 'Anatomy',           count: 3602,   dir: 'undirected' },
  { code: 'DpS',  name: 'Disease–presents–Symptom',             src: 'Disease',  tgt: 'Symptom',           count: 3357,   dir: 'undirected' },
  { code: 'DrD',  name: 'Disease–resembles–Disease',            src: 'Disease',  tgt: 'Disease',           count: 543,    dir: 'undirected' },
  { code: 'DuG',  name: 'Disease–upregulates–Gene',             src: 'Disease',  tgt: 'Gene',              count: 7731,   dir: 'undirected' },
  { code: 'GcG',  name: 'Gene–covaries–Gene',                   src: 'Gene',     tgt: 'Gene',              count: 61690,  dir: 'undirected' },
  { code: 'GiG',  name: 'Gene–interacts–Gene',                  src: 'Gene',     tgt: 'Gene',              count: 147164, dir: 'undirected' },
  { code: 'GpBP', name: 'Gene–participates–BiologicalProcess',  src: 'Gene',     tgt: 'BiologicalProcess', count: 559504, dir: 'undirected' },
  { code: 'GpCC', name: 'Gene–participates–CellularComponent',  src: 'Gene',     tgt: 'CellularComponent', count: 73566,  dir: 'undirected' },
  { code: 'GpMF', name: 'Gene–participates–MolecularFunction',  src: 'Gene',     tgt: 'MolecularFunction', count: 97222,  dir: 'undirected' },
  { code: 'GpPW', name: 'Gene–participates–Pathway',            src: 'Gene',     tgt: 'Pathway',           count: 84372,  dir: 'undirected' },
  { code: 'Gr>G', name: 'Gene→regulates→Gene',                  src: 'Gene',     tgt: 'Gene',              count: 265672, dir: 'directed'   },
  { code: 'PCiC', name: 'PharmacologicClass–includes–Compound', src: 'PharmacologicClass', tgt: 'Compound', count: 1029,   dir: 'undirected' }
];

// Hetionet v1.0 official totals
export const HETIONET_TOTALS = {
  diseases: 137,
  compounds: 1552,
  genes: 20945,
  metaedges: 24,
  totalEdges: 2250197
} as const;

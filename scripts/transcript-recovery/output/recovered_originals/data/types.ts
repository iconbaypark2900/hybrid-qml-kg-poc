// Hetionet v1.0 type definitions

export interface Disease {
  id: string;        // DOID identifier, e.g. "DOID:14330"
  name: string;
  cat: string;       // category: renal, oncology, autoimmune, etc.
}

export interface Compound {
  id: string;        // DrugBank ID, e.g. "DB12015"
  name: string;
  mech: string;      // mechanism description
  cat: string;       // therapeutic class
  pubchemCID?: string;
}

export interface Gene {
  id: string;        // NCBI Gene ID
  symbol: string;    // HGNC symbol
  name: string;      // full name
  cat: string;       // functional category
}

export interface Metaedge {
  code: string;
  name: string;
  src: string;
  tgt: string;
  count: number;
  dir: 'directed' | 'undirected';
}

export type RunPath = 'classical' | 'hybrid' | 'quantum';

export interface Algorithm {
  id: string;
  name: string;
  group: string;
  mech: string;
  params: number;
  runtime: string;
  path: RunPath;
  status: 'live' | 'dev' | 'fallback';
  primary?: boolean;
}

export interface IntegrityGuard {
  id: string;
  group: string;
  level: 'critical' | 'recommended' | 'optional';
  name: string;
  desc: string;
  source: string;
  impact: string;
  enabled: boolean;
}

export interface CompoundContextEntry {
  description: string;
  targets: string[];
  approval: string;
  approvalKind: 'approved' | 'trial';
  indications: string[];
  trials: string[];
  repurposing: string | null;
  equity: string | null;
}

export type CompoundContext = Record<string, CompoundContextEntry>;

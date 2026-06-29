"""CREEDS perturbation signatures → disease-signature reversal scores."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from entity_resolution.compound_mapper import CompoundMapper
from entity_resolution.synonym_resolver import SynonymResolver
from perturbation_layer.reversal_score import compute_reversal_score

logger = logging.getLogger(__name__)

_DEFAULT_CREEDS = Path("artifacts/external/creeds/single_drug_perturbations-v1.0.json")
_DEFAULT_SIGNATURE = Path("artifacts/signatures/tcga_brca_60/disease_signature.json")
_DEFAULT_GENE_MAP = Path("artifacts/external/gdc_tcga_brca/converted/tcga_brca_gene_map.csv")

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_PARENS = re.compile(r"\([^)]*\)")
_SALT_SUFFIXES = (
    " hydrochloride",
    " hcl",
    " sodium",
    " sulfate",
    " sulphate",
    " mesylate",
    " maleate",
    " tartrate",
    " citrate",
    " phosphate",
    " acetate",
    " fumarate",
    " succinate",
    " besylate",
    " tosylate",
    " nitrate",
    " bromide",
    " chloride",
    " potassium",
    " calcium",
    " magnesium",
    " monohydrate",
    " dihydrate",
    " trihydrate",
)

ReversalMethod = Literal["gene_overlap", "cosine"]


def normalize_compound_name(name: str) -> str:
    """Lowercase alphanumeric key with salt/parenthetical stripping."""

    text = str(name).strip().lower()
    text = _PARENS.sub(" ", text)
    for suffix in _SALT_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return _NON_ALNUM.sub("", text.strip())


def compound_name_variants(name: str) -> List[str]:
    """Return normalized lookup keys for a compound display name."""

    variants: List[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        norm = normalize_compound_name(value)
        if norm and norm not in seen:
            seen.add(norm)
            variants.append(norm)

    raw = str(name).strip()
    _add(raw)
    no_parens = _PARENS.sub(" ", raw).strip()
    if no_parens != raw:
        _add(no_parens)
    for suffix in _SALT_SUFFIXES:
        if no_parens.lower().endswith(suffix):
            _add(no_parens[: -len(suffix)])
    return variants


def _creeds_gene_symbols(items: list) -> List[str]:
    symbols: List[str] = []
    for item in items or []:
        if isinstance(item, list) and item:
            symbols.append(str(item[0]).upper())
        elif item:
            symbols.append(str(item).upper())
    return symbols


def _creeds_gene_score_map(items: list) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for item in items or []:
        if isinstance(item, list) and item:
            symbol = str(item[0]).upper()
            value = float(item[1]) if len(item) > 1 else 1.0
            scores[symbol] = value
        elif item:
            scores[str(item).upper()] = 1.0
    return scores


def load_ensg_to_symbol(gene_map_path: str | Path) -> Dict[str, str]:
    path = Path(gene_map_path)
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "gene" not in frame.columns:
        return {}
    symbol_col = "gene_symbol" if "gene_symbol" in frame.columns else "symbol"
    if symbol_col not in frame.columns:
        return {}
    mapping: Dict[str, str] = {}
    for _, row in frame.dropna(subset=["gene", symbol_col]).iterrows():
        gene = str(row["gene"]).split(".", 1)[0]
        mapping[gene] = str(row[symbol_col]).upper()
    return mapping


def load_disease_signature(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def disease_signature_gene_symbols(
    signature: Dict[str, Any],
    ensg_to_symbol: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    def _to_symbols(genes: List[str]) -> List[str]:
        out: List[str] = []
        for gene in genes:
            key = str(gene).split(".", 1)[0]
            symbol = ensg_to_symbol.get(key, key)
            if symbol:
                out.append(str(symbol).upper())
        return out

    return (
        _to_symbols(signature.get("up_genes", [])),
        _to_symbols(signature.get("down_genes", [])),
    )


def signature_gene_symbols_ordered(
    signature: Dict[str, Any],
    ensg_to_symbol: Dict[str, str],
    *,
    top_n: int = 500,
) -> List[str]:
    """Ordered signature genes for cosine scoring (ranked_genes first)."""

    genes: List[str] = []
    for item in signature.get("ranked_genes", []):
        gene = str(item.get("gene", "")).split(".", 1)[0]
        symbol = ensg_to_symbol.get(gene, gene)
        symbol = str(symbol).upper()
        if symbol and symbol not in genes:
            genes.append(symbol)
    for key in ("up_genes", "down_genes"):
        for gene in signature.get(key, []):
            ens = str(gene).split(".", 1)[0]
            symbol = ensg_to_symbol.get(ens, ens)
            symbol = str(symbol).upper()
            if symbol and symbol not in genes:
                genes.append(symbol)
    return genes[:top_n]


def disease_lfc_by_symbol(
    signature: Dict[str, Any],
    ensg_to_symbol: Dict[str, str],
) -> Dict[str, float]:
    """Map HGNC symbols to disease logFC (ranked_genes preferred)."""

    by_symbol: Dict[str, float] = {}
    for item in signature.get("ranked_genes", []):
        gene = str(item.get("gene", "")).split(".", 1)[0]
        symbol = ensg_to_symbol.get(gene, gene)
        by_symbol[str(symbol).upper()] = float(item.get("logfc", 0.0))

    up_symbols, down_symbols = disease_signature_gene_symbols(signature, ensg_to_symbol)
    for symbol in up_symbols:
        by_symbol.setdefault(symbol, 1.0)
    for symbol in down_symbols:
        by_symbol.setdefault(symbol, -1.0)
    return by_symbol


def cosine01(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.5
    return float((np.dot(a, b) / denom + 1.0) / 2.0)


def aggregate_profile_scores(scores: List[float]) -> float:
    """Aggregate multiple CREEDS profile scores via mean of top-3."""

    if not scores:
        return float("-inf")
    top = sorted(scores, reverse=True)[:3]
    return float(sum(top) / len(top))


@dataclass(frozen=True)
class CreedsProfile:
    creeds_id: str
    drug_name: str
    compound_hetionet_id: Optional[str]
    up_genes: Tuple[str, ...]
    down_genes: Tuple[str, ...]
    up_gene_scores: Tuple[Tuple[str, float], ...]
    down_gene_scores: Tuple[Tuple[str, float], ...]
    cell_type: str
    geo_id: str
    organism: str = "human"

    def effect_vector(self, signature_genes: List[str]) -> np.ndarray:
        up = dict(self.up_gene_scores)
        down = dict(self.down_gene_scores)
        values = []
        for gene in signature_genes:
            if gene in up:
                values.append(up[gene])
            elif gene in down:
                values.append(down[gene])
            else:
                values.append(0.0)
        return np.array(values, dtype=float)


def _resolve_profile_compound_id(
    record: Dict[str, Any],
    mapper: CompoundMapper,
) -> Optional[str]:
    drugbank_id = record.get("drugbank_id")
    if drugbank_id:
        mapped = mapper.map(str(drugbank_id))
        if mapped:
            return mapped
    drug_name = str(record.get("drug_name") or "")
    if drug_name:
        mapped = mapper.map(drug_name)
        if mapped:
            return mapped
    return None


def _index_profile_keys(
    profile: CreedsProfile,
    record: Dict[str, Any],
    *,
    by_hetionet: Dict[str, List[CreedsProfile]],
    by_name: Dict[str, List[CreedsProfile]],
    mapper: CompoundMapper,
) -> None:
    drugbank_id = str(record.get("drugbank_id") or "").strip().upper()
    if drugbank_id:
        by_hetionet.setdefault(f"Compound::{drugbank_id}", []).append(profile)
    if profile.compound_hetionet_id:
        by_hetionet.setdefault(profile.compound_hetionet_id, []).append(profile)

    for variant in compound_name_variants(profile.drug_name):
        by_name.setdefault(variant, []).append(profile)

    if profile.compound_hetionet_id:
        canonical = mapper._resolver.get_name(profile.compound_hetionet_id)
        if canonical:
            for variant in compound_name_variants(canonical):
                by_name.setdefault(variant, []).append(profile)


def build_creeds_profile_index(
    records: List[Dict[str, Any]],
    *,
    organism: str,
    mapper: CompoundMapper,
) -> Tuple[Dict[str, List[CreedsProfile]], Dict[str, List[CreedsProfile]]]:
    """Index CREEDS profiles by Hetionet compound ID and normalized drug name."""

    by_hetionet: Dict[str, List[CreedsProfile]] = {}
    by_name: Dict[str, List[CreedsProfile]] = {}
    organism_lc = organism.lower()

    for record in records:
        if organism_lc and organism_lc != "any":
            if str(record.get("organism", "")).lower() != organism_lc:
                continue
        up_scores = _creeds_gene_score_map(record.get("up_genes", []))
        down_scores = _creeds_gene_score_map(record.get("down_genes", []))
        profile = CreedsProfile(
            creeds_id=str(record.get("id", "")),
            drug_name=str(record.get("drug_name") or ""),
            compound_hetionet_id=_resolve_profile_compound_id(record, mapper),
            up_genes=tuple(up_scores.keys()),
            down_genes=tuple(down_scores.keys()),
            up_gene_scores=tuple(up_scores.items()),
            down_gene_scores=tuple(down_scores.items()),
            cell_type=str(record.get("cell_type") or ""),
            geo_id=str(record.get("geo_id") or ""),
            organism=str(record.get("organism") or "unknown"),
        )
        _index_profile_keys(
            profile,
            record,
            by_hetionet=by_hetionet,
            by_name=by_name,
            mapper=mapper,
        )

    return by_hetionet, by_name


def _name_match_profiles(
    compound_name: str,
    by_name: Dict[str, List[CreedsProfile]],
) -> List[CreedsProfile]:
    matches: List[CreedsProfile] = []
    seen_ids: set[str] = set()
    for variant in compound_name_variants(compound_name):
        exact = by_name.get(variant, [])
        if exact:
            for profile in exact:
                if profile.creeds_id not in seen_ids:
                    seen_ids.add(profile.creeds_id)
                    matches.append(profile)
        for norm_name, profiles in by_name.items():
            if variant in norm_name or norm_name in variant:
                for profile in profiles:
                    if profile.creeds_id not in seen_ids:
                        seen_ids.add(profile.creeds_id)
                        matches.append(profile)
    return matches


def profiles_for_candidate(
    compound: str,
    compound_hetionet_id: Optional[str],
    *,
    by_hetionet: Dict[str, List[CreedsProfile]],
    by_name: Dict[str, List[CreedsProfile]],
    mapper: CompoundMapper,
    synonym_resolver: Optional[SynonymResolver] = None,
) -> List[CreedsProfile]:
    seen: set[str] = set()
    matched: List[CreedsProfile] = []
    resolver = synonym_resolver or SynonymResolver(resolver=mapper._resolver)

    def _add(profiles: List[CreedsProfile]) -> None:
        for profile in profiles:
            if profile.creeds_id in seen:
                continue
            seen.add(profile.creeds_id)
            matched.append(profile)

    if compound_hetionet_id:
        _add(by_hetionet.get(compound_hetionet_id, []))

    mapped = mapper.map(compound) if compound else None
    if mapped:
        _add(by_hetionet.get(mapped, []))

    resolved = resolver.resolve(compound) if compound else None
    if resolved and resolved.startswith("Compound::"):
        _add(by_hetionet.get(resolved, []))

    _add(_name_match_profiles(compound, by_name))
    return matched


def score_profile_gene_overlap(
    profile: CreedsProfile,
    disease_up: List[str],
    disease_down: List[str],
) -> float:
    return compute_reversal_score(
        disease_up,
        disease_down,
        list(profile.up_genes),
        list(profile.down_genes),
    )


def score_profile_cosine(
    profile: CreedsProfile,
    signature_genes: List[str],
    lfc_by_symbol: Dict[str, float],
) -> float:
    effect = profile.effect_vector(signature_genes)
    lfc_vec = np.array(
        [lfc_by_symbol.get(gene, 0.0) for gene in signature_genes],
        dtype=float,
    )
    return cosine01(effect, -lfc_vec)


def best_reversal_for_candidate(
    compound: str,
    compound_hetionet_id: Optional[str],
    *,
    disease_up: List[str],
    disease_down: List[str],
    by_hetionet: Dict[str, List[CreedsProfile]],
    by_name: Dict[str, List[CreedsProfile]],
    mapper: CompoundMapper,
    reversal_method: ReversalMethod = "gene_overlap",
    signature_genes: Optional[List[str]] = None,
    lfc_by_symbol: Optional[Dict[str, float]] = None,
    synonym_resolver: Optional[SynonymResolver] = None,
) -> Tuple[float, float, float, Optional[CreedsProfile], int]:
    profiles = profiles_for_candidate(
        compound,
        compound_hetionet_id,
        by_hetionet=by_hetionet,
        by_name=by_name,
        mapper=mapper,
        synonym_resolver=synonym_resolver,
    )
    if not profiles:
        return 0.0, 0.0, 0.0, None, 0

    profile_scores: List[Tuple[CreedsProfile, float]] = []
    for profile in profiles:
        if reversal_method == "cosine":
            if not signature_genes or lfc_by_symbol is None:
                raise ValueError("cosine reversal requires signature_genes and lfc_by_symbol")
            score = score_profile_cosine(profile, signature_genes, lfc_by_symbol)
            profile_scores.append((profile, score))
        else:
            raw = score_profile_gene_overlap(profile, disease_up, disease_down)
            profile_scores.append((profile, raw))

    best_profile, _best_single = max(profile_scores, key=lambda item: item[1])
    aggregated = aggregate_profile_scores([score for _, score in profile_scores])

    if reversal_method == "cosine":
        normalized = aggregated if aggregated != float("-inf") else 0.0
    else:
        normalized = float((aggregated + 1.0) / 2.0) if aggregated != float("-inf") else 0.0

    return normalized, 0.0, 0.0, best_profile, len(profiles)


@dataclass
class CreedsReversalContext:
    disease_up: List[str]
    disease_down: List[str]
    by_hetionet: Dict[str, List[CreedsProfile]]
    by_name: Dict[str, List[CreedsProfile]]
    mapper: CompoundMapper
    signature_path: str
    creeds_path: str
    reversal_method: ReversalMethod = "gene_overlap"
    signature_genes: List[str] = field(default_factory=list)
    lfc_by_symbol: Dict[str, float] = field(default_factory=dict)
    synonym_resolver: SynonymResolver = field(default_factory=SynonymResolver)


def load_creeds_reversal_context(
    *,
    creeds_path: str | Path = _DEFAULT_CREEDS,
    disease_signature_path: str | Path = _DEFAULT_SIGNATURE,
    gene_map_path: str | Path = _DEFAULT_GENE_MAP,
    organism: str = "human",
    reversal_method: ReversalMethod = "gene_overlap",
) -> CreedsReversalContext:
    creeds_file = Path(creeds_path)
    signature_file = Path(disease_signature_path)
    if not creeds_file.exists():
        raise FileNotFoundError(f"CREEDS signatures not found: {creeds_file}")
    if not signature_file.exists():
        raise FileNotFoundError(f"Disease signature not found: {signature_file}")

    records = json.loads(creeds_file.read_text(encoding="utf-8"))
    signature = load_disease_signature(signature_file)
    ensg_to_symbol = load_ensg_to_symbol(gene_map_path)
    disease_up, disease_down = disease_signature_gene_symbols(signature, ensg_to_symbol)
    mapper = CompoundMapper()
    synonym_resolver = SynonymResolver(resolver=mapper._resolver)
    by_hetionet, by_name = build_creeds_profile_index(records, organism=organism, mapper=mapper)
    signature_genes = signature_gene_symbols_ordered(signature, ensg_to_symbol)
    lfc_by_symbol = disease_lfc_by_symbol(signature, ensg_to_symbol)

    logger.info(
        "CREEDS reversal context: %d indexed names, %d Hetionet compound keys, "
        "%d disease up / %d down genes, method=%s",
        len(by_name),
        len(by_hetionet),
        len(disease_up),
        len(disease_down),
        reversal_method,
    )
    return CreedsReversalContext(
        disease_up=disease_up,
        disease_down=disease_down,
        by_hetionet=by_hetionet,
        by_name=by_name,
        mapper=mapper,
        signature_path=str(signature_file),
        creeds_path=str(creeds_file),
        reversal_method=reversal_method,
        signature_genes=signature_genes,
        lfc_by_symbol=lfc_by_symbol,
        synonym_resolver=synonym_resolver,
    )


def _creeds_match_status(profile: Optional[CreedsProfile], *, filter_organism: str) -> str:
    if profile is None:
        return "unmatched"
    org = profile.organism.lower()
    if filter_organism.lower() == "human" or org == "human":
        return "matched_human"
    return "matched_non_human"


def enrich_candidates_with_creeds(
    candidates: List[Dict[str, Any]],
    context: CreedsReversalContext,
    *,
    filter_organism: str = "human",
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    matched = 0
    for candidate in candidates:
        sig_rev, cell_rev, pathway_rev, profile, profile_count = best_reversal_for_candidate(
            str(candidate.get("compound", "")),
            candidate.get("compound_hetionet_id"),
            disease_up=context.disease_up,
            disease_down=context.disease_down,
            by_hetionet=context.by_hetionet,
            by_name=context.by_name,
            mapper=context.mapper,
            reversal_method=context.reversal_method,
            signature_genes=context.signature_genes,
            lfc_by_symbol=context.lfc_by_symbol,
            synonym_resolver=context.synonym_resolver,
        )
        candidate["signature_reversal_score"] = sig_rev
        candidate["cell_type_reversal_score"] = cell_rev
        candidate["pathway_reversal_score"] = pathway_rev
        candidate["creeds_organism"] = filter_organism
        candidate["creeds_match_status"] = _creeds_match_status(profile, filter_organism=filter_organism)
        if profile is not None:
            matched += 1
            candidate["creeds_id"] = profile.creeds_id
            candidate["creeds_drug_name"] = profile.drug_name
            candidate["creeds_geo_id"] = profile.geo_id
            candidate["creeds_cell_type"] = profile.cell_type
            candidate["creeds_profile_organism"] = profile.organism
            candidate["creeds_profile_count"] = profile_count
        else:
            candidate["creeds_profile_count"] = 0
    stats = {
        "n_candidates": len(candidates),
        "n_creeds_matched": matched,
        "n_unmatched": len(candidates) - matched,
        "reversal_method": context.reversal_method,
        "profile_aggregation": "mean_top3",
        "creeds_organism": filter_organism,
    }
    return candidates, stats

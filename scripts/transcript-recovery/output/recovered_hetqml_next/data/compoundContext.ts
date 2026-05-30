import type { CompoundContext } from './types';

// Curated context for top candidates — keyed by DrugBank ID.
// Compounds without an entry here fall back to data derived from their base compound record.
export const COMPOUND_CONTEXT: CompoundContext = {
  'DB12015': { // Inaxaplin
    description: 'First-in-class APOL1 inhibitor in Phase III for APOL1-mediated kidney disease. Connects HTN-attributed ESKD, lupus nephritis, and sickle cell nephropathy via the APOL1 G1/G2 risk genotype enriched in African ancestry.',
    targets: ['APOL1'],
    approval: 'Phase III · investigational',
    approvalKind: 'trial',
    indications: ['APOL1-mediated kidney disease (investigational)'],
    trials: ['AMPLIFY · NCT05312398'],
    repurposing: 'Hypertension-attributed ESKD',
    equity: 'APOL1 G1/G2 risk allele: ≈22% AA frequency vs ~0% in European-ancestry populations.'
  },
  'DB11581': { // Venetoclax
    description: 'Selective BCL-2 inhibitor that displaces pro-apoptotic proteins, restoring programmed cell death in malignancies that depend on BCL-2 for survival.',
    targets: ['BCL2'],
    approval: 'FDA approved (2016)',
    approvalKind: 'approved',
    indications: ['CLL', 'SLL', 'AML (with HMA)'],
    trials: ['60+ active oncology trials'],
    repurposing: 'Multiple myeloma',
    equity: 't(11;14) MM subset shows enriched response.'
  },
  'DB14738': { // Deucravacitinib
    description: 'Selective TYK2 allosteric inhibitor — first-and-only oral psoriasis drug binding the TYK2 pseudokinase domain (avoids JAK1/2/3 off-target).',
    targets: ['TYK2'],
    approval: 'FDA approved (2022)',
    approvalKind: 'approved',
    indications: ['Plaque psoriasis'],
    trials: ['POETYK PsA · IBD studies'],
    repurposing: 'Lupus (IFN-high subset)',
    equity: 'Pseudokinase selectivity reduces hematopoietic JAK off-target risk.'
  },
  'DB09038': { // Empagliflozin
    description: 'SGLT2 inhibitor with proven cardiovascular and renal benefits beyond glucose lowering. Reduces ESKD progression in CKD across diabetic and non-diabetic populations.',
    targets: ['SLC5A2'],
    approval: 'FDA approved (2014)',
    approvalKind: 'approved',
    indications: ['T2DM', 'HFrEF', 'HFpEF', 'CKD'],
    trials: ['EMPA-KIDNEY', 'EMPEROR series'],
    repurposing: 'Salt-sensitive hypertension',
    equity: 'Renal benefit observed across ancestries; AA-specific RAAS-axis interaction under study.'
  },
  'DB01262': { // Decitabine
    description: 'DNA methyltransferase inhibitor. In SCD context, low-dose oral decitabine + cedazuridine reactivates fetal hemoglobin (HbF) via BCL11A demethylation.',
    targets: ['DNMT1', 'BCL11A'],
    approval: 'FDA approved (2006)',
    approvalKind: 'approved',
    indications: ['MDS', 'AML'],
    trials: ['ACCELERATE-SCD · NCT04815005'],
    repurposing: 'Sickle cell disease (low-dose oral)',
    equity: 'AA-relevant: SCD prevalence ≈1/365 AA births; HbF reactivation independent of BCL11A genotype.'
  },
  'DB12218': { // Capivasertib
    description: 'Pan-AKT (AKT1/2/3) inhibitor combined with fulvestrant for HR+/HER2− breast cancer with PIK3CA, AKT1, or PTEN alterations.',
    targets: ['AKT1', 'AKT2', 'AKT3'],
    approval: 'FDA approved (2023)',
    approvalKind: 'approved',
    indications: ['HR+/HER2− breast cancer (with fulvestrant)'],
    trials: ['CAPItello-291'],
    repurposing: 'Castration-resistant prostate cancer (ERG−)',
    equity: 'PIK3CA pathway alteration frequency varies modestly by ancestry.'
  },
  'DB00619': { // Imatinib
    description: 'BCR-ABL tyrosine kinase inhibitor — the first targeted cancer therapy. Revolutionized chronic myeloid leukemia treatment in 2001.',
    targets: ['ABL1', 'KIT', 'PDGFRA'],
    approval: 'FDA approved (2001)',
    approvalKind: 'approved',
    indications: ['CML', 'GIST', 'Ph+ ALL'],
    trials: ['IRIS (long-term f/u)'],
    repurposing: 'Multiple kinase indications',
    equity: null
  },
  'DB00945': { // Aspirin
    description: 'Foundational COX-1/2 inhibitor — antiplatelet and anti-inflammatory. The oldest entry in modern pharmacopoeia (1899).',
    targets: ['PTGS1', 'PTGS2'],
    approval: 'FDA approved',
    approvalKind: 'approved',
    indications: ['Pain', 'Fever', 'Inflammation', 'MI/stroke prevention'],
    trials: ['CAPRIE', 'ARRIVE', 'ASPREE'],
    repurposing: 'Colorectal cancer chemoprevention',
    equity: null
  },
  'DB00331': { // Metformin
    description: 'Biguanide — first-line type 2 diabetes therapy. Acts via mitochondrial complex I inhibition and AMPK activation.',
    targets: ['mitochondrial complex I'],
    approval: 'FDA approved (1995 US)',
    approvalKind: 'approved',
    indications: ['Type 2 diabetes'],
    trials: ['UKPDS', 'ACE'],
    repurposing: 'PCOS · cancer adjuvant',
    equity: null
  }
};

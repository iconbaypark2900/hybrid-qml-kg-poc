"""
build_pdf.py — Generate paper.pdf from PAPER.md content using ReportLab.
Run: .venv/bin/python3 docs/build_pdf.py
Output: docs/paper.pdf
"""

import os, sys, re
sys.path.insert(0, os.path.dirname(__file__))

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether, Image, Preformatted
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

BASE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE, "..", "figures")
OUT = os.path.join(BASE, "paper.pdf")

# ── Colour palette ────────────────────────────────────────────────────────────
C_TITLE   = HexColor("#1a1a2e")
C_HEAD1   = HexColor("#16213e")
C_HEAD2   = HexColor("#0f3460")
C_HEAD3   = HexColor("#533483")
C_ACCENT  = HexColor("#4a9eff")
C_TEXT    = HexColor("#1e293b")
C_LIGHT   = HexColor("#64748b")
C_RULE    = HexColor("#cbd5e1")
C_CODE_BG = HexColor("#f1f5f9")
C_TABLE_H = HexColor("#1e3a5f")
C_TABLE_A = HexColor("#f8fafc")
C_TABLE_B = HexColor("#e2e8f0")

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

STYLES = {
    "title": S("title",
        fontName="Helvetica-Bold", fontSize=17, leading=22,
        textColor=C_TITLE, spaceAfter=4, alignment=TA_CENTER),

    "authors": S("authors",
        fontName="Helvetica", fontSize=11, leading=16,
        textColor=C_HEAD2, spaceAfter=2, alignment=TA_CENTER),

    "affil": S("affil",
        fontName="Helvetica-Oblique", fontSize=9, leading=13,
        textColor=C_LIGHT, spaceAfter=2, alignment=TA_CENTER),

    "date": S("date",
        fontName="Helvetica", fontSize=9, leading=13,
        textColor=C_LIGHT, spaceAfter=14, alignment=TA_CENTER),

    "abstract_head": S("abstract_head",
        fontName="Helvetica-Bold", fontSize=10, leading=14,
        textColor=C_HEAD1, spaceAfter=4, alignment=TA_CENTER),

    "abstract": S("abstract",
        fontName="Helvetica", fontSize=9.5, leading=14,
        textColor=C_TEXT, spaceBefore=0, spaceAfter=14,
        leftIndent=36, rightIndent=36, alignment=TA_JUSTIFY),

    "h1": S("h1",
        fontName="Helvetica-Bold", fontSize=12, leading=16,
        textColor=C_HEAD1, spaceBefore=16, spaceAfter=5,
        borderPad=(0, 0, 3, 0)),

    "h2": S("h2",
        fontName="Helvetica-Bold", fontSize=10.5, leading=14,
        textColor=C_HEAD2, spaceBefore=12, spaceAfter=4),

    "h3": S("h3",
        fontName="Helvetica-BoldOblique", fontSize=10, leading=13,
        textColor=C_HEAD3, spaceBefore=8, spaceAfter=3),

    "body": S("body",
        fontName="Helvetica", fontSize=9.5, leading=14,
        textColor=C_TEXT, spaceBefore=0, spaceAfter=6,
        alignment=TA_JUSTIFY),

    "body_bold": S("body_bold",
        fontName="Helvetica-Bold", fontSize=9.5, leading=14,
        textColor=C_TEXT, spaceBefore=0, spaceAfter=6),

    "bullet": S("bullet",
        fontName="Helvetica", fontSize=9.5, leading=14,
        textColor=C_TEXT, spaceBefore=1, spaceAfter=2,
        leftIndent=18, firstLineIndent=-10),

    "code": S("code",
        fontName="Courier", fontSize=8, leading=11,
        textColor=HexColor("#1e293b"), spaceBefore=4, spaceAfter=4,
        leftIndent=18, rightIndent=18,
        backColor=C_CODE_BG),

    "caption": S("caption",
        fontName="Helvetica-Oblique", fontSize=8.5, leading=12,
        textColor=C_LIGHT, spaceBefore=3, spaceAfter=10,
        alignment=TA_CENTER),

    "ref": S("ref",
        fontName="Helvetica", fontSize=8.5, leading=12,
        textColor=C_TEXT, spaceBefore=1, spaceAfter=2,
        leftIndent=18, firstLineIndent=-18),

    "appendix_h": S("appendix_h",
        fontName="Helvetica-Bold", fontSize=10, leading=14,
        textColor=C_HEAD1, spaceBefore=14, spaceAfter=4),

    "table_h": S("table_h",
        fontName="Helvetica-Bold", fontSize=8, leading=10,
        textColor=colors.white, alignment=TA_CENTER),

    "table_c": S("table_c",
        fontName="Helvetica", fontSize=8, leading=10,
        textColor=C_TEXT, alignment=TA_LEFT),

    "table_n": S("table_n",
        fontName="Helvetica", fontSize=8, leading=10,
        textColor=C_TEXT, alignment=TA_CENTER),

    "table_label": S("table_label",
        fontName="Helvetica-Bold", fontSize=9, leading=12,
        textColor=C_HEAD2, spaceBefore=10, spaceAfter=3),
}

def p(text, style="body"):
    """Inline-markdown → reportlab XML then Paragraph."""
    # bold **text** → <b>text</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # italic *text* → <i>text</i>
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    # inline code `text` → <font face="Courier">text</font>
    text = re.sub(r'`([^`]+)`', r'<font face="Courier" size="8">\1</font>', text)
    # escape bare & not already escaped
    text = re.sub(r'&(?!amp;|lt;|gt;|nbsp;|apos;|quot;|#)', '&amp;', text)
    return Paragraph(text, STYLES[style])

def rule():
    return HRFlowable(width="100%", thickness=0.5, color=C_RULE, spaceAfter=6, spaceBefore=2)

def vspace(n=6):
    return Spacer(1, n)

def section(num, title):
    return [rule(), p(f"{num}. {title}", "h1")]

def subsection(num, title):
    return [p(f"{num} {title}", "h2")]

def subsubsection(title):
    return [p(title, "h3")]

def math_block(latex_approx):
    """Render a math expression as an indented italic paragraph."""
    return p(f"<i>{latex_approx}</i>", "body")

def img(fname, width_in=5.5, caption_text="", max_height_in=4.5):
    path = os.path.join(FIGURES, fname)
    if not os.path.exists(path):
        return [p(f"[Figure: {fname} not found]", "caption")]
    # Load to get native aspect ratio
    from PIL import Image as PILImage
    with PILImage.open(path) as pil:
        native_w, native_h = pil.size
    aspect = native_h / native_w
    w = width_in * inch
    h = w * aspect
    # Clamp height
    if h > max_height_in * inch:
        h = max_height_in * inch
        w = h / aspect
    im = Image(path, width=w, height=h)
    im.hAlign = "CENTER"
    items = [vspace(6), im]
    if caption_text:
        items.append(p(caption_text, "caption"))
    items.append(vspace(6))
    return items

def make_table(headers, rows, col_widths=None, caption=None):
    """Build a styled ReportLab Table."""
    tc = STYLES["table_c"]
    th = STYLES["table_h"]
    tn = STYLES["table_n"]

    header_row = [Paragraph(str(h), th) for h in headers]
    data = [header_row]
    for row in rows:
        data.append([Paragraph(str(c), tc) for c in row])

    if col_widths is None:
        total = 6.5 * inch
        n = len(headers)
        col_widths = [total / n] * n

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    style = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  C_TABLE_H),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  8),
        ("ALIGN",        (0, 0), (-1, 0),  "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_TABLE_A, C_TABLE_B]),
        ("GRID",         (0, 0), (-1, -1), 0.35, C_RULE),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ])
    tbl.setStyle(style)

    items = []
    if caption:
        items.append(p(caption, "table_label"))
    items.append(tbl)
    items.append(vspace(8))
    return items

# ─────────────────────────────────────────────────────────────────────────────
# BUILD STORY
# ─────────────────────────────────────────────────────────────────────────────
story = []

# ── Title block ──────────────────────────────────────────────────────────────
story += [
    vspace(8),
    p("Quantum Kernel Link Prediction on Biomedical Knowledge Graphs:", "title"),
    p("Bridging Graph Embeddings and Quantum Feature Spaces for Drug Repurposing", "title"),
    vspace(4),
    p("Kevin Robinson, Mark Jack, Jonathan Beale", "authors"),
    p("Quantum Global Group", "affil"),
    p("arXiv preprint · April 2026", "date"),
    rule(),
    vspace(4),
    p("Abstract", "abstract_head"),
    p(
        "We present the first system to combine knowledge graph (KG) embeddings with quantum kernel "
        "classifiers for biomedical link prediction. Our hybrid quantum-classical pipeline predicts "
        "<i>Compound-treats-Disease</i> (CtD) relationships on the Hetionet knowledge graph — a "
        "heterogeneous biomedical network with 47,031 entities and 2.25 million edges across 24 "
        "relation types — by training full-graph RotatE embeddings, constructing pharmacologically "
        "enriched pair-wise features, and classifying candidate links with a stacking ensemble of "
        "classical tree models and a quantum support vector classifier (QSVC) using a Pauli feature "
        "map in a 16-qubit circuit. While prior work applies quantum kernels to molecular fingerprints "
        "and classical KG methods to drug repurposing independently, no existing approach feeds learned "
        "KG embedding vectors into quantum feature spaces for link prediction. "
        "Our primary configuration (RotatE-128D, 200 epochs, Pauli-16Q, reps=2, C=0.1, hard negatives) "
        "achieves a test PR-AUC of <b>0.7987</b>, a +1.49 pp improvement over the strongest classical "
        "baseline (RandomForest, 0.7838). An extended configuration with 256D embeddings and "
        "Optuna-tuned classical components reaches <b>0.8581 PR-AUC</b>. A central finding is that "
        "switching from the ZZ to Pauli feature map <i>lowers</i> standalone QSVC performance "
        "(0.7216 to 0.6343) while <i>raising</i> ensemble performance (0.7408 to 0.7987): the Pauli "
        "kernel produces predictions less correlated with classical model errors, providing greater "
        "complementary signal to the stacking meta-learner. We validate top predictions against "
        "ClinicalTrials.gov, identify a score-validity inversion problem in embedding-based scoring, "
        "and introduce a ten-feature mechanism-of-action (MoA) module derived from multi-relational "
        "Hetionet structure to address it.",
        "abstract"),
    rule(),
    vspace(6),
]

# ── 1. Introduction ───────────────────────────────────────────────────────────
story += section("1", "Introduction")
story += subsection("1.1", "Motivation")
story += [
    p("Drug repurposing — identifying new therapeutic indications for approved compounds — offers a "
      "faster, lower-cost path to clinical deployment than de novo drug discovery. Knowledge graph (KG) "
      "link prediction is one of the most tractable computational approaches: by learning representations "
      "of biomedical entities and their relationships, a model can score unseen (compound, disease) pairs "
      "and surface candidates for further investigation."),
    p("The Hetionet knowledge graph [1] encodes 2.25 million biomedical relationships across 47,031 "
      "entities, spanning genes, compounds, diseases, biological processes, pathways, side effects, and "
      "anatomical structures. The <i>Compound-treats-Disease</i> (CtD) relation captures 755 known "
      "drug-disease treatment pairs. Predicting missing CtD links is a well-defined, clinically "
      "interpretable link prediction task."),
    p("Classical KG embedding methods — TransE, RotatE, ComplEx, and GNN-based approaches — are mature "
      "and achieve strong baselines on this task [2]. Quantum machine learning offers kernel methods that "
      "compute similarity in exponentially large Hilbert spaces via parameterized quantum circuits. "
      "Quantum kernels have been applied to drug-target interaction prediction [3] and ligand-based "
      "virtual screening [4] — but exclusively on molecular-level features such as fingerprints and "
      "QSAR descriptors."),
    p("<b>The gap.</b> No prior work feeds learned KG embedding vectors into quantum feature spaces for "
      "link prediction. The two research tracks — classical KG-based repurposing and quantum molecular "
      "drug discovery — have developed independently. This paper bridges them."),
]

story += subsection("1.2", "Problem Statement")
story += [
    p("Given Hetionet with known CtD edges as positives and hard-sampled negatives, train a binary "
      "classifier predicting whether an unseen (compound, disease) pair represents a treatment "
      "relationship. We compare classical-only, quantum-only, and hybrid quantum-classical stacking "
      "ensemble approaches, reporting test PR-AUC as the primary metric — appropriate for the "
      "class-imbalanced link prediction setting."),
]

story += subsection("1.3", "Contributions")
story += [
    p("1. <b>Novel intersection.</b> To our knowledge, this is the first system to combine learned KG "
      "embedding vectors (RotatE, full-graph) with quantum kernel classifiers (QSVC, fidelity kernel) "
      "for biomedical link prediction, bridging two previously disjoint research tracks.", "bullet"),
    p("2. <b>Counterintuitive quantum-classical synergy.</b> The Pauli feature map degrades standalone "
      "QSVC performance (0.7216 to 0.6343, -8.7 pp) relative to ZZ, yet raises ensemble PR-AUC "
      "(0.7408 to 0.7987, +5.8 pp). The Pauli kernel generates predictions with lower correlation "
      "to classical tree model errors, providing greater diversity for the stacking meta-learner.", "bullet"),
    p("3. <b>Mechanism-of-action (MoA) feature module.</b> Ten pharmacological plausibility features "
      "derived from multi-relational Hetionet structure to address the score-validity inversion "
      "problem inherent in embedding-based scoring.", "bullet"),
    p("4. <b>Clinical validation.</b> Top predictions validated against ClinicalTrials.gov. Two of six "
      "novel predictions are supported by 7+ clinical trials each; one (Ezetimibe to gout) represents "
      "a biologically plausible novel hypothesis.", "bullet"),
    p("5. <b>Reproducible pipeline</b> with configurable embeddings (RotatE, ComplEx, DistMult), "
      "feature maps (ZZ, Pauli), ensemble methods, GPU simulation, and IBM Quantum hardware backends.", "bullet"),
]

# ── 2. Background ─────────────────────────────────────────────────────────────
story += section("2", "Background and Related Work")

story += subsection("2.1", "Hetionet and KG-Based Drug Repurposing")
story += [
    p("Hetionet v1.0 [1] is a heterogeneous biomedical knowledge graph with 47,031 nodes and "
      "2,250,198 edges across 24 relation types. Key relations include: Compound-treats-Disease "
      "(CtD, 755 edges), Compound-binds-Gene (CbG, 11,571), Disease-associates-Gene (DaG, 12,623), "
      "Gene-participates-Pathway (GpPW, 84,372), Compound-resembles-Compound (CrC, 6,486), and "
      "Disease-resembles-Disease (DrD, 543). The entity set covers 1,552 compounds, 137 diseases, "
      "and 20,945 genes, among others."),
    p("Mayers et al. [2] benchmarked seven KG embedding methods on biomedical KGs achieving MRR of "
      "0.9792 via ensemble. Recent work has extended KG-based repurposing with LLM-assisted retrieval "
      "augmentation [5] and multi-database KG construction [6]. All are exclusively classical."),
]

story += subsection("2.2", "Quantum Kernel Methods")
story += [
    p("Quantum kernels compute pairwise similarity in a quantum feature space: "
      "<i>k(x, y) = |&lt;0|U(x)U(y)|0&gt;|</i><super>2</super>, where U(x) is a parameterized "
      "encoding circuit. A QSVC uses this kernel as a drop-in for classical kernels. Expressivity "
      "depends on the feature map architecture — ZZ (second-order Pauli-Z interactions) or Pauli "
      "(mixed X, Y, Z, ZZ interactions) — and circuit repetitions. Kernel computation scales as "
      "O(n<super>2</super>) in training set size [10]."),
]

story += subsection("2.3", "Quantum Methods in Drug Discovery")
story += [
    p("QKDTI [3] applied Quantum Support Vector Regression with quantum feature mapping to "
      "drug-target interaction prediction using molecular descriptors. Kruger et al. [4] demonstrated "
      "QSVC for ligand-based virtual screening on molecular fingerprints. A 2024 preprint [7] "
      "described a hybrid quantum-classical pipeline for real-world drug discovery, again at the "
      "molecular level."),
    p("<b>The critical distinction:</b> all prior quantum drug discovery work applies quantum kernels "
      "or circuits to molecular-level features. All prior KG-based drug repurposing uses classical "
      "methods. Our work is the first to apply quantum kernels to <i>learned KG embedding vectors</i>, "
      "encoding the full relational structure of the biomedical knowledge graph as quantum circuit inputs."),
]

story += subsection("2.4", "Stacking Ensembles and Model Diversity")
story += [
    p("Stacking trains a meta-learner on base model predictions, exploiting the principle that "
      "diverse errors — not individually high performance — drive ensemble gains. We exploit this "
      "explicitly: the Pauli QSVC performs below the ZZ QSVC in isolation but provides more diverse "
      "errors relative to random forest and extra trees, enabling the meta-learner to extract "
      "greater combined signal."),
]

# ── 3. Dataset ────────────────────────────────────────────────────────────────
story += section("3", "Dataset")
story += make_table(
    ["Statistic", "Value"],
    [
        ["Source", "Hetionet v1.0 (het.io)"],
        ["Total entities", "47,031"],
        ["— Compounds", "1,552"],
        ["— Diseases", "137"],
        ["— Genes", "20,945"],
        ["— Other (pathways, processes, anatomy, etc.)", "24,597"],
        ["Total edges (all 24 relations)", "2,250,198"],
        ["Target relation: Compound-treats-Disease (CtD)", "755 positive edges"],
        ["Train positives (80%)", "604"],
        ["Test positives (20%)", "151"],
        ["Negative sampling strategy", "Hard negatives, 1:1 ratio"],
        ["Train pairs (positives + hard negatives)", "1,208"],
        ["Test pairs", "302"],
        ["Feature dimensionality — classical path", "299 (base) / 309 (+ MoA module)"],
        ["Feature dimensionality — quantum path", "16 qubits (pre-PCA: 299D to 24D to 16Q)"],
        ["Class balance (constructed)", "50 / 50"],
        ["Unique compounds in training heads", "354 (29.3% of 1,208 pairs)"],
        ["Unique diseases in training tails", "75 (6.2% of 1,208 pairs)"],
    ],
    col_widths=[3.5*inch, 3.0*inch],
    caption="Table 1. Hetionet CtD Dataset Statistics.",
)
story += [
    p("The low tail uniqueness (75 unique diseases from 137 total) reflects that Hetionet's CtD "
      "subgraph is concentrated: most training pairs map compounds to a small set of well-studied "
      "diseases. This is a structural property of the graph, not a data quality issue."),
]

# ── 4. Methods ────────────────────────────────────────────────────────────────
story += section("4", "Methods")

story += subsection("4.1", "Pipeline Overview")
story += [
    p("The full pipeline proceeds as follows: (1) Load Hetionet and extract CtD edges as positives. "
      "(2) Train full-graph RotatE embeddings across all 24 relation types using PyKEEN [9]. "
      "(3) Construct train/test pairs with hard negative sampling at 1:1 ratio. "
      "(4) Build feature vectors for each (compound, disease) pair from embeddings and graph topology. "
      "(5) <b>Classical path:</b> Train RandomForest, ExtraTrees, LogisticRegression with GridSearchCV. "
      "(6) <b>Quantum path:</b> Reduce dimensionality via PCA, encode into a Pauli or ZZ feature map, "
      "compute fidelity quantum kernel, train QSVC. "
      "(7) <b>Stacking ensemble:</b> Train a logistic regression meta-learner on out-of-fold base model "
      "predictions. (8) Evaluate on held-out test set."),
]
story += img("fig1_pipeline.png",
    width_in=6.5,
    caption_text=(
        "Figure 1. Hybrid quantum-classical pipeline for Compound-treats-Disease (CtD) link prediction "
        "on Hetionet. Full-graph RotatE embeddings are trained over all 24 relation types (2.25M edges). "
        "Classical path (blue) and quantum path (purple) merge at the stacking ensemble (green)."
    ))

story += subsection("4.2", "Full-Graph Knowledge Graph Embeddings")
story += [
    p("We train RotatE [8] embeddings over all 2,250,198 Hetionet edges across all 24 relation types "
      "using PyKEEN [9]. RotatE models each relation as an element-wise rotation in complex space: for "
      "a valid triple (h, r, t), the relation r acts as <b>h</b> o <b>r</b> ≈ <b>t</b> where o denotes "
      "complex-space element-wise multiplication. Training on the full graph enriches compound and "
      "disease representations with signal from gene binding (CbG), disease-gene associations (DaG), "
      "pathway participation (GpPW), side effects (CcSE), and all other relation types."),
    p("<b>Primary configuration:</b> 128D, 200 epochs, full-graph, hard negatives, seed 42. "
      "<b>Extended configuration:</b> 256D, 250 epochs, full-graph."),
]

story += subsection("4.3", "Pair Feature Construction")
story += [
    p("For each (compound c, disease d) pair, we construct a feature vector from entity embeddings "
      "and training-graph topology:"),
    p("<i>x_cd = [e_c || e_d || e_c - e_d || e_c o e_d] + g_cd</i>"),
    p("where e_c, e_d are 128-dimensional entity embeddings, || denotes concatenation, o is "
      "element-wise product, and g_cd is a vector of graph topology features — node degrees, "
      "common neighbor count, Jaccard similarity — computed exclusively from training edges to "
      "prevent data leakage. The resulting feature vector has approximately 299 dimensions."),
    p("<b>Hard negative sampling.</b> Negatives are selected as compound-disease pairs that are "
      "structurally close in the graph but do not have a known CtD edge. This prevents the model "
      "from learning trivial heuristics and consistently outperforms diverse and random sampling."),
]

story += subsection("4.4", "Quantum Kernel Classifier")
story += [
    p("<b>Dimensionality reduction.</b> The ~299D classical feature vector is compressed to 24D via "
      "PCA, then projected to 16 qubits by a learned linear layer. For the 256D embedding "
      "configuration, no pre-PCA is applied and projection goes directly to 12 qubits."),
    p("<b>Feature maps.</b> We evaluate two Qiskit [11] feature maps: <b>ZZ feature map</b> — "
      "second-order Pauli-Z interactions between adjacent qubits, reps=2-3; and <b>Pauli feature "
      "map</b> — mixed Pauli interactions (X, Y, Z, ZZ) over all qubit pairs, reps=2."),
    p("<b>Fidelity quantum kernel.</b> The quantum kernel is:"),
    p("<i>k(x, y) = |&lt;0|U(x)U(y)|0&gt;|</i><super>2</super>"),
    p("computed via statevector simulation. No Nystrom approximation is used in the primary run, "
      "resulting in a full 1208 x 1208 kernel matrix requiring approximately 1.46 million circuit "
      "evaluations."),
    p("<b>Computation time.</b> The primary 16-qubit Pauli kernel required <b>2,619 seconds</b> "
      "(43.6 minutes) on CPU statevector simulator — a genuine full-dataset quantum kernel "
      "computation, not a cached or subsampled approximation."),
    p("<b>QSVC.</b> Regularization C=0.1 (primary, manually set) or C=0.676 (extended, "
      "Optuna-tuned). Classification via scikit-learn's SVC with a precomputed quantum kernel matrix."),
]

story += subsection("4.5", "Mechanism-of-Action Feature Module")
story += [
    p("A key limitation of embedding-based link prediction is that high scores may reflect "
      "<i>structural proximity</i> in the graph without corresponding <i>mechanistic plausibility</i>. "
      "For example, Abacavir (an HIV antiretroviral) receives a high CtD embedding score for ocular "
      "cancer because HIV comorbidity with ocular conditions creates graph-structural proximity — "
      "but there is no pharmacological basis for this treatment."),
    p("We introduce 10 mechanism-of-action (MoA) features per (compound, disease) pair, each derived "
      "from a specific multi-relational Hetionet subgraph:"),
]
story += make_table(
    ["#", "Feature Name", "Source Relations", "Signal Captured"],
    [
        ["1", "binding_targets",          "CbG",                "Genes the compound binds"],
        ["2", "disease_genes",            "DaG",                "Genes associated with disease"],
        ["3", "shared_targets",           "CbG ∩ DaG",          "Direct mechanistic evidence"],
        ["4", "target_overlap",           "Jaccard(CbG, DaG)",  "Normalized mechanistic overlap"],
        ["5", "shared_pathway_genes",     "CbG ∩ DaG via GpPW", "Shared pathway involvement"],
        ["6", "pharmacologic_classes",    "PCiC",               "Drug class breadth"],
        ["7", "compound_similarity",      "CrC",                "Chemical neighborhood size"],
        ["8", "similar_compounds_treat",  "CrC ∩ CtD(train)",   "Analogical treatment evidence"],
        ["9", "disease_similarity",       "DrD",                "Disease neighborhood size"],
        ["10","similar_diseases_treated", "DrD ∩ CtD(train)",   "Analogical disease evidence"],
    ],
    col_widths=[0.3*inch, 1.5*inch, 1.6*inch, 2.6*inch],
    caption="Table 2. Mechanism-of-Action Feature Definitions.",
)
story += [
    p("Features 8 and 10 use only <i>training</i> CtD edges for treatment lookups, preventing "
      "data leakage. Enabled with <font face='Courier' size='8'>--use_moa_features</font>."),
]

story += subsection("4.6", "Classical Models and Ensemble")
story += [
    p("<b>Classical base models:</b> RandomForest (500 estimators), ExtraTrees (500 estimators), "
      "LogisticRegression (L2). GridSearchCV with 5-fold cross-validation tunes key hyperparameters."),
    p("<b>Stacking ensemble:</b> A logistic regression meta-learner is trained on stacked out-of-fold "
      "predictions from RF, ET, and QSVC base models. The meta-learner learns the optimal linear "
      "combination of base model outputs, removing the need for manual weight specification."),
]

# ── 5. Experimental Setup ─────────────────────────────────────────────────────
story += section("5", "Experimental Setup")

story += subsection("5.1", "Software and Hardware")
story += [
    p("Python 3.9+; PyKEEN 1.10+ for KG embedding training; Qiskit 1.x and Qiskit-Aer for quantum "
      "circuits and statevector simulation; scikit-learn 1.4+ for classical models; PyTorch 2.x as "
      "PyKEEN backend. Quantum simulation: Qiskit Aer CPU statevector simulator (primary). "
      "IBM Quantum (Heron) for hardware runs. Memory: ~16 GB RAM for 16-qubit full kernel matrix. "
      "Random seed 42 for all splits and training."),
]

story += subsection("5.2", "Reproducing the Primary Result")
story += [
    p("The following command reproduces the primary 0.7987 result (expected QSVC computation: ~43 min on CPU):"),
    Preformatted(
        "python scripts/run_optimized_pipeline.py \\\n"
        "  --relation CtD --full_graph_embeddings \\\n"
        "  --embedding_method RotatE --embedding_dim 128 \\\n"
        "  --embedding_epochs 200 --negative_sampling hard \\\n"
        "  --qml_dim 16 --qml_feature_map Pauli \\\n"
        "  --qml_feature_map_reps 2 --qsvc_C 0.1 \\\n"
        "  --qml_pre_pca_dim 24 --run_ensemble \\\n"
        "  --ensemble_method stacking --tune_classical --fast_mode",
        STYLES["code"]
    ),
    vspace(4),
]

# ── 6. Results ────────────────────────────────────────────────────────────────
story += section("6", "Results")

story += subsection("6.1", "Primary Results")
story += [
    p("<i>Configuration: RotatE 128D, 200 epochs, full-graph, Pauli feature map (reps=2), 16 qubits, "
      "pre-PCA 24D, QSVC C=0.1, hard negatives (1:1), stacking ensemble, GridSearchCV tuning. "
      "Source: optimized_results_20260216-100431.json. QSVC kernel computation: 2,619 s uncached.</i>"),
]
story += make_table(
    ["Model", "PR-AUC", "ROC-AUC", "Accuracy", "Type"],
    [
        ["Ensemble-QC-stacking (Pauli)", "0.7987", "0.7456", "0.5762", "Hybrid quantum-classical"],
        ["RandomForest-Optimized",       "0.7838", "0.7319", "0.5828", "Classical"],
        ["ExtraTrees-Optimized",         "0.7807", "0.7301", "0.6623", "Classical"],
        ["Ensemble-QC-stacking (ZZ)†",   "0.7408", "—",      "0.6490", "Hybrid quantum-classical"],
        ["QSVC-Optimized (ZZ)†",         "0.7216", "—",      "0.6556", "Quantum only"],
        ["QSVC-Optimized (Pauli)",        "0.6343", "0.6313", "0.5861", "Quantum only"],
    ],
    col_widths=[2.3*inch, 0.8*inch, 0.9*inch, 0.85*inch, 1.65*inch],
    caption="Table 3. Primary Benchmark — RotatE-128D, Pauli-16Q. †ZZ results from separate run (genuine 908 s kernel).",
)
story += [
    p("The stacking ensemble improves over the best classical model (RandomForest, 0.7838) by "
      "<b>+0.0149 PR-AUC absolute (+1.49 pp)</b>. RF test precision is 1.000 with recall 0.166 — "
      "indicating very high precision on confident predictions, consistent with the imbalanced "
      "nature of the underlying drug-disease relationship space."),
]

story += subsection("6.2", "Extended Results: 256D Embeddings and Optuna Tuning")
story += [
    p("<i>Configuration: RotatE 256D, 250 epochs, full-graph, Pauli feature map (reps=1), 12 qubits, "
      "no pre-PCA, QSVC C=0.676 (Optuna-tuned), hard negatives, stacking. Source: "
      "optimized_results_20260323-134844.json. Note: quantum kernel was computed once (~99 s) "
      "and reused across Optuna classical hyperparameter trials.</i>"),
]
story += make_table(
    ["Model", "PR-AUC", "ROC-AUC", "Accuracy", "Type"],
    [
        ["Ensemble-QC-stacking (Pauli-256D)", "0.8581", "0.8245", "0.7317", "Hybrid quantum-classical"],
        ["RandomForest-Optimized",             "0.8569", "0.8231", "0.7351", "Classical"],
        ["ExtraTrees-Optimized",               "0.8498", "—",      "0.7351", "Classical"],
        ["QSVC-Optimized (Pauli-256D)",         "0.7222", "0.7272", "0.6391", "Quantum only"],
    ],
    col_widths=[2.3*inch, 0.8*inch, 0.9*inch, 0.85*inch, 1.65*inch],
    caption="Table 4. Extended Configuration — RotatE-256D, Pauli-12Q, Optuna.",
)
story += [
    p("Upgrading from 128D to 256D RotatE lifts the classical RF baseline from 0.7838 to 0.8569 "
      "(+7.3 pp) and the ensemble ceiling from 0.7987 to 0.8581 (+5.9 pp), confirming embedding "
      "quality as the dominant performance driver. The structural pattern — QSVC standalone below "
      "classical, ensemble above classical — is preserved."),
]

story += subsection("6.3", "Feature Map Analysis: The Pauli Inversion Effect")
story += make_table(
    ["Feature Map", "QSVC Standalone", "Ensemble PR-AUC", "Delta vs. RF Baseline"],
    [
        ["ZZ (reps=2, genuine 908 s)", "0.7216", "0.7408", "-0.043"],
        ["Pauli (reps=2, genuine 2,619 s)", "0.6343", "0.7987", "+0.015"],
    ],
    col_widths=[2.2*inch, 1.3*inch, 1.3*inch, 1.7*inch],
    caption="Table 5. Feature Map Ablation — RotatE-128D held constant.",
)
story += [
    p("This is the central quantum finding. The Pauli feature map <i>degrades</i> QSVC standalone "
      "performance by 8.7 pp yet <i>improves</i> ensemble performance by 5.8 pp. Both RF and ET "
      "learn axis-aligned partitions of the 299D embedding feature space. The Pauli kernel — with "
      "mixed X, Y, Z, ZZ interactions across all qubit pairs — encodes a fundamentally different "
      "feature space geometry whose errors are less aligned with the tree ensemble's errors. "
      "The stacking meta-learner exploits this diversity."),
    p("This result supports a broader principle: for quantum kernels in ensemble settings, the "
      "relevant optimization target is not standalone quantum accuracy but quantum-classical "
      "prediction <i>decorrelation</i>."),
]
story += img("fig2_pauli_zz.png",
    width_in=5.5,
    caption_text=(
        "Figure 2. Feature map selection reveals a counterintuitive quantum-classical tradeoff. "
        "Switching from ZZ to Pauli lowers standalone QSVC PR-AUC by 8.7 pp (0.7216 to 0.6343) "
        "while raising ensemble PR-AUC by 5.8 pp (0.7408 to 0.7987). Dashed line: best "
        "classical-only baseline (RandomForest, 0.7838)."
    ))

story += subsection("6.4", "VQC Ablation")
story += [
    p("VQC results are reported for completeness; VQC is not included in reported ensembles."),
]
story += make_table(
    ["Optimizer", "Test PR-AUC"],
    [["SPSA", "0.5456"], ["COBYLA", "0.5086"], ["NFT", "0.4782"]],
    col_widths=[3.25*inch, 3.25*inch],
    caption="Table 6. VQC Optimizer Comparison (RealAmplitudes, 8 qubits, 50 iterations).",
)
story += make_table(
    ["Ansatz", "Repetitions", "Test PR-AUC"],
    [
        ["RealAmplitudes", "4", "0.5474"],
        ["RealAmplitudes", "3", "0.5342"],
        ["EfficientSU2",   "3", "0.5173"],
    ],
    col_widths=[2.5*inch, 1.5*inch, 2.5*inch],
    caption="Table 7. VQC Ansatz Comparison (SPSA, 8 qubits, 50 iterations).",
)
story += [
    p("VQC performance is near-random in the current setup. The gap between VQC (~0.55) and QSVC "
      "(0.63-0.72) suggests the variational optimization landscape is challenging at these circuit "
      "depths and iteration budgets. VQC improvement is deferred to future work."),
]

story += subsection("6.5", "Hardware Validation: IBM Quantum Heron")
story += [
    p("IBM Quantum Heron hardware runs were conducted for a 16-qubit Pauli configuration. The "
      "hardware QSVC PR-AUC (~0.634) is consistent with the statevector simulator result (0.6343), "
      "providing cross-validation that the simulator quantum kernel accurately reflects the "
      "hardware-executable computation. Full noise characterization is planned for v2."),
]

# ── 7. Clinical Validation ────────────────────────────────────────────────────
story += section("7", "Clinical Validation")

story += subsection("7.1", "Validation Methodology")
story += [
    p("After identifying top-scoring novel (compound, disease) predictions, we queried "
      "ClinicalTrials.gov for each predicted compound-disease combination using condition, "
      "intervention, and MeSH term filters. Trials in any phase (I-IV) and any status "
      "(completed, recruiting, active) were counted."),
]

story += subsection("7.2", "Validation Results")
story += make_table(
    ["Rank", "Prediction", "Score", "Trials", "Phase", "Verdict"],
    [
        ["1", "Abacavir to Ocular Cancer",    "0.793", "0",              "—",        "False positive (graph artifact)"],
        ["2", "Ezetimibe to Gout",            "0.693", "0 + 4 indirect", "I-II",     "Novel plausible hypothesis"],
        ["3", "Ramipril to Stomach Cancer",   "0.597", "0",              "—",        "No clinical support"],
        ["4", "Losartan to Atherosclerosis",  "0.528", "7+",             "Phase 4",  "Strongly validated"],
        ["5", "Mitomycin to Liver Cancer",    "0.525", "7",              "Phase 2-3","Strongly validated"],
        ["6", "Salmeterol to Liver Cancer",   "0.520", "0",              "—",        "No clinical support"],
    ],
    col_widths=[0.4*inch, 1.7*inch, 0.55*inch, 0.7*inch, 0.75*inch, 2.4*inch],
    caption="Table 8. ClinicalTrials.gov Validation of Top Novel Predictions.",
)

story += subsection("7.3", "Score-Validity Inversion")
story += [
    p("The highest-scoring prediction (Abacavir to ocular cancer, 0.793) has zero clinical trial "
      "support, while the two most strongly validated predictions (Losartan to atherosclerosis, "
      "Mitomycin to liver cancer) score only 0.52-0.53. This <i>score-validity inversion</i> is a "
      "<i>false positive elevation</i> problem specific to multi-relational graph structure."),
    p("<b>Root cause.</b> Abacavir is a frontline HIV antiretroviral. HIV infection is associated "
      "with elevated incidence of ocular malignancies through immunosuppression pathways. Hetionet "
      "encodes these comorbidity relationships, and RotatE embeddings trained on the full graph "
      "absorb this structural proximity — even though the pathway from HIV antiretroviral action "
      "to ocular cancer treatment is pharmacologically implausible. This is precisely the failure "
      "mode the MoA feature module is designed to address."),
]
story += img("fig3_clinical.png",
    width_in=6.0,
    caption_text=(
        "Figure 3. Score-validity inversion: model prediction score vs. ClinicalTrials.gov "
        "registrations. Highest-scoring prediction (Abacavir, 0.793) has zero clinical support; "
        "validated predictions (Losartan, Mitomycin, 7+ trials) score 0.52-0.53. This inversion "
        "motivates the MoA feature module."
    ))

story += subsection("7.4", "Novel Hypothesis: Ezetimibe for Gout")
story += [
    p("Ezetimibe is a selective cholesterol absorption inhibitor acting at the NPC1L1 transporter. "
      "While no clinical trial directly investigates Ezetimibe for gout, four trials explore its "
      "anti-inflammatory properties. The biological plausibility is grounded in evidence linking "
      "lipid metabolism to urate transport: serum urate levels correlate with cholesterol, and "
      "NPC1L1 is expressed in proximal tubules relevant to urate reabsorption. This prediction "
      "represents a genuine, low-evidence hypothesis worthy of targeted investigation."),
]

# ── 8. Discussion ─────────────────────────────────────────────────────────────
story += section("8", "Discussion")

story += subsection("8.1", "Why the Ensemble Beats Classical-Only")
story += [
    p("The stacking meta-learner treats QSVC as a base model whose predictions carry signal "
      "orthogonal to RF and ET. RF and ET learn axis-aligned splits in the 299D embedding feature "
      "space; the Pauli quantum kernel encodes similarity in a Hilbert space with inductive bias "
      "unrelated to coordinate-aligned decision boundaries. The meta-learner learns to weight the "
      "QSVC vote as a second opinion from a fundamentally different geometry, improving overall "
      "precision-recall performance."),
    p("The quantitative evidence: the Pauli kernel provides +5.8 pp ensemble gain over ZZ despite "
      "performing -8.7 pp worse in isolation. This decorrelation benefit is independent of "
      "absolute QSVC accuracy, suggesting quantum kernel selection for ensembles should prioritize "
      "<i>classical-quantum error independence</i> over quantum standalone performance."),
]

story += subsection("8.2", "Embedding Quality as the Dominant Driver")
story += [
    p("Comparing Tables 3 and 4: upgrading from 128D to 256D RotatE lifts every model — RF (+7.3 pp), "
      "ET (+6.9 pp), QSVC (+8.8 pp), and ensemble (+5.9 pp). The full-graph training regime "
      "(all 2.25M edges) means entity representations are enriched by all 24 relation types. "
      "A compound's embedding encodes not just which diseases it treats, but which genes it binds, "
      "which pathways those genes participate in, what side effects it produces, and which "
      "pharmacologic class it belongs to. This multi-relational context is what makes RotatE "
      "embeddings rich enough to serve as meaningful quantum circuit inputs."),
]

story += subsection("8.3", "Quantum Kernel Scaling and the O(n²) Constraint")
story += [
    p("The fidelity quantum kernel requires O(n<super>2</super>) circuit evaluations: approximately "
      "911K total for 1,208 training samples and 16-qubit Pauli circuit, requiring 2,619 seconds. "
      "For the CbG relation (11,571 positive edges, ~18,500 training pairs), naive full kernel "
      "computation would require ~171M evaluations — infeasible without Nystrom approximation, "
      "hardware parallelism, or dataset subsampling. The 755-edge CtD relation sits at the "
      "practical boundary for exact quantum kernel computation on current simulators."),
]

story += subsection("8.4", "Limitations")
story += [
    p("<b>Single relation.</b> All primary results use CtD (755 positive edges). CpD "
      "(Compound-palliates-Disease, 390 edges) and DrD (Disease-resembles-Disease, 543 edges) "
      "are natural immediate targets.", "bullet"),
    p("<b>Single seed.</b> Results in Tables 3 and 4 are single-seed runs (seed=42). Multi-seed "
      "evaluation (3-5 seeds) is planned for v2.", "bullet"),
    p("<b>Simulation-based quantum.</b> IBM Quantum Heron hardware runs confirm comparable QSVC "
      "scores (~0.634 hardware vs. 0.6343 simulator), but complete hardware benchmarking is "
      "deferred to v2.", "bullet"),
    p("<b>MoA features not yet benchmarked.</b> The MoA module is fully implemented "
      "(kg_layer/moa_features.py). Empirical PR-AUC results with --use_moa_features will be "
      "reported in v2.", "bullet"),
]

# ── 9. Future Work ────────────────────────────────────────────────────────────
story += section("9", "Future Work")
story += [
    p("<b>Immediate (v2):</b> MoA feature benchmark on CtD; CpD relation results; multi-seed "
      "evaluation (5 seeds); degree-heuristic and random baselines."),
    p("<b>Medium-term:</b> Extend to DrD (543 edges), CbG (11,571 edges with Nystrom "
      "approximation); VQC ansatz search with larger budgets; IBM Heron full noise benchmark."),
    p("<b>Longer-term:</b> Extend to DRKG (4.4M edges, 97K entities) with Nystrom-approximated "
      "quantum kernels; variational quantum kernel learning; multi-relational joint training "
      "on CtD + CpD + DrD simultaneously."),
]

# ── 10. Conclusion ────────────────────────────────────────────────────────────
story += section("10", "Conclusion")
story += [
    p("We have presented the first hybrid quantum-classical pipeline that bridges knowledge graph "
      "embeddings with quantum kernel classifiers for biomedical link prediction. On the Hetionet "
      "CtD task — 755 known drug-disease treatments, 47,031 entities, 2.25 million edges across "
      "24 relation types — our system achieves:"),
    p("• <b>PR-AUC 0.7987</b> (primary: RotatE-128D, Pauli-16Q, genuine 2,619 s quantum kernel, "
      "+1.49 pp over best classical)", "bullet"),
    p("• <b>PR-AUC 0.8581</b> (extended: RotatE-256D, Pauli-12Q, Optuna-tuned classical)", "bullet"),
    p("The central finding is a counterintuitive quantum-classical synergy: the Pauli feature map "
      "degrades standalone QSVC PR-AUC by 8.7 pp relative to ZZ but raises ensemble PR-AUC by "
      "5.8 pp, because it generates predictions more decorrelated from classical tree model errors. "
      "This finding reframes the optimization target for quantum kernels in hybrid ensembles: "
      "maximize classical-quantum prediction independence, not standalone quantum accuracy."),
    p("Clinical validation against ClinicalTrials.gov reveals a score-validity inversion problem "
      "and identifies Ezetimibe to gout as a biologically plausible novel hypothesis. We introduce "
      "a ten-feature mechanism-of-action module to address this inversion. This work establishes "
      "a new research direction at the intersection of quantum kernel methods and knowledge graph "
      "reasoning for biomedicine, with a reproducible open pipeline and a clear experimental "
      "roadmap toward multi-relation, multi-seed, and hardware-validated results."),
]

# ── References ─────────────────────────────────────────────────────────────────
story += [rule(), p("References", "h1")]
refs = [
    "[1] Himmelstein, D.S. et al. (2017). Systematic integration of biomedical knowledge prioritizes "
        "drugs for repurposing. eLife, 6, e26726. https://doi.org/10.7554/eLife.26726",
    "[2] Mayers, M. et al. (2023; updated 2024). Drug repurposing using consilience of knowledge "
        "graph completion methods. bioRxiv. https://doi.org/10.1101/2023.05.12.540594",
    "[3] QKDTI (2025). Quantum kernel-based drug-target interaction prediction. Scientific Reports. "
        "https://doi.org/10.1038/s41598-025-07303-z",
    "[4] Kruger, D.M. et al. (2023). Quantum machine learning framework for virtual screening. "
        "Machine Learning: Science and Technology. https://doi.org/10.1088/2632-2153/acb900",
    "[5] Deep learning-based drug repurposing using KG embeddings and GraphRAG (2025). bioRxiv. "
        "https://doi.org/10.64898/2025.12.08.693009",
    "[6] Large-scale quantum computing framework enhances drug discovery (2026). bioRxiv. "
        "https://doi.org/10.64898/2026.02.09.704961",
    "[7] Hybrid classical-quantum pipeline for real-world drug discovery (2024). bioRxiv. "
        "https://doi.org/10.1101/2024.01.08.574600",
    "[8] Sun, Z. et al. (2019). RotatE: Knowledge graph embedding by relational rotation in complex "
        "space. ICLR 2019. https://arxiv.org/abs/1902.10197",
    "[9] Ali, M. et al. (2021). PyKEEN 1.0: A Python library for training and evaluating knowledge "
        "graph embeddings. JMLR, 22(82). https://jmlr.org/papers/v22/20-1531.html",
    "[10] Schuld, M. & Petruccione, F. (2021). Machine Learning with Quantum Computers. Springer. "
         "https://doi.org/10.1007/978-3-030-83098-4",
    "[11] Qiskit contributors (2023). Qiskit: An open-source framework for quantum computing. "
         "https://doi.org/10.5281/zenodo.2573505",
]
for ref in refs:
    story.append(p(ref, "ref"))

# ── Appendix B — Primary Run Config ──────────────────────────────────────────
story += [PageBreak(), p("Appendix B — Full Configuration: Primary Run (0.7987)", "appendix_h")]
story += make_table(
    ["Parameter", "Value"],
    [
        ["relation",             "CtD"],
        ["embedding_method",     "RotatE"],
        ["embedding_dim",        "128"],
        ["embedding_epochs",     "200"],
        ["full_graph_embeddings","True"],
        ["qml_feature_map",      "Pauli"],
        ["qml_feature_map_reps", "2"],
        ["qml_dim",              "16"],
        ["qml_pre_pca_dim",      "24"],
        ["qsvc_C",               "0.1"],
        ["qsvc_nystrom_m",       "None (full kernel)"],
        ["negative_sampling",    "hard"],
        ["ensemble_method",      "stacking"],
        ["tune_classical",       "True"],
        ["fast_mode",            "True"],
        ["random_state",         "42"],
        ["QSVC kernel compute time", "2,618.6 s"],
        ["Result file timestamp","optimized_results_20260216-100431.json"],
    ],
    col_widths=[3.25*inch, 3.25*inch],
)

# ── Appendix C — Extended Run Config ─────────────────────────────────────────
story += [p("Appendix C — Full Configuration: Extended Run (0.8581)", "appendix_h")]
story += make_table(
    ["Parameter", "Value"],
    [
        ["relation",             "CtD"],
        ["embedding_method",     "RotatE"],
        ["embedding_dim",        "256"],
        ["embedding_epochs",     "250"],
        ["full_graph_embeddings","True"],
        ["qml_feature_map",      "Pauli"],
        ["qml_feature_map_reps", "1"],
        ["qml_dim",              "12"],
        ["qml_pre_pca_dim",      "0 (none)"],
        ["qsvc_C",               "0.6756 (Optuna-tuned)"],
        ["qsvc_nystrom_m",       "None"],
        ["negative_sampling",    "hard"],
        ["ensemble_method",      "stacking"],
        ["tune_classical",       "True"],
        ["fast_mode",            "True"],
        ["random_state",         "42"],
        ["QSVC kernel (first trial)", "~99 s"],
        ["Kernel reuse across Optuna","Yes (83 trials, fixed quantum kernel)"],
        ["Result file timestamp","optimized_results_20260323-134844.json"],
    ],
    col_widths=[3.25*inch, 3.25*inch],
)

# ── Build ─────────────────────────────────────────────────────────────────────
def on_page(canvas, doc):
    """Header/footer on every page."""
    canvas.saveState()
    w, h = letter
    # footer line
    canvas.setStrokeColor(C_RULE)
    canvas.setLineWidth(0.5)
    canvas.line(0.75*inch, 0.65*inch, w - 0.75*inch, 0.65*inch)
    # footer text
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(C_LIGHT)
    canvas.drawString(0.75*inch, 0.48*inch,
        "Robinson, Jack, Beale — Quantum Kernel Link Prediction on Biomedical KGs — arXiv 2026")
    canvas.drawRightString(w - 0.75*inch, 0.48*inch, f"Page {doc.page}")
    canvas.restoreState()

doc = SimpleDocTemplate(
    OUT,
    pagesize=letter,
    leftMargin=0.85*inch, rightMargin=0.85*inch,
    topMargin=0.9*inch,   bottomMargin=0.85*inch,
    title="Quantum Kernel Link Prediction on Biomedical Knowledge Graphs",
    author="Kevin Robinson, Mark Jack, Jonathan Beale",
    subject="Quantum Machine Learning, Drug Repurposing, Knowledge Graphs",
    keywords="QSVC, RotatE, Hetionet, drug repurposing, quantum kernels, link prediction",
)

doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"✓ Saved: {OUT}")

# Dashboard presentation flow and glossary

Use this as a one-page guide for presenting the dashboard (~2 minutes) and for explaining technical terms in plain language.

---

## Presentation flow (~2 minutes)

### 1. Overview (≈30 sec)

- **Opening:** “This is a hybrid quantum–classical system for link prediction on a biomedical knowledge graph: we predict whether a compound is likely to treat a disease.”
- Point to the blue **“How this dashboard works”** callout: “The sidebar has the full workflow; here we’ll just hit the highlights.”
- Keep it short: one line on **PR-AUC** (how we measure ranking quality) and **ideal vs noisy** (perfect vs real-world-like quantum) is enough. Don’t open every expander.

---

### 2. Run benchmarks → Generate demo results (≈20 sec)

- “We can run the full pipeline or, in environments like Hugging Face, use **Generate demo results** so you see data without running the heavy stack.”
- Click **Generate demo results** once.
- “That gives us a latest run and a full model ranking so the next tab makes sense.”

---

### 3. Results (≈70 sec)

- “This is where the story is in 2 minutes.”
- **Latest run snapshot:** “One representative quantum setup: QSVC, 12 qubits, ZZ feature map, simulator.”
- **Models in this run:** “We actually compare six models: three classical (e.g. LogisticRegression, RandomForest, Ensemble), two quantum (QSVC variants), and one hybrid that combines them.”
- **Full model ranking table:** “Ranked by PR-AUC. Classical and hybrid are at the top today; quantum is lower but uses far fewer parameters—that’s the scalability angle.”
- **Metrics by model:** “Same numbers, per model, so you can compare PR-AUC, accuracy, and time at a glance.”
- **Optional:** “We also have **Refresh data** and **Refresh full model ranking** when new runs or uploads come in.”

---

## Full glossary (layman definitions)

Use these when someone asks what a term means and why it matters for this project.

| Term | Layman definition |
|------|-------------------|
| **Hybrid quantum–classical** | We use **both** normal computers and quantum-style computation. Classical parts do the heavy data and training; quantum parts handle a specific “similarity” step. The project tests whether adding that quantum step helps. |
| **Link prediction** | Guessing whether a **connection** between two things (here: compound and disease) **should exist**—e.g. “Does this compound treat this disease?” We’re not discovering new drugs; we’re ranking which pairs are most worth investigating. |
| **Knowledge graph** | A big **network of facts**: nodes = entities (compounds, diseases, genes, etc.), edges = relationships (“treats”, “associated with”). The project uses one such graph (Hetionet) as the source of truth. |
| **Compound** | A drug or molecule. |
| **Disease** | A condition (e.g. a medical disease). |
| **CtD (Compound–treats–Disease)** | The relationship we’re predicting: “Does this compound treat this disease?” |
| **PR-AUC (Precision–Recall AUC)** | A **single number** that says how good the model is at putting true links higher than non-links, especially when positive examples are rare. **Higher = better ranking.** We use it to compare classical vs quantum vs hybrid. |
| **Ideal vs noisy** | **Ideal** = perfect, noiseless quantum simulation. **Noisy** = simulation that mimics real quantum hardware errors. Comparing them shows whether the method is sensitive to real-world imperfection or only works in theory. |
| **Generate demo results** | A **one-click way** to fill the dashboard with **sample data** (latest run + full model ranking) **without** running the full pipeline. Lets you demo the dashboard (e.g. on Hugging Face) when the heavy stack isn’t available. |
| **Full model ranking** | A **ordered list of all models** (e.g. six: three classical, two quantum, one hybrid) **sorted by performance** (e.g. PR-AUC). It answers: “Which model did best in this run?” |
| **Latest run snapshot** | A **short summary of the last benchmark**: which quantum setup was used (model type, qubits, feature map, where it ran). It’s the “settings” summary for that run. |
| **QSVC** | **Quantum Support Vector Classifier.** A classifier that uses a **quantum circuit** to compute “similarity” between data points instead of a classical formula. In this project it’s our main quantum model we compare to classical ones. |
| **Qubits** | The **basic units of quantum information** (like bits, but quantum). Here, **number of qubits** = size of the “quantum view” of each data point. More qubits = more capacity but more cost; we use 12 as a practical choice. |
| **Feature map (e.g. ZZ)** | The **recipe** that turns our classical numbers into a quantum state the circuit can use. **ZZ** is one such recipe (involving two-qubit rotations). It defines how data is encoded in the quantum part. |
| **Simulator** | Running the **quantum circuit on a classical computer** that *simulates* quantum behavior—no real quantum hardware. Lets us test the pipeline quickly and cheaply. |
| **Classical (model)** | Standard ML (e.g. logistic regression, random forest)—no quantum. |
| **Quantum (model)** | Uses a quantum circuit (e.g. QSVC) for part of the computation. |
| **Hybrid (model)** | **Combines** one classical and one quantum model (e.g. weighted average of their scores). In the dashboard we compare all three. |
| **Metrics by model** | The **same performance numbers** (e.g. PR-AUC, accuracy, time) **shown per model** so you can compare “this model vs that model” at a glance without opening the big table. |
| **Parameters** | The knobs the model learns (e.g. weights). “Quantum uses far fewer parameters” means less to train and store—one reason people care about scalability as data grows. |
| **Refresh data** | Reloads the cached “latest run” and history so new runs or uploads show up. |
| **Refresh full model ranking** | Reloads the table of all models from the latest results file. Use after a new benchmark or upload so the dashboard is up to date. |
| **Backend** | Where the quantum circuit runs—e.g. “simulator” or a real device name. |
| **Noise (noise model)** | Whether we simulate real-device errors (noisy) or not (ideal). |
| **Embedding** | A fixed-size vector that represents an entity (e.g. compound or disease) learned from the graph; used as input to classical and quantum models. |
| **VQC** | **Variational Quantum Classifier.** A quantum model that tunes circuit “knobs” with a classical optimizer. We focus on QSVC in the main narrative but VQC is in the glossary. |

---

## Quick reference: tab order for the 2‑minute flow

1. **1. Overview** — What this is; point to “How this dashboard works.”
2. **2. Run benchmarks** — Click **Generate demo results**.
3. **3. Results** — Latest run snapshot → Models in this run → Full model ranking → Metrics by model.

---

*Last updated for the dashboard with full model ranking, Metrics by model, and Latest run snapshot (Models in this run).*

```markdown
# Design System Specification: Quantum-Biological Precision

## 1. Overview & Creative North Star
**Creative North Star: "The Observed Observer"**

This design system is built for the intersection of clinical biology and quantum mechanics. It rejects the "dashboard-in-a-box" aesthetic in favor of a high-fidelity, laboratory-grade environment. The visual language mimics the precision of an electron microscope and the multidimensionality of a quantum processor.

To move beyond generic SaaS layouts, we utilize **Intentional Asymmetry**. Rather than a rigid 12-column grid, layouts should prioritize "focal density"—areas of high-complexity data (quantum circuits) contrasted against expansive, breathable "voids" of dark slate (`surface`). We use overlapping elements and depth layering to suggest that data isn't just on the screen, but is being "rendered" in real-time from a deep compute layer.

---

## 2. Colors & Surface Philosophy

### The Palette
The foundation is built on deep, oceanic slates and sharp, luminescent accents.
- **Primary (`#7bd0ff`):** Used for "Active Observation" states and primary navigation paths.
- **Secondary (`#d0bcff`):** Reserved for quantum state indicators (superposition, entanglement).
- **Tertiary (`#3cddc7`):** The "Biological Vitality" accent, used for genomic data and successful execution paths.

### The "No-Line" Rule
**Borders are prohibited for structural sectioning.** We do not use 1px solid lines to separate a sidebar from a main content area. Instead:
- Use **Background Shifts**: A `surface-container-low` (`#131b2e`) sidebar sitting against a `surface` (`#0b1326`) background.
- Use **Vertical Rhythms**: Use the `16` (3.5rem) spacing token to create a clean, mental break between sections.

### Surface Hierarchy & Nesting
Treat the UI as a series of physical, semi-conductive layers.
1.  **Base Layer:** `surface` (#0b1326) – The infinite canvas.
2.  **Sectional Layer:** `surface-container-low` (#131b2e) – Broad grouping of related tools.
3.  **Component Layer:** `surface-container-high` (#222a3d) – Individual cards or data modules.
4.  **Interaction Layer:** `surface-container-highest` (#2d3449) – Active hover states or selected nodes.

### The "Glass & Gradient" Rule
For floating panels (modals, quantum circuit inspectors), use **Glassmorphism**. Apply `surface-variant` (`#2d3449`) at 60% opacity with a `20px` backdrop blur. 
**Signature Texture:** Main Action Buttons should use a subtle linear gradient from `primary` (`#7bd0ff`) to `on-primary-container` (`#008abb`) at a 135-degree angle to provide a "machined" metallic finish.

---

## 3. Typography
We pair the utilitarian precision of **Inter** with the architectural character of **Space Grotesk**.

- **Display & Headlines (Space Grotesk):** These are our "Headers of Authority." Use `display-lg` for macro-view titles. The wide apertures and geometric shapes of Space Grotesk suggest a futuristic, scientific rigor.
- **Technical & Body (Inter):** Inter is used for all data-dense environments. Its high X-height ensures readability in complex tables.
- **Data Mono (Inter - Variable):** For quantum gates and genomic sequences, use Inter with `font-feature-settings: "tnum", "onum"` to ensure tabular spacing for numbers.

---

## 4. Elevation & Depth

### The Layering Principle
Depth is achieved through **Tonal Stacking**. To lift an element, move it up one tier in the `surface-container` scale. A card using `surface-container-lowest` placed on a `surface-container` background creates a "recessed" look, ideal for input areas or terminal shells.

### Ambient Shadows
Traditional shadows are too heavy for a quantum interface. If an element must float (e.g., a context menu):
- **Shadow:** 0px 10px 40px
- **Color:** `on-surface` (`#dae2fd`) at 6% opacity. This creates a "glow" of dark energy rather than a heavy drop-shadow.

### The "Ghost Border" Fallback
If high-density data requires containment (e.g., cell borders in a complex table), use a **Ghost Border**: `outline-variant` (`#45464d`) at **15% opacity**. It should be felt, not seen.

---

## 5. Components

### Complex Data Tables
- **Header:** `surface-container-high` background, `label-md` uppercase typography.
- **Rows:** No dividers. Use a subtle `surface-container-lowest` background shift on hover.
- **Density:** Use spacing `2` (0.4rem) for cell padding to maximize data visibility.

### Quantum Circuit Diagrams
- **Wires:** Use `outline` (`#909097`) at 40% opacity. 
- **Gates:** Use `surface-container-highest` with a `sm` (0.125rem) roundedness. 
- **Active State:** Pulse effect using a 2px outer glow of `tertiary` (`#3cddc7`).

### Buttons
- **Primary:** Gradient (Primary to On-Primary-Container), `md` (0.375rem) roundedness.
- **Secondary:** Ghost style. No background, `outline` border at 20%, `primary` text.
- **Tertiary:** Text only, `title-sm` weight, `tertiary` color.

### Input Fields
- Avoid boxes. Use a "bottom-line only" approach or a slightly darker `surface-container-lowest` fill with no border.
- **Error State:** Use `error` (`#ffb4ab`) for the label and a 1px glow on the bottom-line.

---

## 6. Do’s and Don’ts

### Do
- **Do** use `tertiary` (`#3cddc7`) for any biological "success" or "positive growth" data.
- **Do** lean into `surface-bright` (`#31394d`) for tooltips to make them pop against the dark void.
- **Do** allow content to bleed off the edges in certain modules to suggest a continuous stream of data.

### Don't
- **Don't** use pure black (`#000000`). It kills the depth of the slate palette.
- **Don't** use the `full` (9999px) roundedness scale for anything other than status chips. We want "Scientific Precision," which favors sharp or slightly softened corners (`sm` or `md`).
- **Don't** use standard red for errors unless it’s a critical system failure. For minor data discrepancies, use `secondary` (`#d0bcff`) to keep the palette cool.

---

## 7. Interaction Micro-copy
In a scientific context, clarity is paramount. 
- **Active State:** "Observing..."
- **Quantum Calculation:** "Resolving Probability..."
- **Data Load:** "Sequencing..."

Use `label-sm` for these status indicators, placed in the bottom-right of the affected container.```
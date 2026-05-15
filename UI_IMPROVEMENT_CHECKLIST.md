# UI/UX Improvement Action Checklist

**Project**: Hybrid QML-KG  
**Created**: 2026-04-22  
**Priority Level**: Medium (Improve before public launch)  
**Estimated Total Time**: 49 hours

---

## Phase 1: Critical Fixes (18 hours, ~1 week)

### ✅ Task 1.1: Form Validation Feedback (4 hours)

**Component**: `/frontend/components/predict-form.tsx`

**What to implement:**
- [ ] Add autocomplete dropdown
  - User types "ib..." → Shows: "ibuprofen", "ibandronate", "ibandronic acid"
  - Click to select
  - Show count: "2 matches found"
  
- [ ] Add real-time validation indicators
  - ✓ green check if drug found in database
  - ✗ red X if not found
  - ? gray ? if still loading
  
- [ ] Add loading spinner to submit button
  - When clicked, button shows spinner
  - Text changes: "Predict link" → "Calculating..."
  - Disable button to prevent double-submit
  
- [ ] Add error message display
  - Location: Below form
  - Style: Red background, warning icon
  - Example: "Drug 'xyz123' not found in Hetionet. Try 'ibuprofen' or DrugBank ID 'DB00945'"

**Testing**:
- [ ] Test valid drug name
- [ ] Test invalid drug name
- [ ] Test autocomplete dropdown
- [ ] Test loading state
- [ ] Test error message

**Files to modify**:
```
frontend/components/predict-form.tsx
frontend/app/page.tsx
frontend/app/predict/page.tsx (if separate form)
frontend/lib/api.ts (add drug lookup endpoint)
```

---

### ✅ Task 1.2: Make Experiments Table Interactive (6 hours)

**Component**: Experiments page table

**What to implement:**
- [ ] **Sorting**
  - Click column header to sort ascending/descending
  - Add arrow indicator (↑↓) on active column
  - Columns to support: Model, PR-AUC, Type
  
- [ ] **Filtering**
  - Dropdown filter: "All models" / "Quantum only" / "Classical only" / "Ensemble only"
  - Updates table instantly
  
- [ ] **Expandable rows**
  - Click row or "Show details" button
  - Expands to show:
    - ROC-AUC, Precision, Recall, F1
    - Confusion matrix (2x2 table)
    - Training time
    - Hyperparameters
  
- [ ] **Export button**
  - "Download as CSV" button at top
  - Exports visible table with current filters

**Testing**:
- [ ] Click column header → sorts correctly
- [ ] Filter dropdown changes results
- [ ] Row expands to show details
- [ ] CSV export works

**Files to modify**:
```
frontend/app/experiments/page.tsx
frontend/components/experiment-table.tsx (create if needed)
frontend/lib/api.ts (add export endpoint)
```

---

### ✅ Task 1.3: Knowledge Graph Visualization (8 hours)

**Component**: `/frontend/app/knowledge-graph/page.tsx`

**What to implement:**
- [ ] **Interactive D3 graph**
  - Install: `npm install d3 @types/d3`
  - Show 50-100 sample nodes (subset of 445 entities)
  - Nodes = entities (color by type: compound=blue, gene=green, disease=red, process=yellow)
  - Edges = relationships
  - Hover = tooltip with entity name, type, ID
  - Click node = expand neighborhood (show connected nodes)
  - Drag = move nodes around
  
- [ ] **Search box**
  - Input field: "Search entity..."
  - Real-time search as user types
  - Results: "Found 3 matches"
  - Click result = highlight in graph and center on it
  
- [ ] **Relation type tooltips**
  - Add definitions for all 24 relation types:
    - CpBP = "Compound-related via Biological Process"
    - CdD = "Compound-Disease Direct"
    - CrC = "Compound-related via Gene"
    - Etc.
  - Show on hover over relation type tag
  
- [ ] **Path finding (optional, stretch goal)**
  - Input: "Find path from X to Y"
  - Shows: All paths connecting them
  - Example: Ibuprofen → [CpBP] → Inflammation → [DaD] → Rheumatoid Arthritis

**Testing**:
- [ ] Graph renders without errors
- [ ] Nodes are clickable
- [ ] Hover shows tooltips
- [ ] Search finds entities
- [ ] Zoom/pan works

**Files to modify**:
```
frontend/app/knowledge-graph/page.tsx
frontend/components/graph-visualization.tsx (create new)
frontend/lib/graph-utils.ts (create new - D3 helper functions)
```

**Dependencies to add**:
```bash
npm install d3 @types/d3
```

---

## Phase 2: Important Improvements (17 hours, ~2 weeks)

### 🟡 Task 2.1: Result Display Enhancement (4 hours)

**Component**: Prediction result card

**What to implement:**
- [ ] **Large score display**
  - Show percentage prominently: "78%"
  - Font size: 48px
  - Color-coded (green ≥70%, yellow 40-70%, gray <40%)
  
- [ ] **Interpretation badge**
  - Below score
  - "Strong evidence" / "Moderate signal" / "Weak evidence"
  - Icon + text
  
- [ ] **Model used**
  - "Predicted by: Ensemble-QC-stacking (Pauli)"
  - Small gray text
  
- [ ] **Confidence interval**
  - "±5%" or confidence range
  - Helps users understand uncertainty
  
- [ ] **Related actions**
  - "Save this prediction" button
  - "View similar results" link
  - "Try another prediction" button

**Testing**:
- [ ] Result displays after form submission
- [ ] Score color matches threshold
- [ ] All metadata visible

**Files to modify**:
```
frontend/components/prediction-result.tsx (create new)
frontend/app/predict/page.tsx
```

---

### 🟡 Task 2.2: Quantum Page Visualizations (6 hours)

**Component**: `/frontend/app/quantum/page.tsx`

**What to implement:**
- [ ] **Quantum circuit diagram**
  - Show example 16-qubit Pauli feature map circuit
  - Display: Qubits (vertical lines), gates (boxes), circuit depth (total gates)
  - Use SVG or existing library (e.g., `react-quantum-circuit` if available)
  
- [ ] **Feature map explanation**
  - Diagram: Classical data → Pauli gates → Quantum state
  - Text: "reps=2 means the feature map is applied 2 times"
  
- [ ] **Error mitigation visualization**
  - Before/After comparison
  - Show: Raw result vs Mitigated result
  - Example: "Raw PR-AUC: 0.70 → Mitigated: 0.72 (+2%)"
  
- [ ] **Performance comparison chart**
  - Bar chart: Quantum vs Classical vs Ensemble
  - X-axis: Models
  - Y-axis: PR-AUC score
  - Highlight best model
  
- [ ] **Hyperparameter tuning (optional, stretch)**
  - Sliders: C (0.01-10), reps (1-5)
  - Dropdown: Feature map (Pauli, ZZ, EfficientSU2)
  - Live preview: "Estimated accuracy: 72.5%"

**Testing**:
- [ ] Circuit diagram renders
- [ ] Charts display correctly
- [ ] Sliders work (if implemented)

**Files to modify**:
```
frontend/app/quantum/page.tsx
frontend/components/circuit-diagram.tsx (create new)
frontend/components/performance-chart.tsx (create new)
```

---

### 🟡 Task 2.3: Mobile Responsiveness Testing & Fixes (4 hours)

**Testing Viewports**:
- [ ] 375px (mobile)
- [ ] 768px (tablet)  
- [ ] 1280px (desktop, already working)

**Issues to check and fix**:
- [ ] Sidebar converts to hamburger menu on mobile
- [ ] Form inputs are full-width on mobile
- [ ] Tables are scrollable on mobile
- [ ] Buttons are 48px+ tall (touch target)
- [ ] Text is readable (not cramped)
- [ ] Images scale properly
- [ ] Navigation is accessible

**Tools**:
```bash
# Test in browser DevTools
# Open DevTools (F12) → Toggle device toolbar (Ctrl+Shift+M)
# Test at different viewport sizes
```

**Files to modify**:
```
frontend/app/globals.css (add responsive classes if needed)
frontend/components/sidebar.tsx (add mobile menu)
Various page components (check responsive layout)
```

---

### 🟡 Task 2.4: Accessibility Audit & Fixes (3 hours)

**Tools to use**:
- [ ] WAVE extension (Chrome)
- [ ] axe DevTools (Chrome)
- [ ] Lighthouse (built-in Chrome DevTools)

**Checks to perform**:
- [ ] Color contrast ratios (must be 4.5:1 for body text)
- [ ] Keyboard navigation (Tab through entire site)
- [ ] Focus indicators visible
- [ ] Form labels accessible
- [ ] Alt text on images
- [ ] ARIA roles where needed

**Fixes to implement**:
- [ ] Increase contrast if needed
  ```css
  /* Example: Ensure 4.5:1 ratio */
  color: #e0e0e0; /* was #999999 */
  ```
- [ ] Add visible focus indicators
  ```css
  :focus {
    outline: 2px solid #00d4ff;
    outline-offset: 2px;
  }
  ```
- [ ] Label all form inputs
  ```jsx
  <label htmlFor="drug-input">Drug/Compound</label>
  <input id="drug-input" type="text" />
  ```

**Files to modify**:
```
frontend/app/globals.css
Various component files
frontend/app/layout.tsx (ensure semantic HTML)
```

---

## Phase 3: Nice-to-Have Features (14 hours, Q2 2026)

### 💚 Task 3.1: Prediction History/Bookmarks (4 hours)

**What to implement**:
- [ ] Save predictions to browser localStorage
- [ ] Show "Recent predictions" sidebar
- [ ] Allow bookmarking favorites
- [ ] Compare side-by-side results

---

### 💚 Task 3.2: Advanced Export Options (4 hours)

**What to implement**:
- [ ] Export as CSV (results + metadata)
- [ ] Export as PDF (formatted report)
- [ ] Share link (generate shareable URL)
- [ ] Copy to clipboard

---

### 💚 Task 3.3: Hyperparameter Tuning UI (6 hours)

**What to implement**:
- [ ] Interactive sliders for quantum parameters
- [ ] Real-time accuracy preview
- [ ] Save custom configurations
- [ ] Compare different setups

---

## Testing Checklist

### Before Submitting for Review

**Functionality**:
- [ ] Form validation works (valid/invalid inputs)
- [ ] Loading states display (spinner, disabled button)
- [ ] Error messages show (user-friendly)
- [ ] Results display correctly
- [ ] Sorting/filtering works
- [ ] Graph visualization renders
- [ ] Search finds entities

**Design**:
- [ ] Colors match design system
- [ ] Spacing consistent
- [ ] Typography correct
- [ ] No layout broken on different sizes
- [ ] Dark theme looks good

**Accessibility**:
- [ ] Tab navigation works
- [ ] Color contrast sufficient
- [ ] Focus indicators visible
- [ ] Screen reader compatible

**Performance**:
- [ ] No console errors
- [ ] Page loads in <3 seconds
- [ ] Interactions respond instantly
- [ ] No memory leaks

---

## Rollout Plan

### Week 1: Phase 1 (Critical Fixes)
```
Mon-Tue: Tasks 1.1 + 1.2 (Form + Table)
Wed-Thu: Task 1.3 (Knowledge Graph)
Fri: Testing + code review
```

### Week 2-3: Phase 2 (Important)
```
Week 2: Tasks 2.1 + 2.2 (Results + Quantum)
Week 3: Tasks 2.3 + 2.4 (Mobile + Accessibility)
Final: Testing + QA
```

### Q2 2026: Phase 3 (Nice-to-Have)
```
As time/priority allows
```

---

## Success Criteria

After implementing Phase 1:
- ✅ Users get feedback on form input
- ✅ Tables are interactive (sort/filter)
- ✅ Knowledge graph is visualizable
- ✅ Overall UX score: 8/10

After implementing Phase 2:
- ✅ Results are clear and detailed
- ✅ Quantum details are understandable
- ✅ Mobile works well
- ✅ Accessibility standards met
- ✅ Overall UX score: 8.5/10

After implementing Phase 3:
- ✅ Users can manage predictions
- ✅ Advanced features available
- ✅ Professional polish complete
- ✅ Overall UX score: 9/10

---

## Resources & Tools

**Libraries to add**:
```bash
npm install d3 @types/d3                  # Knowledge graph
npm install chart.js react-chartjs-2     # Charts (if not already there)
npm install axios                         # HTTP requests (if not already there)
```

**Browser extensions for testing**:
- WAVE (accessibility)
- axe DevTools (accessibility)
- Lighthouse (built-in)
- React DevTools
- Network DevTools (F12)

**Documentation links**:
- [Material Design 3 - Text Input](https://m3.material.io/components/text-fields)
- [WAI-ARIA - Form Labels](https://www.w3.org/WAI/tutorials/forms/labels/)
- [D3.js Force Simulation](https://d3js.org/d3-force)

---

## Time Estimate Summary

| Phase | Task | Hours | Week |
|-------|------|-------|------|
| 1 | Form validation | 4 | 1 |
| 1 | Table interactivity | 6 | 1 |
| 1 | Knowledge graph viz | 8 | 1 |
| 2 | Results display | 4 | 2 |
| 2 | Quantum visuals | 6 | 2 |
| 2 | Mobile testing | 4 | 3 |
| 2 | Accessibility | 3 | 3 |
| 3 | Prediction history | 4 | Q2 |
| 3 | Export options | 4 | Q2 |
| 3 | Hyperparameter UI | 6 | Q2 |
| | **TOTAL** | **49** | **3 weeks** |

---

## Next Steps

1. **Review** this checklist with product team
2. **Prioritize** based on timeline and resources
3. **Assign** tasks to developers
4. **Create** GitHub issues for each task
5. **Schedule** code review checkpoints
6. **Test** before deploying to production

---

**Questions?** Contact the product/design team.

**Ready to start?** Pick Task 1.1 (Form validation) - it's the quickest win and highest impact!


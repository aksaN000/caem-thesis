# Build Instructions

This report uses `biblatex` with the `biber` backend. Any change to citations in the `.tex` files — or any edit to `bibliography/references.bib` — requires the biber cache to be refreshed, otherwise new citations will show as `[?]` and cross-references may fail to resolve.

## Full-build sequence

Run from the report root (`pre thesis 1 report/`):

```bash
pdflatex -interaction=nonstopmode main.tex
biber main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The **three `pdflatex` passes** are not optional:

1. First pass — writes out `main.bcf` (the biber control file), `main.aux` (label/reference table), and leaves every citation and `\cref` unresolved.
2. `biber main` — reads `main.bcf`, pulls entries from `bibliography/references.bib`, produces `main.bbl`.
3. Second `pdflatex` — links the bibliography into the document. Cross-references that span chapters (e.g. Chapter 5 → `\cref{sec:purity-theorem}` in Chapter 4) still show as `??` because labels from this pass aren't yet in `main.aux`.
4. Third `pdflatex` — resolves remaining cross-references.

## When biber rerun is specifically needed

You must rerun `biber` (not just the two `pdflatex` passes) whenever any of the following changes:

- A new `\citep{…}` / `\citet{…}` / `\cite{…}` key is added anywhere in the `.tex` sources that wasn't previously in the document.
- An entry in `bibliography/references.bib` is edited (author, title, year, etc.).
- `\addbibresource{…}` is changed in `main.tex`.
- A `.bib` file is added or removed.

Changes that do **not** require rerunning biber (two `pdflatex` passes suffice):

- Text edits that do not introduce new citation keys.
- New `\label{…}` or `\cref{…}` / `\ref{…}` cross-references.
- New figures or tables.

## Troubleshooting checklist

If a citation renders as `[?]` after a full build:

1. Confirm the citation key exists in `bibliography/references.bib` — `grep <key> bibliography/references.bib`.
2. Confirm the key is spelled identically in the `.tex` source and the `.bib` file.
3. Delete `main.aux`, `main.bcf`, `main.bbl`, `main.blg`, `main.run.xml` and run the full four-command sequence again.

If a `\cref` or `\ref` renders as `??`:

1. Run `pdflatex` twice more (four passes total). Forward references can need two extra passes when the referenced label lives in a later chapter.
2. If still unresolved, the label does not exist anywhere in the sources. Use `grep -rn '\\label{<name>}' chapters/` to confirm.

## Notes

- This report imports chapters via `\include{chapters/chapter_N}` from `main.tex`, so edits in `chapters/*.tex` propagate automatically — no separate include step.
- The per-chapter standalone files (`test_ch3.tex`, `test_ch4.tex`, `test_ch4_noalg.tex`) have their own `.aux`/`.log` pairs and can be built independently for fast iteration without rebuilding the full thesis.

# Build report

This report uses the LNCS class from `docs/latex-sample-template` and `fontspec`
for Vietnamese text. Build it with **XeLaTeX** (not `pdflatex`).

From the repository root in PowerShell, add the bundled template directory to
LaTeX's class search path, then compile with XeLaTeX:

```powershell
cd .\docs\reports
$env:TEXINPUTS = (Resolve-Path ..\latex-sample-template).Path + ';' + $env:TEXINPUTS
xelatex -interaction=nonstopmode -halt-on-error car-analysis-report.tex
xelatex -interaction=nonstopmode -halt-on-error car-analysis-report.tex
```

Do not use `pdflatex`: the report uses `fontspec` for Vietnamese text. Run
XeLaTeX twice so figure references and PDF metadata are updated. The output is
`car-analysis-report.pdf` in this directory.

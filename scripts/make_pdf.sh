#!/usr/bin/env bash
# Rebuild INTERVIEW.pdf from INTERVIEW.md (needs the .venv with `markdown` + Google Chrome).
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python - <<'PY'
import markdown
body = markdown.markdown(open("INTERVIEW.md").read(), extensions=["fenced_code","tables","sane_lists"])
CSS = "@page{size:A4;margin:15mm 16mm}body{font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:10.5pt;line-height:1.5;color:#1a1a1a}h1{font-size:21pt;border-bottom:2px solid #333;padding-bottom:6px}h2{font-size:15pt;margin-top:24px;border-bottom:1px solid #ccc;padding-bottom:3px}h3{font-size:12.5pt;margin-top:18px;color:#14365a;page-break-after:avoid}code{font-family:Menlo,Consolas,monospace;font-size:9.5pt;background:#f1f1f3;padding:1px 4px;border-radius:3px}pre{background:#f6f8fa;padding:10px 12px;border-radius:6px;font-size:9pt;page-break-inside:avoid;white-space:pre-wrap}pre code{background:none;padding:0}blockquote{border-left:3px solid #cbd5e0;margin:6px 0;padding:2px 12px;color:#555;font-size:9.8pt}em{color:#6b7280;font-size:9.5pt}hr{border:none;border-top:1px solid #e2e2e2;margin:14px 0}a{color:#2b6cb0;text-decoration:none}"
open("INTERVIEW.html","w").write(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>")
PY
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/INTERVIEW.pdf" "file://$PWD/INTERVIEW.html" 2>/dev/null
rm -f INTERVIEW.html
echo "Rebuilt INTERVIEW.pdf"

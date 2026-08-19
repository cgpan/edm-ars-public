#!/usr/bin/env bash
# Render docs/edm_ars_slides.html to a 16:9 PDF deck.
#
# Chrome headless is used rather than a Python HTML-to-PDF library
# because the diagrams are hand-authored SVG with CSS custom properties;
# weasyprint and friends render them incorrectly. @page in the HTML pins
# 13.333in x 7.5in, so the output is exactly PowerPoint 16:9.
set -euo pipefail
cd "$(dirname "$0")/.."
CHROME="${CHROME:-/c/Program Files/Google/Chrome/Application/chrome.exe}"
[ -x "$CHROME" ] || CHROME="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$(pwd -W 2>/dev/null || pwd)/docs/edm_ars_slides.pdf" \
  "file:///$(pwd -W 2>/dev/null || pwd)/docs/edm_ars_slides.html"
python - <<'PY'
import fitz
d = fitz.open("docs/edm_ars_slides.pdf")
H, W = d[0].rect.height, d[0].rect.width
bad = 0
for i, page in enumerate(d):
    for b in page.get_text("blocks"):
        y0, y1, x1, txt = b[1], b[3], b[2], b[4].strip()
        if not txt or (y0 > H - 0.42*72 and len(txt) < 40):
            continue
        if y1 > H - 0.30*72 or y0 < 0.40*72 or x1 > W - 0.30*72:
            print(f"  OVERFLOW p{i+1}: {txt[:40]!r}"); bad += 1
print(f"{len(d)} pages @ {W:.0f}x{H:.0f}pt | overflow: {bad}")
PY

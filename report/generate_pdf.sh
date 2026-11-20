#!/usr/bin/env bash
# Script to compile finalreport.md into publication-grade finalreport.pdf using Pandoc and XeLaTeX.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Compiling finalreport.md to finalreport.pdf..."
pandoc finalreport.md \
    -o finalreport.pdf \
    --pdf-engine=xelatex \
    -V geometry:margin=1in

echo "Done! Generated $(pwd)/finalreport.pdf"

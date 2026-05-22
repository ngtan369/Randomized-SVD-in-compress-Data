#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "=== Running tests ==="
python src/test_rsvd.py
echo "=== Image compression experiment ==="
python src/experiment_image.py
echo "=== MovieLens-like sparse experiment ==="
python src/experiment_movielens.py
echo "=== Benchmark ==="
python src/benchmark.py
echo "=== Building PDF report ==="
cd report && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
echo "Done. See report/main.pdf"

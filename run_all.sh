#!/usr/bin/env bash
# Reproduces every result and figure. Total runtime: a few minutes on a laptop CPU.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results figures
python3 experiments/e1_bank_frontier.py color results
python3 experiments/e1_bank_frontier.py surface results
python3 experiments/e234_identification.py results 234
python3 experiments/e5_protection.py results
python3 figures/make_figures.py results figures
python3 figures/fig2_schematic.py figures/Fig2_probe.pdf

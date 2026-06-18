"""promptbench.chain — Phase B desain STAGES (rantai multi-agent via KV-cache).

Desain STAGES: prefix yang makin panjang (s1_pc → s6_cross) pada latent_steps
TETAP (top per-agent dari scoreboard) → mengisolasi kontribusi MARGINAL tiap
agent. Bersanding dengan desain CHAINS (runners/bench_chain.py) yang men-sweep
GRID latent_steps per skenario tematik.

Infrastruktur BERSAMA (konsolidasi 2026-06-18):
  - scoring/score_chain.py  : skor terminal + parser_hook (fallback + audit trace).
  - scoring/parsing_hook.py : EXTENSION POINT parser judger/construct (default
                              delegate ke parsers.parse_hypothesis_exprs).
  - diagnostics/collapse.py : detektor KV-growth + degenerasi teks terminal.
  - artifacts.py            : path & format artefak nested.

Disiplin KV = LINEAR penuh (clone-on-transfer via kv_deepcopy).
"""

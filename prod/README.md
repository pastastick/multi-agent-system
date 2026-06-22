# prod/ — QuantaLatent Production Pipeline

Branch `prod/quantalatent-v1`. Pipeline evolusi alpha-mining bersih, terpisah dari
`try/promptbench` (eksperimen). **Status: SKELETON** — lihat [DESIGN.md](DESIGN.md).

Perbaikan inti vs v4: front-end `proposal→design→construct` memakai transfer KV
**NO-CROP** sehingga hipotesis & palette asli merambat utuh antar agent (bukan
hanya vektor laten yang lossy). Lihat DESIGN.md §1-2.

```
config.py    konfigurasi terpusat
prompts.yaml prompts kanonik (promosi redesign_v4.yaml + fix sekunder)  [F2]
transfer.py  kebijakan no-crop + kv_close_turn + chain/concat
agents.py    wrapper load_agent + keep_answer_in_kv
pipeline.py  orkestrator loop evolusi
runner.py    gate + repair + backtest (non-LLM)
run.py       CLI (--generations --latent-steps --dry-run)
results/     output run
```

Reuse: `backend/llm/client.py`, `backend/latent_mas/{agent,kv_ops,parsers}.py`.
Patch bersama yang dibutuhkan (backward-compatible): `keep_answer_in_kv` di
`AgentSpec` + crop kondisional di `client.py` (DESIGN.md §2.1).

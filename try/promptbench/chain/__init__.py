"""promptbench.chain — Phase B: rantai multi-agent via KV-cache.

Modul:
  parsing_hook : EXTENSION POINT parser judger/construct (default delegate ke
                 parsers.parse_hypothesis_exprs; tempat menambal "output bagus
                 tapi gagal terdeteksi" TANPA menyentuh parser produksi).
  collapse     : detektor degradasi KV (repetisi, unparseable, lonjakan token).
  chain        : runner rantai bertahap (proposal→construct→…→feedback) dengan
                 disiplin KV identik pipeline.py + kv_shape_report tiap batas.
"""

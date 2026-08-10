# Keluaran lengan faktor dengan gate mutu MATI (dibuang dari matriks)

Dihasilkan 2026-08-10 sebelum dua bug berikut ditemukan & diperbaiki:

1. `mas/pipeline.py::_build_regulator_gate` mengimpor `dsl.config`
   (FACTOR_COSTEER_SETTINGS) — modul yang ikut terhapus di rombakan 9d4e0bf.
   ImportError-nya ditelan `except Exception` sehingga SELURUH rantai gate
   (arity, variabel, degenerate, semantik, eksekusi) diam-diam nonaktif dan
   gate jatuh ke `default_quality_gate` (sintaks saja).
2. `gate/execution_gate.py` memakai `.parent.parent.parent` warisan lokasi
   lamanya (`backend/factors/regulator/`, 3 tingkat). Setelah pindah ke
   `backend/gate/` (2 tingkat) jalur itu menunjuk ROOT PROYEK, sehingga
   `daily_pv.h5` tak ketemu dan gate eksekusi fail-open SENYAP.

Akibat terukur pada `frontend_kv_raw.json` (11 ekspresi, 6 run):
  - 7 lolos gate, tapi hanya 3 yang benar-benar bisa dievaluasi
  - 4 lolos gate PADAHAL gagal total saat evaluasi:
      2x NameError 'what' (teks placeholder LLM masuk ke ekspresi)
      1x NameError 'IF'            (fungsi tak ada di pustaka DSL)
      1x NameError 'TS_RESIDUAL'   (fungsi tak ada)
      1x TypeError RANK() 1 arg tapi diberi 2   (arity)
  - `validate_semantics` meluluskan 11 dari 11

DISIMPAN sebagai bukti kuantitatif kebocoran gate sintaks-saja, BUKAN sebagai
data eksperimen. Jangan campurkan dengan hasil di `results/factor/` — sel-sel
itu dijalankan ulang dengan gate penuh aktif.

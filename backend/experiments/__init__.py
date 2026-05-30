"""
experiments/ — harness untuk menjalankan & men-debug agent LatentMAS terpisah.

Pengganti ./try lama. Fokus: jalankan satu agent / satu tahap, inspeksi KV,
iterasi prompt — bukan test suite besar yang sulit dibaca.

Skrip:
  run_agent.py   — jalankan satu agent standalone (lihat --help).
  inspect_kv.py  — probe isi KV-cache tersimpan via agent introspect.
"""

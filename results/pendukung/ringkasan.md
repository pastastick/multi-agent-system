# Data pendukung (regenerasi: `python scripts/kumpulkan_pendukung.py`)

## Pemakaian fungsi DSL (265 ekspresi)

| fungsi | dipakai di | % |
|---|---:|---:|
| RANK | 106 | 40.0 |
| TS_PCTCHANGE | 90 | 34.0 |
| TS_ZSCORE | 65 | 24.5 |
| TS_ARGMAX | 45 | 17.0 |
| TS_QUANTILE | 35 | 13.2 |
| REGRESI | 33 | 12.5 |
| TS_MAD | 33 | 12.5 |
| TS_SKEW | 28 | 10.6 |
| TS_ARGMIN | 23 | 8.7 |
| TS_CORR | 21 | 7.9 |
| TS_STD | 17 | 6.4 |
| TS_COVARIANCE | 14 | 5.3 |
| SEQUENCE | 12 | 4.5 |
| DECAYLINEAR | 10 | 3.8 |
| TS_KURT | 9 | 3.4 |
| TS_SUM | 7 | 2.6 |
| TS_MEDIAN | 6 | 2.3 |
| DELAY | 6 | 2.3 |
| TS_MEAN | 5 | 1.9 |
| DELTA | 5 | 1.9 |

## Efektivitas gate (kebocoran = lolos gate tapi gagal dievaluasi)

| sumber | ekspresi | lolos gate | evaluable | BOCOR |
|---|---:|---:|---:|---:|
| arsip_dsl_lama_2026-08-10 | 160 | 19 | 21 | 3 |
| arsip_gate_mati_2026-08-10 | 105 | 7 | 3 | 4 |

## Alasan penolakan gate

| alasan | n |
|---|---:|
| arity | 15 |
| regulator evaluate failed | 14 |
| regulator reject | 7 |
| execution | 3 |
| unparsable expression | 3 |
| ParseException | 3 |
| semantics | 1 |
| variable | 1 |

## Kenapa ekspresi gagal dievaluasi

| error | n |
|---|---:|
| NameError | 14 |
| TypeError | 8 |
| ParseException | 6 |
| timeout>90s | 5 |
| ValueError | 1 |

## Waktu sel lengan faktor (biaya teks vs laten)

| sel | comm | metode | n_run | rerata (dtk) |
|---|---|---|---:|---:|
| frontend_kv_gumbel | kv | gumbel | 6 | 54.0 |
| frontend_kv_moi | kv | moi | 6 | 35.4 |
| frontend_kv_raw | kv | raw | 6 | 65.3 |
| frontend_kv_sample | kv | sample | 6 | 37.4 |
| frontend_kv_soft | kv | soft | 6 | 65.4 |
| frontend_text | text | raw | 6 | 195.5 |

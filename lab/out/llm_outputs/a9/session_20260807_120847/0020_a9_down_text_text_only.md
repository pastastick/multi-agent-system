# Call 0020 — `a9_down_text` (text_only)

## Meta

- ts: 2026-08-07 12:10:15
- conv_id: `0fd79ce6`
- step: 0
- temperature: 0.6
- has_past_kv: False
- input_tokens: 71
- output_tokens: 16
- duration_s: 0.5735
- text_len: 34

## System Prompt

```text
You are a relay agent. You report the payload you received.
```

## User Prompt

```text
An earlier agent was given a payload and reported this:
---
, WMA, TS_MEAN
---
Report the payload. Output ONLY the 5 items, comma-separated, nothing else.
```

## Response

```text
WMA,TS_MEAN,TS_MAX,TS_MIN,TS_COUNT
```

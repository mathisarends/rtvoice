# rtvoice — working scratchpad

Keep this file concise. Write everything (code, comments, docs) as concisely as
possible. When the user corrects a mistake, record it below as a short bullet
(max 2 lines).

## Corrections

- Comments explain the *why*, never the *what/how*. Drop docstrings that just
  restate the signature.
- Built-in tools belong inline in `Tools._register_default_tools`, gated by
  `available_when` — no helper module, no per-tool `_register_*` methods.

# rtvoice — working scratchpad

Keep this file concise. Write everything (code, comments, docs) as concisely as
possible. When the user corrects a mistake, record it below as a short bullet
(max 2 lines).

## Corrections

- Comments explain the *why*, never the *what/how*. Drop docstrings that just
  restate the signature.
- Register injectable dependencies such as `Supervisor` in the tool layer.
- Store constructor dependencies used by handlers, including `event_bus`, on `self`.

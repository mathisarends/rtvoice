# rtvoice — working scratchpad

Keep this file concise. Write everything (code, comments, docs) as concisely as
possible. When the user corrects a mistake, record it below as a short bullet
(max 2 lines).

## Corrections

- Comments explain the _why_, never the _what/how_. Drop docstrings that just
  restate the signature.
- Register injectable dependencies such as `TextAgent` in the tool layer.
- Store constructor dependencies used by handlers, including `event_bus`, on `self`.
- Use semantic aliases for primitive mapping keys/values.
- Prefer private module helpers over static methods that need no class state.
- Keep listener event payloads flat when fields suffice; avoid nested metadata wrappers.

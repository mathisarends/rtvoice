# rtvoice — working scratchpad

Keep this file concise. Write everything (code, comments, docs) as concisely as
possible. When the user corrects a mistake, record it below as a short bullet
(max 2 lines).

## Corrections

- Comments explain the *why*, never the *what/how*. Drop docstrings that just
  restate the signature.
- Register injectable dependencies such as `Subagent` in the tool layer.
- Store constructor dependencies used by handlers, including `event_bus`, on `self`.
- Use semantic aliases for primitive mapping keys/values.
- Prefer private module helpers over static methods that need no class state.
- Keep context assembly in `SystemPrompt`; `Skills` only manages skills.
- Keep public extension APIs minimal and generic; use `system_prompt: str`
  instead of an opinionated prompt/memory model or ambiguous `instructions`.
- Keep listener event payloads flat when fields suffice; avoid nested metadata wrappers.

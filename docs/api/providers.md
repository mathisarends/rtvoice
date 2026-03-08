# Providers

Providers control how `RealtimeAgent` connects to a Realtime API backend —
which WebSocket URL to use and how to authenticate.

The default is `OpenAIProvider`, which requires no explicit configuration when
`OPENAI_API_KEY` is set in the environment.

## OpenAI

::: rtvoice.realtime.providers.OpenAIProvider

## Azure OpenAI

::: rtvoice.realtime.providers.AzureOpenAIProvider

## Custom providers

Implement `RealtimeProvider` to connect to any compatible backend.

::: rtvoice.realtime.providers.RealtimeProvider

# Audio

rtvoice ships ready-to-use implementations for microphone input and speaker output.
For custom hardware or special requirements (e.g. file playback, virtual devices)
you can implement the abstract base classes directly.

## Quickstart
```python
from rtvoice.audio import MicrophoneInput, SpeakerOutput

agent = RealtimeAgent(
    audio_input=MicrophoneInput(sample_rate=24000),
    audio_output=SpeakerOutput(sample_rate=24000),
)
```

## Built-in devices

::: rtvoice.audio.MicrophoneInput

::: rtvoice.audio.SpeakerOutput

## Custom devices

Both built-in classes implement an abstract base interface.
Use these if you need to bring your own audio source or sink.

::: rtvoice.audio.AudioInputDevice

::: rtvoice.audio.AudioOutputDevice

class ElevenLabsVoiceCloner:
    def __init__(self, client):
        self._client = client

    def clone(self, audio_bytes: bytes, name: str) -> str:
        response = self._client.voices.ivc.create(
            name=name,
            files=[("voice_sample.wav", audio_bytes, "audio/wav")],
        )
        return response.voice_id

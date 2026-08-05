def _sniff_audio_filename_and_content_type(audio_bytes: bytes) -> tuple[str, str]:
    if audio_bytes[:4] == b"RIFF":
        return "voice_sample.wav", "audio/wav"
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "voice_sample.mp3", "audio/mpeg"
    return "voice_sample.bin", "application/octet-stream"


class ElevenLabsVoiceCloner:
    def __init__(self, client):
        self._client = client

    def clone(self, audio_bytes: bytes, name: str) -> str:
        filename, content_type = _sniff_audio_filename_and_content_type(audio_bytes)
        response = self._client.voices.ivc.create(
            name=name,
            files=[(filename, audio_bytes, content_type)],
        )
        return response.voice_id

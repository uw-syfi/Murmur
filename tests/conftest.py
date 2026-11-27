import os

import pytest
import soundfile as sf


@pytest.fixture(scope="session")
def meeting_excerpt(tmp_path_factory):
    source = os.environ.get("MURMUR_AMI_ES2004A")
    if not source:
        pytest.skip("set MURMUR_AMI_ES2004A to run hardware inference tests")
    path = tmp_path_factory.mktemp("ami") / "agenda.wav"
    # The introductions and project brief span multiple 30-second encoder windows.
    with sf.SoundFile(source) as audio:
        audio.seek(80 * audio.samplerate)
        frames = audio.read(round(62.83 * audio.samplerate), dtype="float32")
        sf.write(path, frames, audio.samplerate)
    return path

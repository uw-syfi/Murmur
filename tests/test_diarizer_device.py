import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from murmur.audio import Diarizer


@pytest.mark.parametrize("cuda, mps, override, expected", [
    (True, True, None, "cuda"),
    (False, True, None, "mps"),
    (False, False, None, "cpu"),
    (False, True, "cpu", "cpu"),
])
def test_diarizer_device_selection(monkeypatch, cuda, mps, override, expected):
    pipeline = Mock()
    monkeypatch.setitem(sys.modules, "pyannote.audio", SimpleNamespace(Pipeline=pipeline))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    diarizer = Diarizer(token="test", device=override)

    assert diarizer.device == torch.device(expected)
    pipeline.from_pretrained.return_value.to.assert_called_once_with(torch.device(expected))

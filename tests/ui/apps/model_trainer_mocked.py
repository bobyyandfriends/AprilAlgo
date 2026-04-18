"""Streamlit runner: model trainer with subprocess.run mocked."""

from unittest.mock import MagicMock

import aprilalgo.ui.pages.model_trainer as mtr
from aprilalgo.ui.pages.model_trainer import render


def _fake_run(*args, **kwargs):
    return MagicMock(
        returncode=0,
        stdout="trained\n",
        stderr="",
    )


mtr.subprocess.run = _fake_run  # type: ignore[method-assign]

render()

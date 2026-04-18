"""Streamlit runner: ML lab with subprocess.run mocked."""

from unittest.mock import MagicMock

import aprilalgo.ui.pages.model_lab as mlab
from aprilalgo.ui.pages.model_lab import render


def _fake_run(*args, **kwargs):
    return MagicMock(
        returncode=0,
        stdout="cli_ok\n",
        stderr="",
    )


mlab.subprocess.run = _fake_run  # type: ignore[method-assign]

render()

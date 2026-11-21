import runpy
from pathlib import Path

import pytest

import reliability_lab

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_version_is_exposed() -> None:
    assert isinstance(reliability_lab.__version__, str)
    assert reliability_lab.__version__


@pytest.mark.parametrize("script", ["quickstart.py", "simulated.py", "llm_judge.py", "splits.py"])
def test_examples_run_end_to_end(script: str, capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLES_DIR / script), run_name="__main__")
    captured = capsys.readouterr()
    assert captured.out.strip()

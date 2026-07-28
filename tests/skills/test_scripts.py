from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from rtvoice.skills.scripts import _interpreter, _which, run_script


def test_interpreter_runs_unknown_suffix_directly() -> None:
    assert _interpreter(Path("tool.exe")) == []


def test_which_resolves_executable_on_path() -> None:
    with patch("rtvoice.skills.scripts.shutil.which", return_value="/usr/bin/bash"):
        assert _which("bash") == ["/usr/bin/bash"]


def test_which_raises_when_executable_missing() -> None:
    with (
        patch("rtvoice.skills.scripts.shutil.which", return_value=None),
        pytest.raises(ValueError, match="not installed or not on PATH"),
    ):
        _which("bash")


@pytest.mark.asyncio
async def test_run_script_reports_missing_interpreter(tmp_path: Path) -> None:
    script = tmp_path / "check.sh"
    script.write_text("echo hi\n", encoding="utf-8")

    with patch("rtvoice.skills.scripts.shutil.which", return_value=None):
        result = await run_script(script, cwd=tmp_path)

    assert result == "Error: 'bash' is not installed or not on PATH."


@pytest.mark.asyncio
async def test_run_script_reports_timeout(tmp_path: Path) -> None:
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")

    result = await run_script(script, cwd=tmp_path, timeout=1)

    assert result == "Error: Script timed out after 1 seconds."


@pytest.mark.asyncio
async def test_run_script_reports_os_error(tmp_path: Path) -> None:
    script = tmp_path / "check.py"
    script.write_text("print('hi')\n", encoding="utf-8")

    with patch("rtvoice.skills.scripts.subprocess.run", side_effect=OSError("boom")):
        result = await run_script(script, cwd=tmp_path)

    assert result == "Error: boom"

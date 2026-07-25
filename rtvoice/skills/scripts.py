from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

_INTERPRETERS = {
    ".py": lambda: [sys.executable],
    ".sh": lambda: _which("bash"),
    ".bash": lambda: _which("bash"),
}


async def run_script(
    script: Path,
    args: Sequence[str] = (),
    *,
    cwd: Path,
    timeout: int = 60,
) -> str:
    try:
        argv = [*_interpreter(script), str(script), *args]
    except ValueError as exc:
        return f"Error: {exc}"

    logger.debug("Running skill script %s (timeout=%ss)", script, timeout)
    return await asyncio.to_thread(_run, argv, cwd, timeout)


def _interpreter(script: Path) -> list[str]:
    resolve = _INTERPRETERS.get(script.suffix.lower())
    if resolve is None:
        # no shebang handling on Windows, so unknown suffixes must be directly
        # executable
        return []
    return resolve()


def _which(executable: str) -> list[str]:
    path = shutil.which(executable)
    if path is None:
        raise ValueError(f"'{executable}' is not installed or not on PATH.")
    return [path]


def _run(argv: list[str], cwd: Path, timeout: int) -> str:
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"Error: Script timed out after {timeout} seconds."
    except OSError as exc:
        return f"Error: {exc}"

    if result.returncode == 0:
        return result.stdout.strip() or "Success"
    return f"Error (exit code {result.returncode}): {result.stderr.strip()}"

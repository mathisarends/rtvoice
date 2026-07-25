from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from tree_sitter import Node, Parser

    from rtvoice.skills.manager import SkillManager

logger = logging.getLogger(__name__)

_ALLOWED_NAMED_NODES = frozenset(
    {
        "program",
        "command",
        "command_name",
        "declaration_command",
        "pipeline",
        "list",
        "redirected_statement",
        "file_redirect",
        "file_descriptor",
        "variable_assignment",
        "variable_name",
        "special_variable_name",
        "word",
        "string",
        "string_content",
        "raw_string",
        "ansi_c_string",
        "translated_string",
        "concatenation",
        "number",
        "simple_expansion",
        "expansion",
        "arithmetic_expansion",
        "binary_expression",
        "unary_expression",
        "parenthesized_expression",
        "array",
    }
)


class BashArgs(BaseModel):
    command: str = Field(description="The Bash command to execute.")
    timeout: int = Field(
        default=60,
        ge=1,
        le=300,
        description="Timeout in seconds. Defaults to 60; maximum 300.",
    )
    cwd: str | None = Field(
        default=None,
        description=(
            "Working directory for the command. Use the skill's base "
            "directory instead of a `cd` command."
        ),
    )


class BashRunner:
    def __init__(
        self,
        *,
        allowed_commands: tuple[str, ...],
        allowed_script_dirs: tuple[Path, ...],
    ) -> None:
        self._allowed_commands = frozenset(allowed_commands)
        self._allowed_script_dirs = tuple(
            path.resolve() for path in allowed_script_dirs
        )
        for command in self._allowed_commands:
            if not command or any(char.isspace() for char in command):
                raise ValueError(
                    f"Allowed command {command!r} must be one executable token."
                )

    @classmethod
    def for_skills(
        cls,
        skill_manager: SkillManager | None,
        allowed_commands: Iterable[str],
    ) -> BashRunner | None:
        if skill_manager is None or skill_manager.size == 0:
            return None
        return cls(
            allowed_commands=tuple(dict.fromkeys(allowed_commands)),
            allowed_script_dirs=skill_manager.directories,
        )

    async def execute(self, args: BashArgs) -> str:
        error = validate_command(
            args.command,
            allowed_commands=self._allowed_commands,
            allowed_script_dirs=self._allowed_script_dirs,
            cwd=args.cwd,
        )
        if error is not None:
            return f"Command rejected: {error}"

        bash = shutil.which("bash")
        if bash is None:
            return "Error: Bash is not installed or is not on PATH."

        logger.debug(
            "Executing skill bash command: %s (timeout=%ss, cwd=%s)",
            args.command,
            args.timeout,
            args.cwd,
        )
        return await asyncio.to_thread(
            self._run,
            bash,
            args.command,
            args.timeout,
            args.cwd,
        )

    def _run(self, bash: str, command: str, timeout: int, cwd: str | None) -> str:
        try:
            result = subprocess.run(
                [bash, "-lc", command],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds."
        except OSError as exc:
            return f"Error: {exc}"

        if result.returncode == 0:
            return result.stdout.strip() or "Success"
        return f"Error (exit code {result.returncode}): {result.stderr.strip()}"


def validate_command(
    command: str,
    *,
    allowed_commands: frozenset[str],
    allowed_script_dirs: tuple[Path, ...],
    cwd: str | None = None,
) -> str | None:
    if not command.strip():
        return "Empty command."

    resolved_cwd: Path | None = None
    if cwd is not None:
        try:
            resolved_cwd = Path(cwd).resolve(strict=True)
        except OSError:
            return f"cwd '{cwd}' does not exist."
        if not resolved_cwd.is_dir() or not _under_any(
            resolved_cwd, allowed_script_dirs
        ):
            return (
                f"cwd '{cwd}' is not under an allowed skill directory. "
                f"Allowed directories: {[str(path) for path in allowed_script_dirs]}."
            )

    try:
        tree = _get_parser().parse(command.encode("utf-8"))
    except Exception as exc:
        return f"Failed to parse command: {exc}"

    root = tree.root_node
    if root.has_error:
        return "Command has syntax errors."
    if not root.children:
        return "Empty command."

    return _walk(
        root,
        allowed_commands=allowed_commands,
        allowed_script_dirs=allowed_script_dirs,
        cwd=resolved_cwd,
    )


@lru_cache(maxsize=1)
def _get_parser() -> Parser:
    import tree_sitter_bash
    from tree_sitter import Language, Parser

    return Parser(Language(tree_sitter_bash.language()))


def _walk(
    node: Node,
    *,
    allowed_commands: frozenset[str],
    allowed_script_dirs: tuple[Path, ...],
    cwd: Path | None,
) -> str | None:
    if node.is_named and node.type not in _ALLOWED_NAMED_NODES:
        snippet = node.text.decode("utf-8", errors="replace")[:80]
        return f"Disallowed shell construct '{node.type}' in {snippet!r}."

    if node.type == "command":
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            executable = name_node.text.decode("utf-8", errors="replace")
            if executable not in allowed_commands and not _allowed_script(
                executable, allowed_script_dirs, cwd
            ):
                return (
                    f"Command '{executable}' is not allowed. Allowed commands: "
                    f"{sorted(allowed_commands)}."
                )

    for child in node.children:
        error = _walk(
            child,
            allowed_commands=allowed_commands,
            allowed_script_dirs=allowed_script_dirs,
            cwd=cwd,
        )
        if error is not None:
            return error
    return None


def _allowed_script(
    executable: str,
    allowed_script_dirs: tuple[Path, ...],
    cwd: Path | None,
) -> bool:
    candidate = Path(executable)
    if not candidate.is_absolute():
        if cwd is None:
            return False
        candidate = cwd / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return False
    return resolved.is_file() and _under_any(resolved, allowed_script_dirs)


def _under_any(path: Path, directories: tuple[Path, ...]) -> bool:
    for directory in directories:
        try:
            path.relative_to(directory)
        except ValueError:
            continue
        return True
    return False

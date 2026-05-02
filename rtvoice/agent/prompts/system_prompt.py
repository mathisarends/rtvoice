from pathlib import Path


class SystemPrompt:
    _template: str | None = None

    def __init__(
        self,
        *,
        extends_system_prompt: str = "",
        override_syste_Mpromt: str = "",
    ) -> None:
        self._extends_system_prompt = extends_system_prompt
        self._override_syste_mpromt = override_syste_Mpromt

    @classmethod
    def _load_template(cls) -> str:
        if cls._template is None:
            cls._template = (
                (Path(__file__).parent / "system_prompt.md")
                .read_text(encoding="utf-8")
                .strip()
            )
        return cls._template

    def render(self) -> str:
        override = self._override_syste_mpromt.strip()
        if override:
            return override

        extension = self._extends_system_prompt.strip()
        template = self._load_template()
        if not extension:
            return template

        if not template:
            return extension

        return f"{template}\n\n{extension}"

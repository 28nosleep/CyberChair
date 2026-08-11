import os
from pathlib import Path


def _parse_value(raw_value):
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_environment(base_dir=None):
    """Load local deployment settings without overriding server environment."""
    base_dir = Path(base_dir or Path(__file__).resolve().parent)
    loaded = []

    # A private .env has priority. .env.example remains supported as a fallback
    # so an uploaded project works with the user's current configuration layout.
    for path in (base_dir / ".env", base_dir / ".env.example"):
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, separator, raw_value = line.partition("=")
            key = key.strip()
            if not separator or not key or not key.replace("_", "a").isalnum():
                continue
            os.environ.setdefault(key, _parse_value(raw_value))
        loaded.append(path)
    return loaded

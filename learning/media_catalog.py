import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MediaAsset:
    id: str
    type: str
    file: str | None
    contexts: tuple[str, ...]
    tags: tuple[str, ...]
    intensity_min: float
    weight: float
    cooldown_group: str
    archetype: str
    render_profile: str = "overlay"
    text_box: tuple[float, float, float, float] = (0.06, 0.58, 0.94, 0.94)
    source_url: str | None = None


class MediaCatalog:
    def __init__(self, path=None):
        self.path = Path(path or Path(__file__).resolve().parent.parent / "media" / "catalog.json")
        self.root = self.path.parent
        self.version = "0"
        self.assets = ()
        self._by_id = {}
        self.reload()

    def reload(self):
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            payload = {"version": "0", "assets": []}
        assets = []
        for raw in payload.get("assets", []):
            try:
                asset = MediaAsset(
                    id=str(raw["id"]),
                    type=str(raw["type"]),
                    file=str(raw["file"]) if raw.get("file") else None,
                    contexts=tuple(raw.get("contexts", ())),
                    tags=tuple(raw.get("tags", ())),
                    intensity_min=float(raw.get("intensity_min", 0.0)),
                    weight=float(raw.get("weight", 1.0)),
                    cooldown_group=str(raw.get("cooldown_group") or raw["id"]),
                    archetype=str(raw.get("archetype") or "reaction"),
                    render_profile=str(raw.get("render_profile") or "overlay"),
                    text_box=tuple(float(value) for value in raw.get("text_box", (.06, .58, .94, .94))),
                    source_url=str(raw["source_url"]) if raw.get("source_url") else None,
                )
            except (KeyError, TypeError, ValueError):
                continue
            if len(asset.text_box) == 4:
                assets.append(asset)
        self.version = str(payload.get("version", "0"))
        self.assets = tuple(assets)
        self._by_id = {asset.id: asset for asset in assets}

    def get(self, asset_id):
        return self._by_id.get(asset_id)

    def resolve(self, asset):
        if not asset or not asset.file:
            return None
        candidate = (self.root / asset.file).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError:
            return None
        return candidate

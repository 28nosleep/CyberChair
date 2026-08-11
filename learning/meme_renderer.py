import os
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class RenderResult:
    path: Path
    rendered_text: str
    text_box: tuple[int, int, int, int]
    safe_box: tuple[int, int, int, int]
    font_size: int
    line_count: int
    render_profile: str = "overlay"
    font_name: str = ""


class MemeRenderer:
    def __init__(self, catalog, output_dir, font_paths=None):
        self.catalog = catalog
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.font_paths = tuple(font_paths or (
            ("Impact", "/System/Library/Fonts/Supplemental/Impact.ttf"),
            ("Impact", "/Library/Fonts/Impact.ttf"),
            ("Anton", "/System/Library/Fonts/Supplemental/Anton-Regular.ttf"),
            ("Anton", "/Library/Fonts/Anton-Regular.ttf"),
            ("Arial Black", "/System/Library/Fonts/Supplemental/Arial Black.ttf"),
            ("Arial Black", "/Library/Fonts/Arial Black.ttf"),
            ("DejaVu Sans Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ))
        self.cleanup_stale()

    @staticmethod
    def normalize_text(text):
        clean = "".join(
            char for char in str(text or "")
            if unicodedata.category(char) not in {"Cs", "So"}
        )
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    @staticmethod
    def _supports_text(font, text):
        replacement = bytes(font.getmask("\ufffd"))
        for char in text:
            if char.isspace() or char.isascii():
                continue
            if bytes(font.getmask(char)) == replacement:
                return False
        return True

    def _font(self, size, text):
        for name, path in self.font_paths:
            try:
                font = ImageFont.truetype(path, size=size)
                if self._supports_text(font, text):
                    return font, name
            except (OSError, ValueError):
                continue
        return None, ""

    @staticmethod
    def _wrap(draw, text, font, width, max_lines, stroke_width):
        words = text.split()
        lines = []
        current = ""
        for word in words:
            trial = f"{current} {word}".strip()
            if draw.textbbox((0, 0), trial, font=font, stroke_width=stroke_width)[2] <= width:
                current = trial
                continue
            if not current or len(lines) + 1 >= max_lines:
                return None
            lines.append(current)
            current = word
        if current:
            lines.append(current)
        return lines if len(lines) <= max_lines else None

    @staticmethod
    def _pixel_box(box, width, height):
        return (
            int(box[0] * width), int(box[1] * height),
            int(box[2] * width), int(box[3] * height),
        )

    def _draw_caption(self, image, text, safe_box):
        left, top, right, bottom = safe_box
        draw = ImageDraw.Draw(image)
        chosen = None
        for size in range(min(72, max(28, image.width // 18)), 17, -2):
            font, font_name = self._font(size, text)
            if font is None:
                return None
            stroke = max(3, size // 12)
            lines = self._wrap(draw, text, font, right - left, 4, stroke)
            if not lines:
                continue
            spacing = max(4, size // 6)
            boxes = [
                draw.textbbox((0, 0), line, font=font, stroke_width=stroke)
                for line in lines
            ]
            total_height = sum(box[3] - box[1] for box in boxes) + spacing * (len(lines) - 1)
            if total_height <= bottom - top:
                chosen = font, font_name, size, stroke, lines, spacing, boxes, total_height
                break
        if not chosen:
            return None
        font, font_name, size, stroke, lines, spacing, boxes, total_height = chosen
        y = top + max(0, (bottom - top - total_height) // 2)
        bounds = [right, bottom, left, top]
        for line, box in zip(lines, boxes):
            line_width, line_height = box[2] - box[0], box[3] - box[1]
            x = left + max(0, (right - left - line_width) // 2)
            draw.text(
                (x, y), line, font=font, fill="white",
                stroke_width=stroke, stroke_fill="black",
            )
            bounds[0] = min(bounds[0], x)
            bounds[1] = min(bounds[1], y)
            bounds[2] = max(bounds[2], x + line_width)
            bounds[3] = max(bounds[3], y + line_height)
            y += line_height + spacing
        return tuple(bounds), size, len(lines), font_name

    def render(self, template_id, text):
        asset = self.catalog.get(template_id)
        template = self.catalog.resolve(asset)
        raw_text = str(text or "")
        if asset and asset.render_profile == "top_bottom":
            raw_parts = re.split(r"\s*(?:\n\|\n|\|)\s*", raw_text, maxsplit=1)
        else:
            raw_parts = [raw_text]
        captions = [self.normalize_text(part) for part in raw_parts if self.normalize_text(part)]
        clean = " | ".join(captions)
        if not asset or asset.type != "meme_template" or not clean or not template or not template.exists():
            return None
        try:
            with Image.open(template) as source:
                image = source.convert("RGBA").copy()
            if max(image.size) > 1600:
                image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            width, height = image.size
            profile = asset.render_profile
            if profile == "top_bottom":
                safe_boxes = (
                    self._pixel_box((.05, .04, .95, .24), width, height),
                    self._pixel_box((.05, .76, .95, .96), width, height),
                )
            else:
                safe_boxes = (self._pixel_box(asset.text_box, width, height),)
            rendered = []
            for index, caption in enumerate(captions):
                if index >= len(safe_boxes):
                    return None
                result = self._draw_caption(image, caption, safe_boxes[index])
                if not result:
                    return None
                rendered.append(result)
            if not rendered:
                return None
            bounds = (
                min(item[0][0] for item in rendered),
                min(item[0][1] for item in rendered),
                max(item[0][2] for item in rendered),
                max(item[0][3] for item in rendered),
            )
            fd, raw_path = tempfile.mkstemp(
                prefix="cyberchair_", suffix=".png", dir=self.output_dir
            )
            os.close(fd)
            path = Path(raw_path)
            image.save(path, "PNG", optimize=True)
            return RenderResult(
                path, clean, bounds, safe_boxes[0],
                min(item[1] for item in rendered),
                sum(item[2] for item in rendered), profile,
                rendered[0][3],
            )
        except (OSError, ValueError):
            return None

    @staticmethod
    def cleanup(result_or_path):
        path = getattr(result_or_path, "path", result_or_path)
        try:
            Path(path).unlink(missing_ok=True)
        except (OSError, TypeError, ValueError):
            pass

    def cleanup_stale(self, max_age_seconds=3600):
        cutoff = time.time() - max_age_seconds
        for path in self.output_dir.glob("cyberchair_*.png"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                continue

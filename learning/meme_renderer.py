import os
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


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
    SAFE_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF"}

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
        self._font_cache = {}
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
        # Font support depends on the character set, not its frequency. Long
        # repeated captions previously rasterized the same Cyrillic glyph
        # thousands of times during every font-size probe.
        for char in dict.fromkeys(text):
            if char.isspace() or char.isascii():
                continue
            if bytes(font.getmask(char)) == replacement:
                return False
        return True

    def _font(self, size, text):
        for name, path in self.font_paths:
            try:
                key = (path, size)
                font = self._font_cache.get(key)
                if font is None:
                    font = ImageFont.truetype(path, size=size)
                    self._font_cache[key] = font
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
            word_box = draw.textbbox(
                (0, 0), word, font=font, stroke_width=stroke_width
            )
            if word_box[2] - word_box[0] > width:
                return None
            trial = f"{current} {word}".strip()
            trial_box = draw.textbbox(
                (0, 0), trial, font=font, stroke_width=stroke_width
            )
            if trial_box[2] - trial_box[0] <= width:
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

    def _draw_caption(self, image, text, safe_box, fill="white", stroke_fill="black"):
        left, top, right, bottom = safe_box
        draw = ImageDraw.Draw(image)
        chosen = None
        candidate_text = text.upper()
        maximum = min(
            190,
            max(42, image.width // 4),
            max(42, int((bottom - top) * .80)),
            max(42, (right - left) // 2),
        )
        for size in range(maximum, 17, -3):
            font, font_name = self._font(size, text)
            if font is None:
                return None
            stroke = max(4, size // 11)
            lines = self._wrap(draw, candidate_text, font, right - left, 4, stroke)
            if not lines:
                continue
            spacing = max(2, size // 14)
            boxes = [
                draw.textbbox((0, 0), line, font=font, stroke_width=stroke)
                for line in lines
            ]
            total_height = sum(box[3] - box[1] for box in boxes) + spacing * (len(lines) - 1)
            if total_height <= bottom - top:
                chosen = font, font_name, size, stroke, lines, spacing, boxes, total_height
                break
        # A renderer must never cover half a photo merely to preserve every
        # word. Keep the strongest leading phrase and mark the shortening.
        while not chosen and len(candidate_text.split()) > 4:
            candidate_text = " ".join(candidate_text.split()[:-1]).rstrip(".,;:") + "…"
            for size in range(min(150, maximum), 17, -3):
                font, font_name = self._font(size, candidate_text)
                if font is None:
                    return None
                stroke = max(4, size // 11)
                lines = self._wrap(draw, candidate_text, font, right - left, 4, stroke)
                if not lines:
                    continue
                spacing = max(2, size // 14)
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
            x = left + max(0, (right - left - line_width) // 2) - box[0]
            draw_y = y - box[1]
            draw.text(
                (x, draw_y), line, font=font, fill=fill,
                stroke_width=stroke, stroke_fill=stroke_fill,
            )
            bounds[0] = min(bounds[0], x + box[0])
            bounds[1] = min(bounds[1], draw_y + box[1])
            bounds[2] = max(bounds[2], x + box[2])
            bounds[3] = max(bounds[3], draw_y + box[3])
            y += line_height + spacing
        return tuple(bounds), size, len(lines), font_name

    @staticmethod
    def _profile_boxes(profile, width, height, default_box=None):
        if profile == "top_bottom":
            return (
                MemeRenderer._pixel_box((.05, .035, .95, .255), width, height),
                MemeRenderer._pixel_box((.05, .745, .95, .965), width, height),
            )
        if profile == "bottom_caption":
            return (MemeRenderer._pixel_box((.035, .64, .965, .97), width, height),)
        if profile == "top_caption":
            return (MemeRenderer._pixel_box((.035, .025, .965, .36), width, height),)
        if profile in {"center", "top_center", "center_bottom"}:
            boxes = {
                "center": (.035, .31, .965, .69),
                "top_center": (.035, .14, .965, .53),
                "center_bottom": (.035, .47, .965, .86),
            }
            return (MemeRenderer._pixel_box(boxes[profile], width, height),)
        return (MemeRenderer._pixel_box(default_box, width, height),)

    @staticmethod
    def _split_top_bottom(text):
        clean = MemeRenderer.normalize_text(text)
        explicit = re.split(r"\s*(?:\n\|\n|\||\n)\s*", str(text or ""), maxsplit=1)
        if len(explicit) == 2 and all(MemeRenderer.normalize_text(part) for part in explicit):
            return [MemeRenderer.normalize_text(part) for part in explicit]
        words = clean.split()
        if len(words) < 4:
            return [clean]
        # Prefer a natural clause boundary near the middle.
        middle = len(words) // 2
        boundaries = [
            index for index, word in enumerate(words[1:-1], 1)
            if word.casefold().strip(",:;—-") in {"а", "но", "и", "когда", "зато"}
        ]
        split = min(boundaries, key=lambda value: abs(value - middle)) if boundaries else middle
        return [" ".join(words[:split]), " ".join(words[split:])]

    def _render_image(self, image, captions, profile, safe_boxes,
                      caption_fill="white", caption_stroke="black"):
        rendered = []
        for index, caption in enumerate(captions):
            if index >= len(safe_boxes):
                return None
            result = self._draw_caption(
                image, caption, safe_boxes[index], caption_fill, caption_stroke
            )
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
        try:
            image.save(path, "PNG", optimize=True)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return RenderResult(
            path, " | ".join(captions), bounds, safe_boxes[0],
            min(item[1] for item in rendered), sum(item[2] for item in rendered),
            profile, rendered[0][3],
        )

    def render_image(self, source_path, text, render_profile="top_caption",
                     max_bytes=20 * 1024 * 1024, max_dimension=12000,
                     max_pixels=48_000_000):
        """Render over a validated arbitrary image while preserving its aspect ratio."""
        source_path = Path(source_path)
        clean = self.normalize_text(text)
        if not clean or render_profile not in {
            "top_caption", "bottom_caption", "top_bottom", "center",
            "top_center", "center_bottom",
        }:
            return None
        try:
            if not source_path.is_file() or source_path.stat().st_size > max_bytes:
                return None
            with Image.open(source_path) as source:
                if (source.format or "").upper() not in self.SAFE_FORMATS:
                    return None
                width, height = source.size
                if (
                    width <= 0 or height <= 0
                    or max(width, height) > max_dimension
                    or width * height > max_pixels
                ):
                    return None
                image = ImageOps.exif_transpose(source).convert("RGBA").copy()
            if max(image.size) > 1600:
                image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            captions = [clean]
            if render_profile == "top_bottom":
                captions = self._split_top_bottom(text)
                if len(captions) == 1:
                    render_profile = "top_caption"
            boxes = self._profile_boxes(render_profile, *image.size)
            return self._render_image(image, captions, render_profile, boxes)
        except (OSError, ValueError, Image.DecompressionBombError, UnidentifiedImageError):
            return None

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
            safe_boxes = self._profile_boxes(profile, width, height, asset.text_box)
            caption_fill, caption_stroke = (
                ("black", "white") if profile == "phone_screen_light"
                else ("white", "black")
            )
            return self._render_image(
                image, captions, profile, safe_boxes, caption_fill, caption_stroke
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

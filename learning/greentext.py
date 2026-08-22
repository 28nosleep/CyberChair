"""Bounded, source-grounded greentext episode selection and validation."""

from dataclasses import dataclass
import hashlib
import re

from .llm_provider import GenerateRequest
from .preprocessing import normalize_spaces, significant_words


COMMAND_RE = re.compile(r"^\s*стул\s+(?:грин\s*текст|гринтекст|greentext|green\s+text)\s*$", re.I)
GENERIC = {"быть", "зайти", "выйти", "сказать", "написать", "решить", "начать", "продолжить", "потом", "снова", "через", "минуту", "час", "конфу", "чат", "всё", "это", "что", "как", "не", "да", "ну"}


@dataclass(frozen=True)
class GreentextEpisode:
    rows: tuple[dict, ...]
    moment_type: str | None
    signature: str
    evidence_ids: tuple[int, ...] = ()

    @property
    def source_message_ids(self):
        return tuple(int(row["message_id"]) for row in self.rows if row.get("message_id") is not None)


def is_greentext_command(text):
    return bool(COMMAND_RE.fullmatch(normalize_spaces(text or "")))


def _signature(rows):
    return hashlib.sha256(",".join(str(row.get("message_id")) for row in rows).encode()).hexdigest()[:16]


def select_episode(rows, detector, recent_generated=(), evidence_rows=()):
    """Pick one recent detector episode, penalising persisted prior signatures."""
    usable = [dict(row) for row in rows if row.get("text") and not is_greentext_command(row.get("text"))]
    if len(usable) < 2:
        return None
    used = {str(row.get("kind", "")).split(":", 1)[1] for row in recent_generated
            if str(row.get("kind", "")).startswith("greentext:")}
    candidates = []
    # Each row gets a chance to be the detector's current turn, so an older
    # contradiction is not hidden merely because the command is now latest.
    for end in range(1, len(usable) + 1):
        window = usable[max(0, end - 30):end]
        current_id = window[-1].get("message_id")
        for moment in detector.detect(window, current_id)[:3]:
            ids = set(moment.supporting_message_ids)
            source = [row for row in window if row.get("message_id") in ids]
            if not source:
                continue
            sig = _signature(source)
            penalty = .50 if sig in used else 0.0
            candidates.append((moment.score - penalty, len(source), moment, source, sig))
    if candidates:
        _, _, moment, source, sig = max(candidates, key=lambda item: (item[0], item[1]))
        evidence_ids = tuple(int(item["id"]) for item in evidence_rows
                             if item.get("source_message_id") in {row.get("message_id") for row in source})
        return GreentextEpisode(tuple(source[-6:]), moment.moment_type, sig, evidence_ids)
    # A compact real exchange is still a valid episode, but never invent one.
    source = usable[-4:]
    if len(source) < 2:
        return None
    evidence_ids = tuple(int(item["id"]) for item in evidence_rows
                         if item.get("source_message_id") in {row.get("message_id") for row in source})
    return GreentextEpisode(tuple(source), None, _signature(source), evidence_ids)


def build_request(chat_id, episode, safety_identifier):
    lines = []
    for row in episode.rows:
        name = normalize_spaces(str(row.get("username") or f"user_{row.get('user_id')}"))
        text = normalize_spaces(row.get("text"))[:220]
        lines.append(f"id={row.get('message_id')} participant={name}: {text}")
    source = "\n".join(lines)
    instructions = (
        "Ты CyberChair. Сделай только классический русскоязычный 4chan-style greentext "
        "по закрытому набору реальных сообщений. Факты важнее шутки."
    )
    prompt = (
        "Выбран один эпизод; нельзя пересказывать весь чат. Верни ровно 5–10 коротких строк, "
        "каждая начинается с `>`, без заголовка, пояснений, markdown-обрамления или текста после. "
        "lowercase по умолчанию, сухая последовательность, развязка в конце. Можно сокращать и "
        "слегка гиперболизировать только уже произошедшее. Нельзя добавлять людей, действия, слова, "
        "внешние события или новую premise. Используй только участников и факты ниже; для ключевых "
        "слов предпочитай лексику источника. machine flavour не нужен.\n\n"
        f"moment_type={episode.moment_type or 'recent_exchange'}\n"
        f"source_message_ids={','.join(map(str, episode.source_message_ids))}\n"
        f"linked_evidence_ids={','.join(map(str, episode.evidence_ids)) or 'none'}\n"
        f"SOURCE:\n{source}"
    )
    return GenerateRequest(instructions, prompt, 180, safety_identifier, {
        "chat_id": chat_id, "purpose": "greentext", "response_purpose": "greentext",
        "call_type": "greentext", "episode_signature": episode.signature,
    })


def validate_output(output, episode):
    lines = [normalize_spaces(line) for line in str(output or "").splitlines() if normalize_spaces(line)]
    if not 5 <= len(lines) <= 10 or any(not line.startswith(">") for line in lines):
        return False
    source_text = " ".join(row.get("text", "") for row in episode.rows)
    allowed = set(significant_words(source_text))
    names = {normalize_spaces(str(row.get("username") or "")).casefold() for row in episode.rows if row.get("username")}
    for line in lines:
        words = set(significant_words(line[1:]))
        # A line must be tied to a source detail, unless it is only connective
        # greentext grammar. This rejects fabricated events conservatively.
        meaningful = words - GENERIC
        if meaningful and not (meaningful & allowed):
            return False
        for token in re.findall(r"[\wё-]{3,}", line.casefold()):
            if token.startswith("user_") or token in names:
                continue
    return bool(allowed)


def fallback(episode):
    """Five grounded lines assembled from actual source text, without inference."""
    snippets = [normalize_spaces(row.get("text"))[:115] for row in episode.rows if normalize_spaces(row.get("text"))]
    if not snippets:
        return ">сейчас гринтекст лепить не из чего"
    out = []
    for item in snippets:
        out.append(">" + item.casefold())
    while len(out) < 5:
        out.append(out[-1])
    return "\n".join(out[:6])

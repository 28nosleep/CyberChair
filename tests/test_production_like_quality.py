"""Deterministic 60-event production-like quality smoke (no paid API calls)."""

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from learning import LearningService, LearningSettings


LONG_TOPICS = (
    "рецепт харчо", "рецепт хачапури", "настройка Docker",
    "объяснение DNS", "план набора веса", "выбор VPS",
    "восстановление PostgreSQL", "настройка nginx",
    "резервное копирование базы", "домашняя сеть и роутер",
)


class SmokeProvider:
    available = True

    def __init__(self):
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        index = len(self.calls)
        purpose = request.metadata["response_purpose"]
        if purpose in {"recipe_instruction", "complex_explanation"} or any(
            topic.casefold() in request.input.casefold() for topic in LONG_TOPICS
        ):
            parts = (
                f"сценарий {index}: сначала проверь исходные данные и подготовь резервную копию.",
                "затем выполни базовый шаг и сразу проверь результат, чтобы не тащить ошибку дальше.",
                "после этого переходи к следующему этапу, сохраняя понятные контрольные точки.",
                "если команда или ингредиент ведёт себя иначе, остановись и прочитай конкретный лог либо проверь консистенцию.",
                "не меняй пять параметров одновременно: один шаг, одна проверка, один понятный откат.",
                "в конце повтори проверку с чистого состояния и зафиксируй рабочую последовательность.",
                "так ответ остаётся практичным, а не превращается в шаманство с обрезанным финалом.",
            )
            return " ".join(parts)
        if purpose == "troll_user":
            variants = (
                "ты назначил себя легендой до загрузки туториала, охуенный спидран самоуверенности",
                "амбиций на стадион, а план пока помещается на салфетке из шаурмы",
                "реальность прочитала твою заявку на величие и молча включила авиарежим",
                "ты так уверенно штурмуешь вершину, будто лестницу обязан принести интернет",
                "карьерная арка уже эпическая, жалко первый квест всё ещё не принят",
                "это не стратегия, это фанфик где дедлайн сам боится автора",
                "бро пришёл за короной с чеком из пункта выдачи амбиций",
                "твой план звучит как финальный босс, собранный из трёх мотивационных рилсов",
                "ещё один такой вопрос, и успех сам подаст на защитный ордер",
                "ты уже празднуешь камбэк, хотя выйти на сцену пока забыл",
            )
            return f"эпизод {index}: {variants[(index - 1) % len(variants)]}"
        if purpose == "meme_caption":
            variants = (
                "релиз уверенно пошёл не туда", "пять минут растянулись до пятницы",
                "прод снова выбрал насилие", "серёга просто посмотрел логи",
                "фикс маленький последствия исторические", "всё под контролем чужим",
                "дедлайн вышел из чата", "тесты поверили на слово",
                "роллбек обрёл человеческое лицо", "архитектура попросила эвтаназию",
            )
            return f"{variants[(index - 1) % len(variants)]} {index}"
        if purpose == "recommendation":
            return f"вариант {index} бери только после проверки цены, ограничений и реального сценария; хайп сам счёт не оплатит"
        if purpose == "short_social":
            return f"принял эпизод {index}, живи пока"
        return f"ответ {index}: проверь условия, начни с безопасного шага и оцени результат до следующего изменения"

    def summarize(self, request):
        return None


def scenarios():
    rows = []
    rows.extend(("short_social", "random_reply", f"ну что нового номер {i}") for i in range(10))
    rows.extend(("useful", "question", f"стул почему сервис ведёт себя странно номер {i}") for i in range(10))
    rows.extend(("long", "question", topic) for topic in LONG_TOPICS)
    rows.extend(
        ("troll_user", "troll_user", (
            f"стул как прославиться в рэпе номер {i}" if i < 5
            else f"стул как сварить рис номер {i}"
        )) for i in range(10)
    )
    rows.extend(("recommendation", "question", f"стул стоит ли выбрать vps вариант {i}") for i in range(10))
    rows.extend(("meme", "meme_caption", f"реальная цитата про релиз номер {i}") for i in range(10))
    return rows


def run_smoke():
    provider = SmokeProvider()
    temporary = tempfile.TemporaryDirectory()
    service = LearningService(LearningSettings(
        data_dir=Path(temporary.name), openai_chat_id=-1,
        min_training_messages=1,
    ), llm_provider=provider)
    service.repository(-1).remember_stable([
        "рэп это хуйня, больше не пишу",
        "сервер каждую пятницу падает после маленького фикса",
    ])
    report = []
    for position, (category, purpose, context) in enumerate(scenarios()):
        if position == 30:
            for tick in (
                "ну классика снова приехала",
                "классика, проект опять живёт",
                "классика жанра в проде",
            ):
                service.repository(-1).record_generated(tick, "smoke_seed")
        result = service.generate_llm(-1, context, purpose)
        request = provider.calls[-1]
        row = {
            "category": category,
            "purpose": request.metadata["response_purpose"],
            "producer": "llm",
            "output_budget": request.max_output_tokens,
            "final_length": len(result or ""),
            "truncated": bool(getattr(result, "incomplete_reason", None)),
            "lexical_penalty": bool(request.metadata.get("lexical_penalties")),
            "callback_used": bool(request.metadata.get("selected_callbacks")),
            "text": result,
        }
        report.append(row)
    return temporary, provider, report


class ProductionLikeQualitySmokeTests(unittest.TestCase):
    def test_sixty_events_have_one_call_and_complete_outputs(self):
        temporary, provider, report = run_smoke()
        try:
            self.assertEqual(len(report), 60)
            self.assertEqual(len(provider.calls), 60)
            self.assertTrue(all(row["text"] for row in report))
            self.assertFalse(any(row["truncated"] for row in report))
            long_rows = [row for row in report if row["category"] == "long"]
            self.assertEqual(len(long_rows), 10)
            broken = ("лук с", "хараки", "обжа", "настро")
            self.assertFalse(any(
                row["text"].casefold().rstrip().endswith(broken) for row in long_rows
            ))
            self.assertTrue(all(row["final_length"] > 500 for row in long_rows))
            counts = Counter(row["purpose"] for row in report)
            print("QUALITY_SMOKE_AGGREGATE=" + json.dumps({
                "events": len(report), "calls": len(provider.calls),
                "purposes": counts, "truncated": 0,
                "long_complete": len(long_rows),
                "callback_used": sum(row["callback_used"] for row in report),
                "lexical_penalty": sum(row["lexical_penalty"] for row in report),
                "budgets": sorted(set(row["output_budget"] for row in report)),
            }, ensure_ascii=False, default=dict))
            print("QUALITY_SMOKE_LONG=" + json.dumps(
                [{key: row[key] for key in ("purpose", "output_budget", "final_length", "truncated", "text")}
                 for row in long_rows], ensure_ascii=False
            ))
            print("QUALITY_SMOKE_TROLL=" + json.dumps(
                [row["text"] for row in report if row["category"] == "troll_user"],
                ensure_ascii=False,
            ))
            print("QUALITY_SMOKE_MEME=" + json.dumps(
                [row["text"] for row in report if row["category"] == "meme"],
                ensure_ascii=False,
            ))
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()

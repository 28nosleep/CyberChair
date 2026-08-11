import re
from dataclasses import dataclass


def _normalized(text):
    text = str(text or "").casefold().replace("ё", "е")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", text)).strip()


@dataclass(frozen=True)
class MemeEntry:
    id: str
    output: str
    aliases: tuple[str, ...]
    meaning: str
    contexts: tuple[str, ...]
    intensity_min: float = 0.4
    weight: float = 1.0
    cooldown_group: str | None = None

    def recognizes(self, text):
        haystack = f" {_normalized(text)} "
        return any(f" {_normalized(alias)} " in haystack for alias in self.aliases)


def _entry(identifier, output, aliases, meaning, contexts, intensity=0.4, group=None):
    return MemeEntry(
        identifier, output, tuple(dict.fromkeys((output, *aliases))), meaning,
        tuple(contexts), intensity, 1.0, group or identifier,
    )


class MemeLexicon:
    """Small, local and versioned vocabulary. Aliases are recognition-only."""

    VERSION = "1.0.0"

    def __init__(self, entries=None):
        self.entries = tuple(entries or DEFAULT_ENTRIES)
        self._by_id = {entry.id: entry for entry in self.entries}

    def get(self, identifier):
        return self._by_id.get(identifier)

    def recognize(self, text):
        return [entry for entry in self.entries if entry.recognizes(text)]

    def select(self, text, contexts, intensity, excluded_ids=(), excluded_groups=(), limit=3):
        contexts = set(contexts or ())
        recognized = {entry.id for entry in self.recognize(text)}
        excluded_ids, excluded_groups = set(excluded_ids), set(excluded_groups)
        signal_contexts = set(contexts)
        normalized = _normalized(text)
        signals = {
            "failure": r"\b(?:упал|сломал|ошиб|провал|факап|не работает|откат|роллбек)\w*\b",
            "overengineering": r"\b(?:архитект|микросервис|абстракц|энтерпрайз|сложн)\w*\b",
            "praise": r"\b(?:лучший|идеальн|гений|легенд|хвал|обожаю)\w*\b",
            "focus": r"\b(?:сосредоточ|работаю|делаю|пишу|чиню)\w*\b",
            "conspiracy": r"\b(?:заговор|теори|тайн|рептилоид|подзем)\w*\b",
            "work": r"\b(?:релиз|деплой|прод|баг|сервер|дедлайн|джира|созвон)\w*\b",
        }
        signal_contexts.update(name for name, pattern in signals.items() if re.search(pattern, normalized))
        scored = []
        for entry in self.entries:
            explicit = entry.id in recognized
            if entry.intensity_min > intensity and not explicit:
                continue
            if entry.id in excluded_ids or entry.cooldown_group in excluded_groups:
                continue
            overlap = len(set(entry.contexts) & signal_contexts)
            if not explicit and not overlap:
                continue
            scored.append((10 if explicit else 0, overlap, entry.weight, entry.id, entry))
        scored.sort(reverse=True)
        return [item[-1] for item in scored[: max(0, int(limit))]]


DEFAULT_ENTRIES = (
    _entry("soy", "соя", ("soy",), "иронически беззубая или показная реакция", ("mocking", "humor"), .45),
    _entry("soyish", "соевый", ("soyish",), "показно мягкий или нелепый образ", ("mocking",), .5, "soy"),
    _entry("soyjak", "сойджак", ("soyjak",), "карикатурно эмоциональная реакция", ("mocking", "humor"), .55),
    _entry("wojak", "вояк", ("wojak",), "типовой интернет-образ участника", ("mocking", "humor"), .5),
    _entry("chud", "чад", ("chud", "chad"), "гиперболизированный грубый персонаж", ("mocking", "humor"), .55),
    _entry("based", "бейсд", ("based",), "постироническое одобрение", ("humor", "praise"), .35),
    _entry("base", "база", (), "уверенное постироническое одобрение", ("humor", "praise"), .3, "based"),
    _entry("based_ru", "базированный", (), "иронически принципиальная позиция", ("humor", "praise"), .35, "based"),
    _entry("cringe", "кринж", ("cringe",), "неловкий социальный провал", ("mocking", "failure"), .3),
    _entry("gem", "гем", ("gem",), "иронически отличный контент", ("humor", "praise"), .45),
    _entry("coal", "коал", ("coal",), "провальный или унылый контент", ("mocking", "failure"), .5),
    _entry("seethe", "сид", ("seethe",), "ироническое описание сильного раздражения", ("argument", "mocking"), .65),
    _entry("cope", "коуп", ("cope",), "самоуспокоительное оправдание", ("argument", "mocking"), .55),
    _entry("dilate", "дилейт", ("dilate",), "абсурдная имиджбордная отсылка", ("humor", "mocking"), .75),
    _entry("brainrot", "брейнрот", ("brainrot",), "обсуждение поглотил интернетный абсурд", ("humor",), .45),
    _entry("cooked", "кукд", ("cooked",), "ситуация очевидно идёт плохо", ("failure", "mocking"), .45),
    _entry("let_him_cook", "дай ему готовить", ("let him cook", "let him cooke"), "постиронически дать идее развиться", ("humor", "praise"), .4),
    _entry("locked_in", "локед ин", ("locked in",), "кто-то чрезмерно сосредоточен", ("focus", "work", "humor"), .35),
    _entry("aura", "аура", ("aura",), "иронический социальный статус", ("humor", "mocking"), .4),
    _entry("aura_loss", "минус аура", ("aura loss", "loss of aura"), "комический социальный провал", ("failure", "mocking"), .45, "aura"),
    _entry("mogging", "моггинг", ("mogging", "mog"), "демонстративное превосходство", ("mocking", "humor"), .55),
    _entry("mog", "мог", (), "краткое описание превосходства в сравнении", ("mocking", "humor"), .55, "mogging"),
    _entry("glazing", "глейзинг", ("glazing",), "чрезмерная похвала кому-то", ("praise", "mocking"), .5),
    _entry("crashout", "крашаут", ("crashout",), "демонстративный эмоциональный срыв", ("argument", "humor"), .65),
    _entry("delulu", "делулу", ("delulu",), "постироническая оторванность идеи от реальности", ("humor", "mocking"), .45),
    _entry("npc", "нпс", ("npc",), "шаблонное повторяющееся поведение", ("mocking",), .5),
    _entry("touch_grass", "потрогай траву", ("touch grass",), "пора выйти из интернетного зацикливания", ("mocking", "argument"), .6),
    _entry("skill_issue", "скилл ишью", ("skill issue",), "ироническое объяснение неудачи нехваткой навыка", ("failure", "mocking"), .45),
    _entry("rent_free", "рент фри", ("rent free",), "идея слишком долго сидит у кого-то в голове", ("argument", "mocking"), .55),
    _entry("bro_thinks", "бро думает", ("bro thinks",), "насмешка над самоуверенной ролью", ("mocking", "humor"), .5),
    _entry("lil_bro", "лил бро", ("lil bro",), "снисходительное дружеское обращение", ("mocking", "humor"), .55),
    _entry("mid", "мид", ("mid",), "посредственный результат", ("failure", "mocking"), .35),
    _entry("peak", "пик", ("peak",), "гиперболизированно высокий уровень", ("praise", "humor"), .35),
    _entry("canon_event", "канон ивент", ("canon event",), "неизбежный комический эпизод", ("failure", "humor"), .45),
    _entry("main_character", "мейн персонаж", ("main character",), "кто-то ведёт себя как главный герой", ("mocking", "humor"), .5),
    _entry("side_quest", "сайдквест", ("side quest",), "побочная задача вместо главной", ("work", "humor"), .35),
    _entry("sigma", "сигма", ("sigma",), "постироническая самодостаточность", ("humor", "mocking", "praise"), .4),
    _entry("gigachad", "гигачад", ("gigachad",), "гиперболизированное одобрение или ирония", ("praise", "humor"), .45),
    _entry("alpha", "альфа", ("alpha",), "только ироническая классификация", ("mocking", "humor"), .5),
    _entry("grindset", "грайндсет", ("grindset",), "карикатурная одержимость продуктивностью", ("work", "mocking"), .45),
    _entry("normie", "нормис", ("normie",), "обычный немемный участник", ("mocking",), .45),
    _entry("skuf", "скуф", ("скуф",), "мемный бытовой архетип", ("mocking", "humor"), .5),
    _entry("altushka", "альтушка", ("альтушка",), "мемный интернет-архетип", ("humor",), .45),
    _entry("larp", "ларп", ("larp", "ларпить"), "ролевая имитация серьёзной деятельности", ("mocking", "work"), .4),
    _entry("larping", "ларпить", (), "изображать деятельность или роль", ("mocking", "work"), .4, "larp"),
    _entry("enterprise_larp", "энтерпрайз ларп", ("enterprise larp",), "избыточно сложное рабочее решение", ("overengineering", "work", "mocking"), .4, "larp"),
    _entry("shizotheory", "шизотеория", ("шиза",), "мемное описание абсурдной идеи, не диагноз", ("conspiracy", "humor"), .55),
    _entry("shiza", "шиза", (), "мемное описание ситуации или идеи, не человека и не диагноз", ("conspiracy", "humor"), .55, "shizotheory"),
    _entry("warmup", "прогрев", ("прогрев",), "подготовка к сомнительной продаже или идее", ("mocking",), .4),
    _entry("infogypsy", "инфоцыганство", ("инфоцыган",), "показная продажа пустого знания", ("mocking",), .55),
    _entry("torn", "порвался", ("порвался",), "ироническая чрезмерная реакция", ("argument", "humor"), .6),
    _entry("dumped", "высрал", ("высрал",), "резко выдал сомнительную идею", ("mocking",), .7),
    _entry("bore", "душнила", ("душнила",), "чрезмерно педантичная реплика", ("serious", "mocking"), .45),
    _entry("terpila", "терпила", (), "редкий дружеский стёб, не постоянная персональная травля", ("mocking", "humor"), .75),
    _entry("legend", "легенда", ("легенда",), "гиперболизированное одобрение", ("praise", "humor"), .35),
    _entry("goida", "гойда", ("гойда",), "только абсурдная постироническая реакция", ("humor",), .65),
    _entry("agartha", "агарта", ("agartha",), "абсурдный псевдозаговор", ("conspiracy", "humor"), .55),
    _entry("hyperborea", "гиперборея", ("hyperborea",), "абсурдный псевдоисторический слой", ("conspiracy", "humor"), .55),
    _entry("friday_deploy", "пятничный деплой", ("friday deploy",), "рискованный релиз перед выходными", ("work", "failure"), .35),
    _entry("works_on_machine", "у меня работает", ("works on my machine",), "локальное оправдание проблем в окружении", ("work", "failure"), .35),
    _entry("architect_moment", "архитектор момент", ("architect moment",), "ирония над архитектурным усложнением", ("work", "overengineering"), .4),
    _entry("senior_moment", "синьор момент", ("senior moment",), "ирония над уверенным техническим решением", ("work", "humor"), .4),
    _entry("junior_moment", "джун момент", ("junior moment",), "ирония над простой технической ошибкой", ("work", "failure"), .4),
    _entry("prod", "прод", ("production",), "рабочее production-окружение", ("work", "failure"), .3),
    _entry("legacy", "легаси", ("legacy",), "старый код или система с накопленными ограничениями", ("work", "overengineering"), .3),
    _entry("crutch", "костыль", ("workaround",), "временное техническое решение", ("work", "failure"), .3),
    _entry("enterprise", "энтерпрайз", ("enterprise",), "корпоративная сложность", ("work", "overengineering"), .3),
    _entry("microservices", "микросервисы", ("microservices",), "распределённая архитектура", ("work", "overengineering"), .35),
    _entry("kubernetes", "кубер", ("kuber", "kubernetes", "кубернетес"), "оркестрация контейнеров", ("work", "overengineering"), .35),
    _entry("docker", "докер", ("docker",), "контейнерный рабочий контекст", ("work",), .3),
    _entry("jira", "джира", ("jira",), "тикеты и корпоративный процесс", ("work",), .3),
    _entry("excel", "эксель", ("excel",), "табличный корпоративный процесс", ("work", "humor"), .3),
    _entry("meeting", "созвон", ("call", "meeting"), "рабочая встреча", ("work",), .3),
    _entry("deadline", "дедлайн", ("deadline",), "крайний срок", ("work", "failure"), .3),
    _entry("release", "релиз", ("release",), "выпуск версии", ("work",), .3),
    _entry("deploy", "деплой", ("deploy",), "выкладка версии", ("work", "failure"), .3),
    _entry("rollback", "роллбек", ("rollback",), "откат неудачного изменения", ("work", "failure"), .3),
)

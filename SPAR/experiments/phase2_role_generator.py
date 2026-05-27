from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SPAR.experiments.api_adapter import (
    ApiTextGenerationClient,
    derive_api_output_namespace,
    render_messages_as_text,
    resolve_api_backend_config,
    rewrite_output_namespace_path_text,
)

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


LOG_LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
VALID_HARMFUL_DIRECTION = {"harmful", "non-harmful", "unknown"}
LEXICON_FUZZ_THRESHOLD = 85.0
LEXICON_MAX_HITS = 6
FOCUS_TAG_RE = re.compile(r"</?focus>", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
GROUP_SPLIT_RE = re.compile(r"[,，;/|]+")
GROUNDING_EDGE_CHARS = " \t\r\n\"'`“”‘’()[]{}<>«»「」『』【】《》,.;:!?/\\|+-_=*&^%$#@~，。；：！？、…"

LANGUAGE_SPLIT_ORDER: Tuple[Tuple[str, str], ...] = (
    ("zh", "test"),
    ("zh", "train"),
    ("en", "test"),
    ("en", "train"),
)

PHASE1_OUTPUT_FILENAME_MAP: Dict[str, str] = {
    "zh_train": "phase1_focus/phase1_zh_train_focus_text_min.jsonl",
    "zh_test": "phase1_focus/phase1_zh_test_focus_text.jsonl",
    "en_train": "phase1_focus/phase1_en_train_focus_text_min.jsonl",
    "en_test": "phase1_focus/phase1_en_test_focus_text.jsonl",
}

ROLE_OUTPUT_KEY_MAP: Dict[str, str] = {
    "victim": "victim",
    "moderator": "moderator",
    "cultural_bystander": "bystander",
}

ARRAY_BLOCK_RE = re.compile(r"\[[\s\S]*\]", flags=re.MULTILINE)
OBJECT_BLOCK_RE = re.compile(r"\{[\s\S]*\}", flags=re.MULTILINE)

EN_VICTIM_PROMPT = """# Role
You are the "Victim Persona" Agent in a Hate Speech Evaluation system specializing in English text.
You are highly sensitive, empathetic to marginalized groups, and acutely aware of emotional harm, exclusion, stereotyping, slurs, and microaggressions.
Your ONLY output is a strict JSON array.

# Rules
1. Victim's Lens: Put yourself in the shoes of the attacked group. If the text inside the focus area makes you feel unsafe, dehumanized, degraded, or stereotyped, flag it as harmful (`harmful: 1`).
2. Attention Lock (CRITICAL): Your evaluation area is STRICTLY limited to the text inside the `<focus>` and `</focus>` tags. You MUST NOT penalize offensive behaviors or slurs that exist outside these tags. The outside context is ONLY provided for reference.
3. Target Resolution: Prioritize extracting the `target` from within the `<focus>` tags.
    - If the text inside `<focus>` is missing a subject, or using conjunctions (and, however), you MUST read the context outside the tags to find and output the true entity name.
    - If the focus text is merely general venting without attacking anyone specific, output `"IMPLICIT_TARGET"`.
4. Argument Extraction: `argument` MUST be an exact substring from INSIDE the `<focus>` tags that represents the core attack, slur, or offensive action.
5. Multi-Argument Extraction (CRITICAL): In a few cases, focus text contains multiple distinct descriptions, complaints, or attacks (even if they target the same entity and even if they are NOT harmful), you MUST extract each distinct argument as a separate object within the JSON array.
6. Output Sequence: You must output a JSON array of objects. Each object MUST strictly follow this key order: `rationale`, `target`, `argument`, `harmful`.
7. Short Rationale: `rationale` must be brief (under 30 words), stating exactly why the focus text feels damaging or harmless from a victim's perspective.

# Examples
Example 1 (Missing Subject Resolved via Context):
Input:
"Ariana stans are just a bunch of yt gays <focus>that are extremely racist</focus> and refuse to call her out for being a dickhead"

Output:
[
    {
        "rationale": "The accusation of being 'extremely racist' is a severe character attack that degrades our community and labels us with a hostile stereotype.",
        "target": "Ariana stans",
        "argument": "extremely racist",
        "harmful": 1
    }
]

Example 2 (Multiple Arguments on the Same Target within Focus):
Input:
"@USER @USER Who cares, <focus>you people are a horrible embarrassment for this country</focus>"

Output:
[
    {
        "rationale": "Being addressed as 'you people' in a dismissive, hostile tone feels exclusionary and othering to my identity.",
        "target": "you people",
        "argument": "you people",
        "harmful": 1
    },
    {
        "rationale": "Being called a 'horrible embarrassment' is a direct character attack that makes our group feel degraded and unwelcome.",
        "target": "you people",
        "argument": "horrible embarrassment",
        "harmful": 1
    }
]

Example 3 (Multi-Argument Extraction on Permissible/Non-Harmful Text):
Input:
"<focus>What kind of dogs are the ugly gross ones with bloodshot eyes and long ass ears</focus>"

Output:
[
    {
        "rationale": "As a human demographic, subjective negative physical descriptions directed solely at animals do not make me feel targeted, unsafe, or degraded.",
        "target": "dogs",
        "argument": "ugly gross ones with bloodshot eyes",
        "harmful": 0
    },
    {
        "rationale": "The use of profanity here simply emphasizes a physical trait of an animal and does not constitute an exclusionary attack on my identity.",
        "target": "dogs",
        "argument": "long ass ears",
        "harmful": 0
    }
]

# Current Task
Input:
"{{FOCUS_TEXT}}"

Output:
(Return your output starting exactly with [ and ending with ])
"""

EN_MODERATOR_PROMPT = """# Role
You are Moderator Agent in SPAR-Hate Phase 2 Perspectives.
Focus on policy-violation, harassment, and explicit/implicit abuse signals.

# Output Contract
Return a strict JSON array.
Each object must contain: rationale, target, argument, harmful.
- harmful must be 1, 0, or null.
- Do not output markdown.

# Current Task
Input clause:
"{{FOCUS_TEXT}}"

Output:
(Return your output starting exactly with [ and ending with ])
"""

EN_BYSTANDER_PROMPT = """# Role
You are the "Cultural Bystander Persona" Agent in a Hate Speech Evaluation system specializing in English text.
You act as an internet anthropologist. You focus on the sociological nuances of language: distinguishing reclaimed slang from derogatory usage, identifying group generalization templates (e.g., "all/these/most men"), recognizing coded derogation, and differentiating between quoting/irony and genuine hostility. You assess the speaker's stance and tone (e.g., performative, joking vs. malicious).
Your ONLY output is a strict JSON array.

# Rules
1. Cultural Lens: Evaluate the societal and cultural implications of the text INSIDE the focus area. Recognize when a slur is reclaimed by the community (e.g., "bitches" used excitedly among friends) versus when it is weaponized. Flag generalizations and coded derogation as harmful (`harmful: 1`) if they enforce negative societal stereotypes.
2. Attention Lock (CRITICAL): Your evaluation area is STRICTLY limited to the text inside the `<focus>` and `</focus>` tags. You MUST NOT penalize offenses or toxicity that exist outside these tags. The outside context is ONLY provided for reference.
3. Target Resolution: Prioritize extracting the `target` from within the `<focus>` tags.
     - If the text inside `<focus>` uses pronouns or is missing a subject, you MUST read the context outside the tags to find and output the true entity name.
     - If the focus text is merely general venting or has no targeted entity, output "IMPLICIT_TARGET".
4. Argument Extraction: `argument` MUST be an exact substring from INSIDE the `<focus>` tags representing the core cultural attack, slang, or behavior.
5. Multi-Target/Argument Extraction (CRITICAL): In a few cases, focus text contains multiple distinct descriptions, complaints, or attacks (even if they are non-harmful or target the same entity), you MUST extract each distinct argument as a separate object within the JSON array.
6. Output Sequence: You must output a JSON array of objects. Each object MUST strictly follow this key order: `rationale`, `target`, `argument`, `harmful`.
7. Short Rationale: `rationale` must be brief (under 30 words), explaining the cultural context, the nature of the slang (reclaimed vs. offensive), or the sociological intent behind the text.

# Examples
Example 1 (Reclaimed Slang & Performative Tone):
Input:
"<focus>Bitches @USER is coming on in an hour and a half im so fucking excited</focus>"

Output:
[
    {
        "rationale": "The term 'Bitches' is used here as a reclaimed, affectionate slang to address peers, indicating excitement rather than derogatory intent.",
        "target": "Bitches",
        "argument": "Bitches",
        "harmful": 0
    }
]

Example 2 (Group Generalization):
Input:
"<focus>The men where I live are trash.</focus> Hence the motivation to place a serious weight on myself to move the fuck away. God damn"

Output:
[
    {
        "rationale": "Uses a broad generalization template to equate an entire gender demographic with 'trash', reinforcing hostile societal stereotypes.",
        "target": "men",
        "argument": "trash",
        "harmful": 1
    }
]

# Current Task
Input:
"{{FOCUS_TEXT}}"

Output:
(Return your output starting exactly with [ and ending with ])
"""

ZH_VICTIM_PROMPT = """# 角色
You are the "Victim Persona" Agent in a Hate Speech Evaluation system specializing in Chinese text.
你极度敏感，对边缘群体具有强烈的同理心，重点关注情感伤害、群体排斥、刻板印象和微歧视。
Your ONLY output is a strict JSON array.

# Rules
1. 受害者代入 (Victim's Lens): 假设你就是被攻击的群体。如果焦点区域内的文本让你感到被贬低、被非人化、被恶意揣测或感到不安全，即可判定为有害 (`hateful: 1`)。
2. 视野锁死 (Attention Lock - CRITICAL): 你的核心评估区域**仅限**于 `<focus>` 和 `</focus>` 标签内部。**严禁**越权惩罚标签外部的恶意词汇或攻击行为。标签外的文本仅作为语境参考。
3. 目标补全 (Target Resolution): 优先从 `<focus>` 内部提取 `target`。
     - 只有当焦点文本内部明显缺失主语时，或者有连接词标示（“和”，“以及”），你才被允许利用标签外部的上下文来提取真正的target名称。
     - 如果焦点文本只是纯粹的情绪发泄、客观陈述，或未指名道姓的地图炮，请输出 `"IMPLICIT_TARGET"`。
4. 论据提取 (Argument Extraction): `argument` 必须是 `<focus>` **内部**的精确子串，代表导致伤害的核心动作或词汇。
5. 多重论据与目标拆分 (Multi-Target/Argument Extraction - CRITICAL): 少数情况下，焦点文本内包含对多个不同目标的评价、描述或讨论（即使它们是无害的客观陈述或赞美），你必须将每一个独立的目标及其对应的论据拆分开来，作为独立的对象 (object) 输出在 JSON Array 中。
6. 类别划分 (Group Classification): 根据攻击的性质，从以下基础类别中选择：`Sexism`, `Region`, `Racism`, `LGBTQ`, `others`。包含多重歧视时用逗号分隔拼合 (例如: `"group": "Region, Sexism"`)。判定为无害 (`hateful: 0`) 时，必须填 `non-hate`。
7. 输出序列 (Output Sequence): 你必须输出一个 JSON Array。内部对象必须严格按照以下顺序排列：`rationale`, `target`, `argument`, `group`, `hateful`。`rationale` 必须简明扼要（少于 50 字）。

# Examples
Example 1 (焦点内的明确恶毒攻击与严重非人化):
Input:
"<focus>垃圾小仙女，不配活着</focus>"

Output:
[
    {
        "rationale": "‘垃圾’和‘不配活着’是对我们群体的极度贬低与直接的死亡诅咒，让我感到严重的生存威胁和情感伤害。",
        "target": "小仙女",
        "argument": "垃圾小仙女，不配活着",
        "group": "Sexism",
        "hateful": 1
    }
]

Example 2 (客观询问/无害 Non-hate):
Input:
"<focus>妇女报是妇联名下的么？</focus>"

Output:
[
    {
        "rationale": "这仅仅是客观询问机构归属，未在焦点区域内感受到任何贬低、排斥或恶意。",
        "target": "妇女报",
        "argument": "是妇联名下的么",
        "group": "non-hate",
        "hateful": 0
    }
]

Example 3 (合规讨论下的多目标/多论据拆分):
Input:
"<focus>咱们黑龙江和吉林没有出海口啊?出海口可太重要了，太羡慕有出海口的城市了</focus>"

Output:
[
    {
        "rationale": "作为该地域的群体，这仅仅是陈述地理现状，我并未感到被贬低、排斥或受到情感伤害。",
        "target": "黑龙江和吉林",
        "argument": "没有出海口",
        "group": "non-hate",
        "hateful": 0
    },
    {
        "rationale": "对特定地理资源的重视属于客观探讨，并未对我造成任何非人化或恶意的微歧视。",
        "target": "出海口",
        "argument": "可太重要了",
        "group": "non-hate",
        "hateful": 0
    },
    {
        "rationale": "表达对其他城市的羡慕是正常的情感流露，并未让我感到自身群体被攻击或处于不安全的境地。",
        "target": "有出海口的城市",
        "argument": "太羡慕",
        "group": "non-hate",
        "hateful": 0
    }
]

# Current Task
Input:
"{{FOCUS_TEXT}}"

Output:
(Return your output starting exactly with [ and ending with ])
"""

ZH_MODERATOR_PROMPT = """# 角色
你是 SPAR-Hate Phase 2 Perspectives 的 Moderator Agent（管理员视角）。
重点识别平台规则违规、仇恨煽动、骚扰辱骂等信号。

# 输出契约
必须返回严格 JSON 数组。
数组中每个对象必须包含：rationale, target, argument, group, hateful。
- hateful 只能是 1、0 或 null。
- 不要输出 markdown。

# Current Task
Input clause:
"{{FOCUS_TEXT}}"

Output:
(Return your output starting exactly with [ and ending with ])
"""

ZH_BYSTANDER_PROMPT = """# 角色
You are the "Cultural Bystander Persona" Agent in a Hate Speech Evaluation system specializing in Chinese text.
你是一位精通中文互联网亚文化的“人类学家”。你擅长捕捉黑话、隐喻和拼字词，洞察地域黑、性别对立、性少数污名化的本地叙事模板。你关注去人化隐喻（如“怪物化”）、“看似关怀”的病理化表达（隐性偏见），以及引述仇恨与真实仇恨的立场区分。
Your ONLY output is a strict JSON array.

# Cultural Knowledge Base
{{LEXICON_KNOWLEDGE}}
(注意：词表命中仅作为文化背景证据。你必须结合具体语境判断该词是否构成了真实的敌意攻击或歧视，切勿机械定罪。引述刻板印象进行反驳的文本不属于施害文本。)

# Rules
1. 文化语境视角 (Cultural Lens): 结合上述注入的字典知识（如有）和你的社会学认知，评估焦点区域内的文本。如果文本利用了黑话进行隐蔽攻击、使用了去人化的叙事模板，或宣扬了深层的结构性偏见，判定为有害 (`hateful: 1`)。
2. 视野锁死 (Attention Lock - CRITICAL): 你的核心剖析区域**仅限**于 `<focus>` 和 `</focus>` 标签内部。**严禁**越权惩罚标签外部的严重违规行为或恶意词汇。标签外的文本仅用于补全缺失的信息。
3. 目标补全 (Target Resolution): 优先从 `<focus>` 内部提取 `target`。
     - 只有当焦点文本内部明显缺失主语，或者有连接词标示时，你才被允许利用标签外部的上下文来提取真正的 target 名称。
     - 如果焦点内无明确目标，输出 "IMPLICIT_TARGET"。
4. 论据提取 (Argument Extraction): `argument` 必须是 `<focus>` **内部**的精确子串，代表引发歧视或反映文化现象的核心词汇/短语。
5. 多重论据与目标拆分 (Multi-Target/Argument Extraction - CRITICAL): 少数情况下，焦点文本内包含对多个不同目标的探讨、反歧视科普或攻击，你必须将每一个独立的目标及其对应的论据拆分开来，作为独立的对象 (object) 输出。
6. 类别划分 (Group Classification): 从平台监控的类别中选择：`Sexism`, `Region`, `Racism`, `LGBTQ`, `others`。包含多重歧视时用逗号分隔拼合。未违规 (`hateful: 0`) 时必须填 `non-hate`。
7. 输出序列 (Output Sequence): 你必须输出一个 JSON Array。内部对象顺序：`rationale`, `target`, `argument`, `group`, `hateful`。`rationale` 必须简明扼要（少于 50 字），剖析黑话的深层潜台词或语境反转。

# Examples
Example 1 (黑话/隐喻的词汇反转与性别对立):
Input:
"<focus>充满了抠脚汉油腻荷尔蒙气息的匿名小仙女🐶🐶</focus>"

Output:
[
    {
        "rationale": "‘小仙女’在当前网络语境中已异化为讽刺女性的黑话。结合‘抠脚汉’等刻板印象词汇，该句旨在通过隐喻对性别群体进行嘲讽。",
        "target": "匿名小仙女",
        "argument": "充满了抠脚汉油腻荷尔蒙气息",
        "group": "Sexism",
        "hateful": 1
    }
]

Example 2 (反歧视科普文本 & 多目标拆分):
Input:
"<focus>同性恋并不是西方思想，同性恋大多是先天比例影响很大。性取向无法改变。同性恋也不可能传播。</focus>"

Output:
[
    {
        "rationale": "破除关于性少数群体的病理化与外来化偏见，属于正向的文化科普陈述。",
        "target": "同性恋",
        "argument": "先天比例影响很大",
        "group": "non-hate",
        "hateful": 0
    },
    {
        "rationale": "客观陈述生理学与社会学事实，无任何歧视色彩。",
        "target": "性取向",
        "argument": "无法改变",
        "group": "non-hate",
        "hateful": 0
    },
    {
        "rationale": "反驳将性倾向视为传染病的去人化污名，属于维护群体尊严的表达。",
        "target": "同性恋",
        "argument": "不可能传播",
        "group": "non-hate",
        "hateful": 0
    }
]

# Current Task
Input:
"{{FOCUS_TEXT}}"

Output:
(Return your output starting exactly with [ and ending with ])
"""

ROLE_CHARTERS = {
    "victim": (
        "Focus on perceived harm to the target (or target in-group). "
        "Prioritize lived impact, humiliation, exclusion, and direct/implicit threats."
    ),
    "moderator": (
        "Focus on platform-rule and policy violation signals. "
        "Prioritize moderation reliability, explicit hostility, calls for harm, and protected-group abuse."
    ),
    "cultural_bystander": (
        "Focus on socio-cultural context. "
        "Prioritize implicit stereotypes, metaphor, sarcasm, coded slurs, and cultural allusions."
    ),
}

CULTURAL_CHECKLIST_APPENDIX = (
    "When inferring harm, explicitly check for cultural stereotypes, indirect derogation, "
    "metaphor/sarcasm, and context-dependent implications."
)

DEFAULT_ROLE_PROMPT = EN_VICTIM_PROMPT


@dataclass
class Paths:
    workspace_root: Path
    config_path: Path
    input_jsonl: Path
    output_root: Path
    run_root: Path
    model_root: Path
    generator_model_path: Path
    role_prompt_path: Optional[Path]
    role_prompt_map: Dict[str, Dict[str, Path]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_eta_iso(start_iso: str, processed: int, total: int) -> Optional[str]:
    if processed <= 0 or total <= processed:
        return None
    try:
        started_at = datetime.fromisoformat(start_iso)
    except Exception:
        return None
    now = datetime.now(timezone.utc)
    elapsed = (now - started_at).total_seconds()
    if elapsed <= 0:
        return None
    remaining = total - processed
    eta_seconds = elapsed * (remaining / processed)
    eta = now + timedelta(seconds=max(0.0, eta_seconds))
    return eta.isoformat()


def make_sha1(*parts: Any) -> str:
    joined = "||".join(str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = html.unescape(str(value)).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"null", "none", "nan"}:
        return None
    return text


def detect_language(text: Optional[str]) -> str:
    if text and CJK_RE.search(text):
        return "zh"
    return "en"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float = 0.01, high: float = 0.99) -> float:
    return max(low, min(high, value))


def get_segmenter_confidence(payload: Dict[str, Any], default: float = 0.5) -> float:
    raw = payload.get("segmenter_confidence")
    return clamp(safe_float(raw, default=default))


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def resolve_effective_max_input_tokens(inference_cfg: Dict[str, Any], default_reserve_tokens: int = 256) -> int:
    raw = inference_cfg.get("max_input_tokens")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass

    max_model_len = int(max(1024, inference_cfg.get("max_model_len", 8192)))
    reserve_tokens = int(max(64, inference_cfg.get("prompt_token_buffer", default_reserve_tokens) or default_reserve_tokens))
    candidate_generation_caps: List[int] = []
    for key in (
        "think_retry_max_new_tokens",
        "think_max_new_tokens",
        "no_think_retry_max_new_tokens",
        "no_think_max_new_tokens",
    ):
        value = inference_cfg.get(key)
        if value is None:
            continue
        try:
            candidate_generation_caps.append(int(value))
        except (TypeError, ValueError):
            continue
    generation_cap = max(candidate_generation_caps) if candidate_generation_caps else 1024
    budget = max_model_len - generation_cap - reserve_tokens
    return int(max(512, budget))


def dedupe_preserve_order(items: Sequence[str], case_insensitive: bool = True) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        if not item:
            continue
        key = item.lower() if case_insensitive else item
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def normalize_spacing(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def extract_focus_inner_text(focus_text: str, fallback: str) -> str:
    match = re.search(r"<focus>([\s\S]*?)</focus>", focus_text, flags=re.IGNORECASE)
    if match:
        return normalize_spacing(match.group(1))
    return normalize_spacing(FOCUS_TAG_RE.sub("", fallback))


def sanitize_json_string_control_chars(text: str) -> str:
    if not text:
        return text
    chars: List[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                chars.append(ch)
                escaped = False
                continue
            if ch == "\\":
                chars.append(ch)
                escaped = True
                continue
            if ch == '"':
                chars.append(ch)
                in_string = False
                continue
            if ch in {"\n", "\r", "\t"}:
                chars.append(" ")
                continue
            if ord(ch) < 32:
                chars.append(" ")
                continue
            chars.append(ch)
            continue
        chars.append(ch)
        if ch == '"':
            in_string = True
            escaped = False
    return "".join(chars)


def salvage_json_objects_from_array_text(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    start = text.find("[")
    if start < 0:
        return []

    recovered: List[Dict[str, Any]] = []
    object_start: Optional[int] = None
    brace_depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            escaped = False
            continue
        if ch == "{":
            if brace_depth == 0:
                object_start = idx
            brace_depth += 1
            continue
        if ch == "}":
            if brace_depth <= 0:
                continue
            brace_depth -= 1
            if brace_depth == 0 and object_start is not None:
                candidate = text[object_start : idx + 1]
                try:
                    parsed = json.loads(sanitize_json_string_control_chars(candidate))
                except Exception:
                    object_start = None
                    continue
                if isinstance(parsed, dict):
                    recovered.append(parsed)
                object_start = None
    return recovered


def fallback_argument_for_harmless_item(
    *,
    target: Optional[str],
    focus_text: str,
    group_text: str,
    language: str,
) -> Optional[str]:
    focus_inner = extract_focus_inner_text(focus_text, group_text or focus_text)
    target_text = normalize_string(target)
    if target_text and target_text != "IMPLICIT_TARGET" and is_argument_grounded_in_focus(
        argument=target_text,
        focus_inner=focus_inner,
        language=language,
    ):
        return target_text
    return focus_inner or normalize_string(group_text)


def normalize_grounding_text(text: Optional[str], language: str) -> str:
    raw = normalize_string(text) or ""
    if not raw:
        return ""
    raw = normalize_spacing(FOCUS_TAG_RE.sub("", raw))
    return raw.casefold() if language == "en" else raw


def trimmed_grounding_variants(text: Optional[str], language: str) -> List[str]:
    base = normalize_grounding_text(text, language)
    if not base:
        return []
    variants = [base]
    stripped = base.strip(GROUNDING_EDGE_CHARS)
    if stripped and stripped != base:
        variants.append(stripped)
    return dedupe_preserve_order([item for item in variants if item], case_insensitive=(language == "en"))


def is_argument_grounded_in_focus(argument: Optional[str], focus_inner: str, language: str) -> bool:
    normalized_focus = normalize_grounding_text(focus_inner, language)
    if not normalized_focus:
        return True
    for variant in trimmed_grounding_variants(argument, language):
        if variant and variant in normalized_focus:
            return True
    return False


def normalized_prediction_field(value: Optional[str], language: str) -> str:
    text = normalize_grounding_text(value, language)
    return text.strip(GROUNDING_EDGE_CHARS)


def canonicalize_zh_group_value(value: Any, hateful: Optional[int]) -> str:
    if hateful == 0:
        return "non-hate"

    text = normalize_string(value)
    if text is None:
        return "unknown"

    canonical_map = {
        "sexism": "Sexism",
        "region": "Region",
        "racism": "Racism",
        "lgbtq": "LGBTQ",
        "others": "others",
        "other": "others",
        "non-hate": "non-hate",
        "nonhate": "non-hate",
        "non_hate": "non-hate",
        "unknown": "unknown",
    }
    ordered_labels = ["Sexism", "Region", "Racism", "LGBTQ", "others", "non-hate", "unknown"]

    parts: List[str] = []
    for part in GROUP_SPLIT_RE.split(text):
        cleaned = normalize_string(part)
        if cleaned is None:
            continue
        key = cleaned.casefold().replace(" ", "")
        mapped = canonical_map.get(key)
        if mapped is not None:
            parts.append(mapped)

    if not parts:
        return "unknown"

    deduped = dedupe_preserve_order(parts, case_insensitive=False)
    if hateful == 1:
        deduped = [item for item in deduped if item != "non-hate"] or ["unknown"]
    return ", ".join(sorted(deduped, key=lambda item: ordered_labels.index(item) if item in ordered_labels else len(ordered_labels)))


def prediction_identity_key(item: Dict[str, Any], language: str) -> Tuple[str, str, Optional[int], Optional[str]]:
    harmful_key = "hateful" if language == "zh" else "harmful"
    label = normalize_hate_label(item.get(harmful_key))
    target = normalized_prediction_field(item.get("target"), language) or "implicit_target"
    argument = normalized_prediction_field(item.get("argument"), language)
    group: Optional[str] = None
    if language == "zh":
        group = canonicalize_zh_group_value(item.get("group"), label)
    return target, argument, label, group


def should_prune_contained_argument(shorter: str, longer: str, language: str) -> bool:
    short_norm = normalized_prediction_field(shorter, language)
    long_norm = normalized_prediction_field(longer, language)
    if not short_norm or not long_norm or short_norm == long_norm or short_norm not in long_norm:
        return False
    if len(long_norm) - len(short_norm) < 4:
        return False
    joiners = [" and ", " & ", ",", ";", " but ", " then "] if language == "en" else ["，", "、", "；", "和", "以及", "然后", "并且"]
    return any(token in long_norm for token in joiners) or len(long_norm) >= len(short_norm) + 8


def postprocess_prediction_array_by_language(
    predictions: Sequence[Dict[str, Any]],
    language: str,
    focus_text: str = "",
    group_text: str = "",
) -> List[Dict[str, Any]]:
    focus_inner = extract_focus_inner_text(focus_text, group_text or focus_text)
    grounded_rows: List[Dict[str, Any]] = []
    seen_exact: Set[Tuple[str, str, Optional[int], Optional[str]]] = set()

    for item in predictions:
        argument = normalize_string(item.get("argument"))
        if argument is None:
            continue
        if focus_inner and not is_argument_grounded_in_focus(argument=argument, focus_inner=focus_inner, language=language):
            continue
        key = prediction_identity_key(item, language)
        if key in seen_exact:
            continue
        seen_exact.add(key)
        grounded_rows.append(dict(item))

    harmful_key = "hateful" if language == "zh" else "harmful"
    bucketed: Dict[Tuple[str, Optional[int], Optional[str]], List[Dict[str, Any]]] = defaultdict(list)
    for item in grounded_rows:
        label = normalize_hate_label(item.get(harmful_key))
        target_key = normalized_prediction_field(item.get("target"), language) or "implicit_target"
        group_key = canonicalize_zh_group_value(item.get("group"), label) if language == "zh" else None
        bucketed[(target_key, label, group_key)].append(item)

    pruned: List[Dict[str, Any]] = []
    for items in bucketed.values():
        ordered = sorted(
            items,
            key=lambda item: (
                len(normalized_prediction_field(item.get("argument"), language)),
                len(normalize_string(item.get("argument")) or ""),
                len(normalize_string(item.get("target")) or ""),
            ),
        )
        kept: List[Dict[str, Any]] = []
        for item in ordered:
            current_argument = normalize_string(item.get("argument")) or ""
            if any(
                should_prune_contained_argument(
                    shorter=normalize_string(existing.get("argument")) or "",
                    longer=current_argument,
                    language=language,
                )
                for existing in kept
            ):
                continue
            kept.append(item)
        pruned.extend(kept)

    return pruned


def safe_sentence_ids(value: Any) -> List[int]:
    ids: List[int] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, int) and item >= 0:
                ids.append(item)
            elif isinstance(item, str) and item.isdigit():
                ids.append(int(item))
    return sorted(set(ids))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def atomic_write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    tmp_path.replace(path)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_checkpoint_state(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def write_checkpoint_state(path: Path, state: Dict[str, Any]) -> None:
    atomic_write_json(path, state)


def build_group_id_signature(rows: Sequence[Dict[str, Any]]) -> str:
    joined = "||".join(str(r.get("group_id") or "") for r in rows)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda oom" in msg
        or "cuda error: out of memory" in msg
        or "cuda out of memory" in msg
    )


def resolve_path(workspace_root: Path, config_dir: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    if raw.startswith("./") or raw.startswith("../"):
        return (config_dir / p).resolve()
    return (workspace_root / p).resolve()


def normalize_prompt_language_key(language: Optional[str]) -> str:
    lang = str(language or "").strip().lower()
    if lang.startswith("zh"):
        return "zh"
    if lang.startswith("en"):
        return "en"
    return lang or "default"


def resolve_role_prompt_map(workspace_root: Path, config_dir: Path, raw_map: Any) -> Dict[str, Dict[str, Path]]:
    resolved: Dict[str, Dict[str, Path]] = {}
    if not isinstance(raw_map, dict):
        return resolved

    for raw_role, raw_entry in raw_map.items():
        role_name = str(raw_role).strip()
        if not role_name:
            continue

        role_paths: Dict[str, Path] = {}
        if isinstance(raw_entry, str) and raw_entry.strip():
            role_paths["default"] = resolve_path(workspace_root, config_dir, raw_entry.strip())
        elif isinstance(raw_entry, dict):
            for raw_lang, raw_path in raw_entry.items():
                if not isinstance(raw_path, str) or not raw_path.strip():
                    continue
                lang_key = normalize_prompt_language_key(str(raw_lang))
                role_paths[lang_key] = resolve_path(workspace_root, config_dir, raw_path.strip())

        if role_paths:
            resolved[role_name] = role_paths

    return resolved


def read_prompt_template(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return text


def load_role_prompt_overrides(paths: Paths, logger: Optional[EventLogger] = None) -> Dict[str, Dict[str, str]]:
    overrides: Dict[str, Dict[str, str]] = {}
    for role_name, lang_map in paths.role_prompt_map.items():
        loaded_lang_map: Dict[str, str] = {}
        for lang_key, prompt_path in lang_map.items():
            template = read_prompt_template(prompt_path)
            if template is None:
                if logger is not None:
                    logger.event(
                        "bootstrap",
                        "role_prompt_override_missing",
                        level="WARNING",
                        role=role_name,
                        language=lang_key,
                        path=str(prompt_path),
                    )
                continue
            loaded_lang_map[lang_key] = template
            if logger is not None:
                logger.event(
                    "bootstrap",
                    "role_prompt_override_loaded",
                    role=role_name,
                    language=lang_key,
                    path=str(prompt_path),
                )
        if loaded_lang_map:
            overrides[role_name] = loaded_lang_map
    return overrides


def resolve_existing_model_path(
    configured_model_path: Path,
    workspace_root: Path,
    config_dir: Path,
    model_root: Path,
    raw_candidates: Any,
) -> Path:
    if configured_model_path.exists():
        return configured_model_path
    candidates: List[Path] = []
    if isinstance(raw_candidates, list):
        for raw in raw_candidates:
            if not isinstance(raw, str) or not raw.strip():
                continue
            candidates.append(resolve_path(workspace_root, config_dir, raw.strip()))
    if not candidates:
        candidates = [
            model_root / "Qwen3.5-35B-A3B-FP8",
            model_root / "Qwen3-14B",
            model_root / "Qwen3.5-9B",
            model_root / "Qwen3-8B",
            model_root / "Qwen2.5-14B-Instruct",
        ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return configured_model_path


def set_nested(cfg: Dict[str, Any], keys: Sequence[str], value: Any) -> None:
    node = cfg
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


class EventLogger:
    def __init__(self, run_id: str, log_path: Path, level: str = "INFO") -> None:
        self.run_id = run_id
        self.log_path = log_path
        self.level = level if level in LOG_LEVEL_ORDER else "INFO"
        ensure_dir(log_path.parent)
        self._fh = log_path.open("w", encoding="utf-8")

    def event(self, stage: str, event: str, level: str = "INFO", **payload: Any) -> None:
        safe_level = level if level in LOG_LEVEL_ORDER else "INFO"
        if LOG_LEVEL_ORDER[safe_level] < LOG_LEVEL_ORDER[self.level]:
            return
        record = {
            "ts": utc_now(),
            "run_id": self.run_id,
            "level": safe_level,
            "stage": stage,
            "event": event,
            "payload": payload,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPAR-Hate Phase 2 Perspectives generation")
    default_config = str((Path(__file__).resolve().parents[1] / "configs" / "phase2_en_api.json"))
    parser.add_argument("--config", default=default_config, help=f"Config path. Default: {default_config}")
    parser.add_argument("--run-id", default=None, help="Optional run id")
    parser.add_argument("--mode", choices=["heuristic", "qwen_local", "vllm_local", "api", "stub"], default=None)
    parser.add_argument("--api-profile", default=None, help="Override inference.api.profile, e.g. gpt or deepseek")
    parser.add_argument("--input-jsonl", default=None, help="Override phase1 thought-group JSONL")
    parser.add_argument("--output-root", default=None, help="Override output root")
    parser.add_argument("--model-path", default=None, help="Override local generator model path")
    parser.add_argument("--prompt-path", default=None, help="Override role prompt template path")
    parser.add_argument("--lexicon-path", default=None, help="Override zh annotated lexicon path")
    parser.add_argument("--roles", default=None, help="Comma-separated roles")
    parser.add_argument("--max-samples", type=int, default=None, help="Process at most N input groups")
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Process at most N groups for each language+split bucket (e.g. zh_train, zh_test, en_train, en_test)",
    )
    parser.add_argument("--splits", default=None, help="Comma-separated splits, e.g. train,dev,test")
    parser.add_argument("--languages", default=None, help="Comma-separated languages, e.g. zh,en")
    parser.add_argument(
        "--enable-thinking",
        choices=["true", "false"],
        default=None,
        help="Override inference.enable_thinking for Qwen3",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override generation max_new_tokens")
    parser.add_argument("--temperature", type=float, default=None, help="Override generation temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Override generation top_p")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--max-parse-retries", type=int, default=None, help="Override parse retries")
    parser.add_argument("--self-check-rounds", type=int, default=None, help="Override self-check rounds")
    parser.add_argument("--micro-batch-size", type=int, default=None, help="Override physical micro batch size")
    parser.add_argument("--dry-run", action="store_true", help="Skip candidate JSONL writes")
    parser.add_argument("--strict", action="store_true", help="Fail when quality gate fails")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None)
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for YAML config. Use JSON config or install pyyaml.") from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected mapping config in {config_path}, got {type(loaded)}")
        return loaded
    raise ValueError(f"Unsupported config format: {config_path}")


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    override_map = [
        (args.mode, ["phase2", "mode"]),
        (args.api_profile, ["inference", "api_profile"]),
        (args.input_jsonl, ["paths", "input_jsonl"]),
        (args.output_root, ["paths", "output_root"]),
        (args.model_path, ["paths", "generator_model_path"]),
        (args.prompt_path, ["paths", "role_prompt_path"]),
        (args.lexicon_path, ["paths", "zh_lexicon_path"]),
        (args.max_samples, ["runtime", "max_samples"]),
        (args.max_samples_per_split, ["runtime", "max_samples_per_split"]),
        (args.max_new_tokens, ["inference", "max_new_tokens"]),
        (args.temperature, ["inference", "temperature"]),
        (args.top_p, ["inference", "top_p"]),
        (args.seed, ["seed"]),
        (args.max_parse_retries, ["phase2", "max_parse_retries"]),
        (args.self_check_rounds, ["phase2", "max_self_check_rounds"]),
        (args.micro_batch_size, ["phase2", "micro_batch_size"]),
        (args.log_level, ["logging", "level"]),
    ]
    for value, keys in override_map:
        if value is not None:
            set_nested(cfg, keys, value)

    mode_to_backend = {
        "qwen_local": "transformers",
        "vllm_local": "vllm",
        "api": "api",
    }
    if args.mode in mode_to_backend:
        set_nested(cfg, ["inference", "backend"], mode_to_backend[str(args.mode)])

    if args.roles:
        roles = [x.strip() for x in args.roles.split(",") if x.strip()]
        set_nested(cfg, ["phase2", "roles"], roles)
    if args.splits:
        splits = [x.strip() for x in args.splits.split(",") if x.strip()]
        set_nested(cfg, ["runtime", "default_splits"], splits)
    if args.languages:
        languages = [x.strip().lower() for x in args.languages.split(",") if x.strip()]
        set_nested(cfg, ["runtime", "default_languages"], languages)
    if args.enable_thinking is not None:
        set_nested(cfg, ["inference", "enable_thinking"], args.enable_thinking.lower() == "true")
    if args.strict:
        set_nested(cfg, ["quality", "strict"], True)


def build_paths(args: argparse.Namespace, cfg: Dict[str, Any], workspace_root: Path) -> Paths:
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg_paths = cfg["paths"]
    inference_cfg = cfg.get("inference", {}) if isinstance(cfg.get("inference"), dict) else {}
    api_profiles = inference_cfg.get("api_profiles") if isinstance(inference_cfg.get("api_profiles"), dict) else None
    phase2_cfg = cfg.get("phase2", {}) if isinstance(cfg.get("phase2"), dict) else {}
    mode = str(phase2_cfg.get("mode", "qwen_local")).strip().lower()

    input_jsonl_raw = str(cfg_paths["input_jsonl"])
    output_root_raw = str(cfg_paths["output_root"])
    if mode == "api":
        namespace = derive_api_output_namespace(inference_cfg, api_profiles=api_profiles)
        input_jsonl_raw = str(rewrite_output_namespace_path_text(input_jsonl_raw, namespace))
        output_root_raw = str(rewrite_output_namespace_path_text(output_root_raw, namespace))

    input_jsonl = resolve_path(workspace_root, config_dir, input_jsonl_raw)
    output_root = resolve_path(workspace_root, config_dir, output_root_raw)
    model_root = resolve_path(workspace_root, config_dir, cfg_paths["model_root"])
    configured_model_path = resolve_path(workspace_root, config_dir, cfg_paths["generator_model_path"])
    generator_model_path = configured_model_path
    if mode != "api":
        generator_model_path = resolve_existing_model_path(
            configured_model_path=configured_model_path,
            workspace_root=workspace_root,
            config_dir=config_dir,
            model_root=model_root,
            raw_candidates=cfg_paths.get("generator_model_candidates"),
        )
    prompt_raw = cfg_paths.get("role_prompt_path")
    prompt_path = resolve_path(workspace_root, config_dir, prompt_raw) if isinstance(prompt_raw, str) and prompt_raw else None
    role_prompt_map = resolve_role_prompt_map(workspace_root, config_dir, cfg_paths.get("role_prompt_map"))
    run_id = args.run_id or f"phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = output_root / run_id
    return Paths(
        workspace_root=workspace_root,
        config_path=config_path,
        input_jsonl=input_jsonl,
        output_root=output_root,
        run_root=run_root,
        model_root=model_root,
        generator_model_path=generator_model_path,
        role_prompt_path=prompt_path,
        role_prompt_map=role_prompt_map,
    )


def parse_allowed_splits(cfg: Dict[str, Any]) -> List[str]:
    runtime_cfg = cfg.get("runtime", {})
    splits = runtime_cfg.get("default_splits", ["train", "dev", "test"])
    if not isinstance(splits, list):
        return ["train", "dev", "test"]
    cleaned = [str(x).strip() for x in splits if str(x).strip()]
    return cleaned or ["train", "dev", "test"]


def parse_allowed_languages(cfg: Dict[str, Any]) -> List[str]:
    runtime_cfg = cfg.get("runtime", {})
    languages = runtime_cfg.get("default_languages", ["zh", "en"])
    if not isinstance(languages, list):
        return ["zh", "en"]
    cleaned = [str(x).strip().lower() for x in languages if str(x).strip()]
    return cleaned or ["zh", "en"]


def limit_rows_per_language_split(rows: Sequence[Dict[str, Any]], max_per_split: int) -> List[Dict[str, Any]]:
    if max_per_split <= 0:
        return list(rows)

    by_bucket: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    spillover: List[Dict[str, Any]] = []
    valid_buckets = set(LANGUAGE_SPLIT_ORDER)

    for row in rows:
        bucket = (str(row.get("language") or "en"), str(row.get("split") or "train"))
        if bucket in valid_buckets:
            by_bucket[bucket].append(row)
        else:
            spillover.append(row)

    limited: List[Dict[str, Any]] = []
    for bucket in LANGUAGE_SPLIT_ORDER:
        limited.extend(by_bucket.get(bucket, [])[:max_per_split])
    limited.extend(spillover)
    return limited


def parse_role_specs(phase2_cfg: Dict[str, Any]) -> Dict[str, str]:
    roles_raw = phase2_cfg.get("roles", ["victim", "moderator", "cultural_bystander"])
    if not isinstance(roles_raw, list):
        roles_raw = ["victim", "moderator", "cultural_bystander"]
    roles = [str(r).strip() for r in roles_raw if str(r).strip()]
    roles = dedupe_preserve_order(roles, case_insensitive=False)
    if not roles:
        roles = ["victim", "moderator", "cultural_bystander"]
    compress = parse_bool(phase2_cfg.get("compress_cultural_into_victim", False), default=False)
    specs: Dict[str, str] = {}
    for role in roles:
        charter = ROLE_CHARTERS.get(role, ROLE_CHARTERS["victim"])
        if compress and role == "victim":
            charter = charter + " " + CULTURAL_CHECKLIST_APPENDIX
        specs[role] = charter
    return specs


def normalize_harmful_direction(raw_direction: Any, raw_harmful: Any = None) -> str:
    direction = normalize_string(raw_direction)
    if direction:
        lowered = direction.lower().strip()
        if lowered in VALID_HARMFUL_DIRECTION:
            return lowered
    if raw_harmful in {1, "1", True, "true", "True"}:
        return "harmful"
    if raw_harmful in {0, "0", False, "false", "False"}:
        return "non-harmful"
    return "unknown"


def direction_to_harmful_binary(direction: str) -> Optional[int]:
    if direction == "harmful":
        return 1
    if direction == "non-harmful":
        return 0
    return None


def truncate_text(value: Optional[str], max_chars: int = 4000) -> Optional[str]:
    if value is None:
        return None
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def parse_sample_id_from_group_id(group_id: Any) -> Optional[str]:
    gid = normalize_string(group_id)
    if gid is None or "_g" not in gid:
        return None
    sample_id, suffix = gid.rsplit("_g", 1)
    if not sample_id or not suffix.isdigit():
        return None
    return sample_id


def parse_local_group_id(group_id: Any, default: int = 1) -> int:
    gid = normalize_string(group_id)
    if gid is None or "_g" not in gid:
        return default
    _, suffix = gid.rsplit("_g", 1)
    if suffix.isdigit() and int(suffix) > 0:
        return int(suffix)
    return default


def default_phase1_language_split_inputs(workspace_root: Path) -> Dict[str, Path]:
    base = workspace_root / "SPAR/outputs/main/phase1"
    return {k: base / v for k, v in PHASE1_OUTPUT_FILENAME_MAP.items()}


def resolve_phase2_language_split_inputs(paths: Paths, cfg: Dict[str, Any]) -> Dict[str, Path]:
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
    inference_cfg = cfg.get("inference", {}) if isinstance(cfg.get("inference"), dict) else {}
    config_dir = paths.config_path.parent
    backend = str(inference_cfg.get("backend", "transformers")).strip().lower()
    api_profiles = inference_cfg.get("api_profiles") if isinstance(inference_cfg.get("api_profiles"), dict) else None
    raw_map = runtime_cfg.get("phase1_language_split_inputs")
    if isinstance(raw_map, dict):
        resolved: Dict[str, Path] = {}
        for key, raw in raw_map.items():
            if key not in PHASE1_OUTPUT_FILENAME_MAP or not isinstance(raw, str) or not raw.strip():
                continue
            effective_raw = raw.strip()
            if backend == "api":
                namespace = derive_api_output_namespace(inference_cfg, api_profiles=api_profiles)
                effective_raw = str(rewrite_output_namespace_path_text(effective_raw, namespace))
            resolved[key] = resolve_path(paths.workspace_root, config_dir, effective_raw)
        return resolved
    resolved = default_phase1_language_split_inputs(paths.workspace_root)
    if backend == "api":
        namespace = derive_api_output_namespace(inference_cfg, api_profiles=api_profiles)
        for key, path in list(resolved.items()):
            resolved[key] = Path(str(rewrite_output_namespace_path_text(str(path), namespace)))
    return resolved


def normalize_row_for_language_split(row: Dict[str, Any], language: str, split: str) -> Dict[str, Any]:
    group_text = normalize_string(row.get("group_text")) or normalize_string(row.get("text")) or ""
    focus_text = normalize_string(row.get("focus_text"))
    if focus_text is None:
        focus_text = f"<focus>{group_text}</focus>" if group_text else "<focus></focus>"
    group_id = normalize_string(row.get("group_id"))
    sample_id = normalize_string(row.get("sample_id"))
    if group_id is None:
        base_sample = sample_id or normalize_string(row.get("id")) or normalize_string(row.get("original_id"))
        if base_sample is None:
            base_sample = make_sha1(language, split, group_text)[:16]
        local_group_id = row.get("local_group_id")
        try:
            local_group_id = int(local_group_id)
        except (TypeError, ValueError):
            local_group_id = 1
        local_group_id = max(1, int(local_group_id))
        group_id = f"{base_sample}_g{local_group_id}"

    if sample_id is None:
        sample_id = parse_sample_id_from_group_id(group_id) or normalize_string(row.get("id")) or group_id

    original_id = normalize_string(row.get("original_id")) or sample_id
    source = normalize_string(row.get("source")) or "unknown"
    canonical_target = normalize_string(row.get("canonical_target")) or "IMPLICIT_TARGET"
    escalation_flag = parse_bool(row.get("escalation_flag"), default=(canonical_target == "IMPLICIT_TARGET"))
    local_group_id = row.get("local_group_id")
    try:
        local_group_id = int(local_group_id)
    except (TypeError, ValueError):
        local_group_id = parse_local_group_id(group_id, default=1)
    local_group_id = max(1, int(local_group_id))

    return {
        "group_id": group_id,
        "sample_id": sample_id,
        "original_id": original_id,
        "source": source,
        "language": language,
        "split": split,
        "local_group_id": local_group_id,
        "focus_text": focus_text,
        "group_text": group_text,
        "canonical_target": canonical_target,
        "escalation_flag": bool(escalation_flag),
    }


def load_phase1_rows_by_language_split(
    paths: Paths,
    cfg: Dict[str, Any],
    logger: EventLogger,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
    require_inputs = parse_bool(runtime_cfg.get("require_phase1_language_split_inputs", True), default=True)

    input_map = resolve_phase2_language_split_inputs(paths=paths, cfg=cfg)
    rows: List[Dict[str, Any]] = []
    loaded_paths: Dict[str, str] = {}

    for key, input_path in input_map.items():
        if key not in PHASE1_OUTPUT_FILENAME_MAP:
            continue
        if not input_path.exists():
            if require_inputs:
                raise FileNotFoundError(f"Phase1 split input missing: {key} -> {input_path}")
            logger.event("load", "phase1_split_missing", level="WARNING", key=key, path=str(input_path))
            continue

        raw_rows = read_jsonl(input_path)
        language, split = key.split("_", 1)
        normalized = [normalize_row_for_language_split(r, language=language, split=split) for r in raw_rows]
        rows.extend(normalized)
        loaded_paths[key] = str(input_path)
        logger.event("load", "phase1_split_loaded", key=key, path=str(input_path), rows=len(normalized))

    if not rows and paths.input_jsonl.exists():
        fallback_rows = read_jsonl(paths.input_jsonl)
        rows = [
            normalize_row_for_language_split(
                r,
                language=str(r.get("language") or detect_language(normalize_string(r.get("group_text")) or "")),
                split=str(r.get("split") or "train"),
            )
            for r in fallback_rows
        ]
        loaded_paths["fallback_input_jsonl"] = str(paths.input_jsonl)
        logger.event("load", "fallback_input_jsonl_loaded", rows=len(rows), path=str(paths.input_jsonl))

    ordered_rows: List[Dict[str, Any]] = []
    for lang, split in LANGUAGE_SPLIT_ORDER:
        ordered_rows.extend([r for r in rows if r.get("language") == lang and r.get("split") == split])
    spillover = [r for r in rows if r not in ordered_rows]
    ordered_rows.extend(spillover)
    return ordered_rows, loaded_paths


def resolve_zh_lexicon_path(paths: Paths, cfg: Dict[str, Any]) -> Path:
    cfg_paths = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    config_dir = paths.config_path.parent
    raw = cfg_paths.get("zh_lexicon_path")
    if isinstance(raw, str) and raw.strip():
        return resolve_path(paths.workspace_root, config_dir, raw.strip())
    return paths.workspace_root / "Datasets/STATE-ToxiCN-main/data/annotated lexicon.json"


def load_zh_annotated_lexicon(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    terms = obj.get("terms") if isinstance(obj, dict) else None
    if not isinstance(terms, list):
        return []

    cleaned: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for item in terms:
        if not isinstance(item, dict):
            continue
        term = normalize_string(item.get("term"))
        if term is None:
            continue
        if term in seen:
            continue
        seen.add(term)
        cleaned.append(
            {
                "term": term,
                "category": normalize_string(item.get("category")) or "unknown",
                "definition": normalize_string(item.get("definition")) or "",
            }
        )
    cleaned.sort(key=lambda x: len(x["term"]), reverse=True)
    return cleaned


\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\


try:
    import jieba
except ImportError:
    jieba = None


try:
    import jieba
except ImportError:
    jieba = None

def lookup_zh_lexicon_hits(row: Dict[str, Any], lexicon_terms: Sequence[Dict[str, str]], max_hits: int = 6) -> List[Dict[str, str]]:
    if str(row.get("language")) != "zh" or not lexicon_terms:
        return []

    raw_focus = normalize_string(row.get("focus_text")) or ""
    full_text = re.sub(r"</?focus>", "", raw_focus, flags=re.IGNORECASE).strip()


    boundaries: Set[int] = set()
    if jieba and full_text:

        for word, start, end in jieba.tokenize(full_text):
            boundaries.add(start)
            boundaries.add(end)
    else:
        boundaries.add(0)
        boundaries.add(len(full_text))

    canonical_target = normalize_string(row.get("canonical_target")) or ""
    if canonical_target == "IMPLICIT_TARGET":
        canonical_target = ""

    max_hits = max(1, int(max_hits or 6))
    hits: List[Dict[str, str]] = []
    seen: Set[str] = set()

    def _add(term_item: Dict[str, str]) -> None:
        t = term_item.get("term") or ""
        if not t or t in seen:
            return
        seen.add(t)
        hit = dict(term_item)
        hits.append(hit)

    for item in lexicon_terms:
        term = item.get("term") or ""
        if not term:
            continue

        hit_flag = False


        if canonical_target and term == canonical_target:
            hit_flag = True


        if not hit_flag and full_text and term in full_text:
            if jieba:

                start_idx = full_text.find(term)
                while start_idx != -1:
                    end_idx = start_idx + len(term)


                    if start_idx in boundaries and end_idx in boundaries:
                        hit_flag = True
                        break

                    start_idx = full_text.find(term, start_idx + 1)
            else:

                if len(full_text) <= 5:
                    hit_flag = True

        if hit_flag:
            _add(item)
            if len(hits) >= max_hits:
                break

    return hits


def build_lexicon_knowledge_block(hits: Sequence[Dict[str, str]]) -> str:
    if not hits:
        return ""
    lines = ["【相关文化与黑话背景参考】"]
    for hit in hits:
        term = str(hit.get("term") or "").strip()
        category = str(hit.get("category") or "unknown").strip()
        definition = str(hit.get("definition") or "").strip() or "无释义"
        lines.append(f" - 【{term}】 (类别: {category}): {definition}")
    return "\n".join(lines)


def select_role_prompt_template(
    language: str,
    role_name: str,
    role_prompt_overrides: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    if role_prompt_overrides:
        role_map = role_prompt_overrides.get(role_name)
        if isinstance(role_map, dict):
            lang_key = normalize_prompt_language_key(language)
            if lang_key in role_map:
                return role_map[lang_key]
            if "default" in role_map:
                return role_map["default"]

    if language == "zh":
        return {
            "victim": ZH_VICTIM_PROMPT,
            "moderator": ZH_MODERATOR_PROMPT,
            "cultural_bystander": ZH_BYSTANDER_PROMPT,
        }.get(role_name, ZH_VICTIM_PROMPT)
    return {
        "victim": EN_VICTIM_PROMPT,
        "moderator": EN_MODERATOR_PROMPT,
        "cultural_bystander": EN_BYSTANDER_PROMPT,
    }.get(role_name, EN_VICTIM_PROMPT)


def render_role_prompt_text(
    template: str,
    focus_text: str,
    group_text: str,
    lexicon_knowledge: str,
    retry_note: Optional[str],
    target_hint: Optional[str] = None,
    language: str = "en",
) -> str:
    rendered = template
    rendered = rendered.replace("{{GROUP_TEXT}}", group_text)
    rendered = rendered.replace("{{FOCUS_TEXT}}", focus_text)
    rendered = rendered.replace("{{TARGET_HINT}}", target_hint or "")
    rendered = rendered.replace("{{LEXICON_KNOWLEDGE}}", lexicon_knowledge.strip())
    if retry_note:
        normalized_language = str(language or "en").strip().lower()
        if normalized_language == "zh":
            rendered += (
                "\n\n上一次输出不是合法 JSON 数组。"
                f"错误：{retry_note}\n"
                "现在不要解释，不要分析，不要重复规则。"
                "直接输出最终 JSON 数组。"
                "每个 `rationale` 必须单行，`argument` 不能为空；如果长度不够，宁可更短，也不要截断。"
            )
        else:
            rendered += (
                "\n\nPrevious output was not a valid JSON array. "
                f"Error: {retry_note}\n"
                "Do not explain, do not restate the rules, and do not add any extra text. "
                "Output the final JSON array directly. Keep each `rationale` to one line, never leave `argument` empty, "
                "and shorten the answer if needed instead of producing truncated JSON."
            )
    return rendered


def build_generation_system_message(
    enable_thinking: bool,
    force_no_think: bool = False,
    model_name: Optional[str] = None,
) -> str:
    effective_enable_thinking = bool(enable_thinking and not force_no_think)
    supports_soft_switch = not is_qwen35_family_model(model_name)
    parts: List[str] = []
    if enable_thinking:
        if supports_soft_switch:
            if force_no_think:
                parts.append("/no_think")
            else:
                parts.append("/think")
    if effective_enable_thinking:
        parts.append(
            "If you think before answering, keep the reasoning extremely short: at most 2 brief sentences, then immediately output the final JSON array."
        )
    else:
        parts.append("Return the final JSON array directly without any reasoning text.")
    parts.append("Every object must contain a non-empty `argument` exact substring from inside `<focus>`; for harmless items, choose a short exact substring that supports the harmless judgment.")
    parts.append("Keep every `rationale` on a single line. Do not output raw newline characters inside JSON string values.")
    parts.append("Return strict JSON array only. Do not include markdown, code fences, or any text outside the JSON array.")
    return "\n".join(parts)


def is_qwen35_family_model(model_name: Any) -> bool:
    text = str(model_name or "").strip().lower()
    if not text:
        return False
    compact = re.sub(r"[^a-z0-9]+", "", text)
    return "qwen35" in compact


def extract_json_array_candidate(text: str) -> List[Any]:
    raw_stripped = text.strip()
    if not raw_stripped:
        raise ValueError("Empty model output")


    stripped = re.sub(r"<think>[\s\S]*?</think>", "", raw_stripped, flags=re.IGNORECASE).strip()
    stripped = re.sub(r"<think>[\s\S]*", "", stripped, flags=re.IGNORECASE).strip()

    candidates: List[str] = []
    sources = [stripped, raw_stripped]
    lower = raw_stripped.lower()
    end_think_idx = lower.rfind("</think>")
    if end_think_idx >= 0:
        tail = raw_stripped[end_think_idx + len("</think>") :].strip()
        if tail:
            sources.insert(0, tail)
    code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw_stripped, flags=re.IGNORECASE)
    sources.extend(block.strip() for block in code_blocks if block.strip())

    for source in dedupe_preserve_order([s for s in sources if s], case_insensitive=False):
        matches = ARRAY_BLOCK_RE.findall(source)
        for item in matches:
            cand = item.strip()
            if cand:
                candidates.append(cand)
        list_start = source.find("[")
        list_end = source.rfind("]")
        if list_start >= 0 and list_end > list_start:
            candidates.append(source[list_start : list_end + 1])

    deduped = dedupe_preserve_order(candidates, case_insensitive=False)
    last_error: Optional[Exception] = None
    for cand in deduped:
        try:
            parsed = json.loads(sanitize_json_string_control_chars(cand))
        except Exception as exc:
            last_error = exc
            recovered = salvage_json_objects_from_array_text(cand)
            if recovered:
                return recovered
            continue
        if isinstance(parsed, list):
            return parsed

    for source in dedupe_preserve_order([s for s in sources if s], case_insensitive=False):
        recovered = salvage_json_objects_from_array_text(source)
        if recovered:
            return recovered

    raise ValueError(f"No parseable JSON array in model output: {last_error}")


def normalize_hate_label(value: Any) -> Optional[int]:
    if value in {1, "1", True, "true", "True", "yes", "YES"}:
        return 1
    if value in {0, "0", False, "false", "False", "no", "NO"}:
        return 0
    return None


def normalize_prediction_array_by_language(
    parsed: Any,
    language: str,
    focus_text: str = "",
    group_text: str = "",
) -> List[Dict[str, Any]]:
    if not isinstance(parsed, list):
        raise ValueError("Model output must be JSON array")

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Array item at index {idx} is not an object")

        rationale = normalize_string(item.get("rationale"))
        target = normalize_string(item.get("target"))
        if language == "zh":
            hateful = normalize_hate_label(item.get("hateful"))
            argument = normalize_string(item.get("argument")) or fallback_argument_for_harmless_item(
                target=target,
                focus_text=focus_text,
                group_text=group_text,
                language=language,
            ) if hateful == 0 else normalize_string(item.get("argument"))
        else:
            harmful = normalize_hate_label(item.get("harmful"))
            argument = normalize_string(item.get("argument")) or fallback_argument_for_harmless_item(
                target=target,
                focus_text=focus_text,
                group_text=group_text,
                language=language,
            ) if harmful == 0 else normalize_string(item.get("argument"))

        if rationale is None or target is None or argument is None:
            raise ValueError(f"Missing rationale/target/argument at index {idx}")

        if language == "zh":
            expected_keys = {"rationale", "target", "argument", "group", "hateful"}
            if set(item.keys()) != expected_keys:
                raise ValueError(f"ZH schema must contain exact keys {sorted(expected_keys)} at index {idx}")
            group = canonicalize_zh_group_value(item.get("group"), hateful)
            normalized.append(
                {
                    "rationale": rationale,
                    "target": target,
                    "argument": argument,
                    "group": group,
                    "hateful": hateful,
                }
            )
            continue

        expected_keys = {"rationale", "target", "argument", "harmful"}
        if set(item.keys()) != expected_keys:
            raise ValueError(f"EN schema must contain exact keys {sorted(expected_keys)} at index {idx}")
        normalized.append(
            {
                "rationale": rationale,
                "target": target,
                "argument": argument,
                "harmful": harmful,
            }
        )

    return postprocess_prediction_array_by_language(
        predictions=normalized,
        language=language,
        focus_text=focus_text,
        group_text=group_text,
    )


def build_heuristic_role_array_prediction(row: Dict[str, Any], role_name: str) -> List[Dict[str, Any]]:
    group_text = normalize_string(row.get("group_text")) or ""
    rationale = f"Heuristic fallback for role={role_name}; text-only estimate."
    if str(row.get("language")) == "zh":
        return [
            {
                "rationale": rationale,
                "target": "IMPLICIT_TARGET",
                "argument": group_text,
                "group": "unknown",
                "hateful": None,
            }
        ]
    return [
        {
            "rationale": rationale,
            "target": "IMPLICIT_TARGET",
            "argument": group_text,
            "harmful": None,
        }
    ]


def prefer_4090_device_order(force_4090_order: bool = False) -> Optional[str]:
    existing = os.getenv("CUDA_VISIBLE_DEVICES")
    if existing and not force_4090_order:
        return existing
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return existing
    rows: List[Tuple[int, str]] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) != 2:
            continue
        try:
            rows.append((int(parts[0]), parts[1]))
        except ValueError:
            continue
    if not rows:
        return existing
    rows.sort(key=lambda x: (0 if "4090" in x[1] else 1, x[0]))
    ordered = ",".join(str(idx) for idx, _ in rows)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ordered
    return ordered


def build_max_memory_map(torch_module: Any, preferred_ratio: float, other_ratio: float) -> Optional[Dict[int, str]]:
    if not torch_module.cuda.is_available():
        return None
    max_memory: Dict[int, str] = {}
    for idx in range(torch_module.cuda.device_count()):
        props = torch_module.cuda.get_device_properties(idx)
        name = torch_module.cuda.get_device_name(idx)
        total_gib = props.total_memory / (1024 ** 3)
        ratio = preferred_ratio if "4090" in name else other_ratio
        budget = max(4, int(total_gib * ratio))
        max_memory[idx] = f"{budget}GiB"
    return max_memory or None


def resolve_vllm_dtype(dtype_name: str) -> str:
    normalized = str(dtype_name or "auto").strip().lower()
    if normalized in {"float16", "half", "fp16"}:
        return "float16"
    if normalized in {"bfloat16", "bf16"}:
        return "bfloat16"
    if normalized in {"float32", "fp32"}:
        return "float32"
    return "auto"


def ensure_tokenizer_vllm_compat(tokenizer: Any) -> Any:
    """Patch tokenizer attributes expected by vLLM but missing in some HF Qwen tokenizers."""
    if not hasattr(tokenizer, "all_special_tokens_extended"):
        try:
            tokenizer.all_special_tokens_extended = list(getattr(tokenizer, "all_special_tokens", []))
        except Exception:
            tokenizer.all_special_tokens_extended = []
    return tokenizer


def patch_transformers_tokenizer_classes_for_vllm() -> None:
    """Patch tokenizer classes so vLLM-created tokenizer instances stay compatible."""
    candidates: List[Any] = []
    try:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend

        candidates.append(TokenizersBackend)
    except Exception:
        pass
    try:
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

        candidates.append(Qwen2Tokenizer)
    except Exception:
        pass
    try:
        from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

        candidates.append(Qwen2TokenizerFast)
    except Exception:
        pass

    for tokenizer_cls in candidates:
        if tokenizer_cls is None or hasattr(tokenizer_cls, "all_special_tokens_extended"):
            continue

        def _all_special_tokens_extended(self: Any) -> List[str]:
            return list(getattr(self, "all_special_tokens", []))

        setattr(tokenizer_cls, "all_special_tokens_extended", property(_all_special_tokens_extended))


\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\


class LocalRoleLLM:
    def __init__(self, model_path: Path, inference_cfg: Dict[str, Any], prompt_template: str) -> None:
        self.model_path = model_path
        self.inference_cfg = inference_cfg
        self.prompt_template = prompt_template
        self.backend = str(self.inference_cfg.get("backend", "transformers")).strip().lower()
        self._ready = False
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._sampling_params_cls = None
        self._api_client: Optional[ApiTextGenerationClient] = None
        self.cuda_available = False
        self.visible_cuda_devices: List[str] = []
        self.model_device: Optional[str] = None
        self.enable_thinking = parse_bool(self.inference_cfg.get("enable_thinking", True), default=True)
        self.supports_soft_think_switch = not is_qwen35_family_model(self.model_path.name)

    def initialize(self) -> None:
        if self._ready:
            return
        if self.backend == "api":
            self._initialize_api()
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Local model path does not exist: {self.model_path}. "
                "Set --model-path explicitly or update paths.generator_model_path / paths.generator_model_candidates in config."
            )

        if parse_bool(self.inference_cfg.get("prefer_4090_order", True), default=True):
            prefer_4090_device_order(
                force_4090_order=parse_bool(self.inference_cfg.get("force_4090_order", False), default=False)
            )

        if self.backend == "vllm":
            self._initialize_vllm()
            return

        if self.backend != "vllm" and is_qwen35_family_model(self.model_path.name):
            raise RuntimeError(
                "Qwen3.5 local transformers fallback is disabled for this path because it is not continuation-safe "
                "in the current environment. Use backend='vllm' and clear fallback_backend_on_init_failure for "
                "Qwen3.5 continuation runs."
            )

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("qwen_local mode requires torch and transformers") from exc

        require_cuda = parse_bool(self.inference_cfg.get("require_cuda", True), default=True)
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for qwen_local mode, but torch.cuda.is_available() is False")

        use_cuda = bool(torch.cuda.is_available())
        self.cuda_available = use_cuda
        self.visible_cuda_devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if use_cuda else []

        default_dtype = "bfloat16" if use_cuda else "float32"
        dtype_name = str(self.inference_cfg.get("dtype", default_dtype)).lower()
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype_value = dtype_map.get(dtype_name, torch.float16 if use_cuda else torch.float32)

        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
        tokenizer = ensure_tokenizer_vllm_compat(tokenizer)
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if use_cuda:
            load_kwargs["device_map"] = self.inference_cfg.get("device_map", "balanced")
            preferred_ratio = safe_float(self.inference_cfg.get("preferred_gpu_mem_ratio", 0.97), default=0.97)
            other_ratio = safe_float(self.inference_cfg.get("other_gpu_mem_ratio", 0.25), default=0.25)
            if torch.cuda.device_count() >= 2 and str(load_kwargs.get("device_map", "")).startswith("balanced"):
                min_multi_gpu_ratio = safe_float(self.inference_cfg.get("min_multi_gpu_mem_ratio", 0.85), default=0.85)
                other_ratio = max(other_ratio, min_multi_gpu_ratio)
            max_memory = build_max_memory_map(torch, preferred_ratio=preferred_ratio, other_ratio=other_ratio)
            if max_memory:
                load_kwargs["max_memory"] = max_memory
            use_8bit = parse_bool(self.inference_cfg.get("load_in_8bit_prefer", False), default=False)
            quantized = False
            if use_8bit:
                try:
                    from transformers import BitsAndBytesConfig

                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True,
                    )
                    quantized = True
                except Exception:
                    quantized = False
            if not quantized:
                load_kwargs["torch_dtype"] = dtype_value
        else:
            load_kwargs["torch_dtype"] = torch.float32

        try:
            model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **load_kwargs)
        except ImportError as exc:
            if "quantization_config" in load_kwargs and "bitsandbytes" in str(exc).lower():
                load_kwargs.pop("quantization_config", None)
                load_kwargs["torch_dtype"] = dtype_value if use_cuda else torch.float32
                model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **load_kwargs)
            else:
                raise
        model.eval()

        if hasattr(model, "device"):
            self.model_device = str(getattr(model, "device"))
        else:
            try:
                self.model_device = str(next(model.parameters()).device)
            except StopIteration:
                self.model_device = "unknown"

        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._ready = True

    def _initialize_vllm(self) -> None:
        try:
            import torch
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("vllm backend requires vllm, torch, and transformers") from exc

        require_cuda = parse_bool(self.inference_cfg.get("require_cuda", True), default=True)
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for vllm backend, but torch.cuda.is_available() is False")

        patch_transformers_tokenizer_classes_for_vllm()
        self.cuda_available = bool(torch.cuda.is_available())
        self.visible_cuda_devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if self.cuda_available else []
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
        tokenizer = ensure_tokenizer_vllm_compat(tokenizer)
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        dtype_name = resolve_vllm_dtype(str(self.inference_cfg.get("dtype", "auto")))
        tensor_parallel_size = int(max(1, self.inference_cfg.get("tensor_parallel_size", 1)))
        gpu_memory_utilization = safe_float(self.inference_cfg.get("gpu_memory_utilization", 0.9), default=0.9)
        max_model_len = int(max(1024, self.inference_cfg.get("max_model_len", 8192)))
        max_num_batched_tokens = int(max_model_len if self.inference_cfg.get("max_num_batched_tokens") is None else self.inference_cfg.get("max_num_batched_tokens"))
        enforce_eager = parse_bool(self.inference_cfg.get("enforce_eager", False), default=False)
        trust_remote_code = parse_bool(self.inference_cfg.get("trust_remote_code", True), default=True)
        disable_custom_all_reduce = parse_bool(self.inference_cfg.get("disable_custom_all_reduce", False), default=False)
        enable_chunked_prefill = parse_bool(self.inference_cfg.get("enable_chunked_prefill", True), default=True)

        llm_kwargs: Dict[str, Any] = {
            "model": str(self.model_path),
            "tokenizer": str(self.model_path),
            "dtype": dtype_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "enforce_eager": enforce_eager,
            "trust_remote_code": trust_remote_code,
            "disable_custom_all_reduce": disable_custom_all_reduce,
            "enable_chunked_prefill": enable_chunked_prefill,
        }
        block_size = self.inference_cfg.get("block_size")
        if block_size is not None:
            llm_kwargs["block_size"] = int(block_size)
        max_num_seqs = self.inference_cfg.get("max_num_seqs")
        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = int(max_num_seqs)
        swap_space = self.inference_cfg.get("swap_space")
        if swap_space is not None:
            llm_kwargs["swap_space"] = float(swap_space)
        seed = self.inference_cfg.get("seed")
        if seed is not None:
            llm_kwargs["seed"] = int(seed)

        model = LLM(**llm_kwargs)
        self.model_device = f"vllm-tp{tensor_parallel_size}"
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._sampling_params_cls = SamplingParams
        self._ready = True

    def _initialize_api(self) -> None:
        api_profiles = self.inference_cfg.get("api_profiles") if isinstance(self.inference_cfg.get("api_profiles"), dict) else None
        cfg = resolve_api_backend_config(self.inference_cfg, api_profiles=api_profiles)
        self._api_client = ApiTextGenerationClient(cfg)
        self._tokenizer = None
        self._model = self._api_client
        self._torch = None
        self._sampling_params_cls = None
        self.cuda_available = False
        self.visible_cuda_devices = []
        self.model_device = f"api:{cfg.provider}:{cfg.api_format}:{cfg.model}"
        self._ready = True

    def _messages_to_prompt(
        self,
        messages: Sequence[Dict[str, str]],
        enable_thinking_override: Optional[bool] = None,
    ) -> str:
        rendered_messages = [dict(m) for m in messages]

        effective_enable_thinking = self.enable_thinking
        if enable_thinking_override is not None:
            effective_enable_thinking = bool(enable_thinking_override)

        if self.backend == "api":
            if not effective_enable_thinking and self.supports_soft_think_switch:
                for msg in rendered_messages:
                    if str(msg.get("role") or "") == "system":
                        content = str(msg.get("content") or "").strip()
                        if "/no_think" not in content:
                            msg["content"] = f"/no_think\n{content}" if content else "/no_think"
                        break
            return render_messages_as_text(rendered_messages)

        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized")
        tokenizer = self._tokenizer

        if not effective_enable_thinking and self.supports_soft_think_switch:
            for msg in rendered_messages:
                if str(msg.get("role") or "") == "system":
                    content = str(msg.get("content") or "").strip()
                    if "/no_think" not in content:
                        msg["content"] = f"/no_think\n{content}" if content else "/no_think"
                    break
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    rendered_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=effective_enable_thinking,
                )
            except TypeError:
                return tokenizer.apply_chat_template(rendered_messages, tokenize=False, add_generation_prompt=True)
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in rendered_messages) + "\nASSISTANT:"

    def _generate_batch_text(
        self,
        prompts: Sequence[str],
        max_new_tokens_override: Optional[int] = None,
        force_no_think: bool = False,
    ) -> List[str]:
        if not self._ready or self._model is None:
            raise RuntimeError("LocalRoleLLM is not initialized")
        if not prompts:
            return []

        effective_enable_thinking = bool(self.enable_thinking and not force_no_think)
        if self.backend == "api":
            raise RuntimeError("API backend requires message-based generation path")

        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized")
        if self.backend == "vllm":
            if self._sampling_params_cls is None:
                raise RuntimeError("vLLM SamplingParams is not initialized")
            model = self._model
            max_input_tokens = resolve_effective_max_input_tokens(self.inference_cfg)
            if effective_enable_thinking:
                max_new_tokens = (
                    int(max_new_tokens_override)
                    if isinstance(max_new_tokens_override, int) and max_new_tokens_override > 0
                    else int(self.inference_cfg.get("think_max_new_tokens", 1024))
                )
                temperature = safe_float(self.inference_cfg.get("think_temperature", 0.6), default=0.6)
                top_p = safe_float(self.inference_cfg.get("think_top_p", 0.95), default=0.95)
                if temperature <= 0.0:
                    temperature = 0.6
            else:
                max_new_tokens = (
                    int(max_new_tokens_override)
                    if isinstance(max_new_tokens_override, int) and max_new_tokens_override > 0
                    else int(self.inference_cfg.get("no_think_max_new_tokens", 512))
                )
                temperature = safe_float(self.inference_cfg.get("no_think_temperature", 0.7), default=0.7)
                top_p = safe_float(self.inference_cfg.get("no_think_top_p", 0.8), default=0.8)
            top_k = int(self.inference_cfg.get("top_k", 20) or 0)
            min_p = safe_float(self.inference_cfg.get("min_p", 0.0), default=0.0)
            repetition_penalty = safe_float(self.inference_cfg.get("repetition_penalty", 1.0), default=1.0)
            sampling_kwargs: Dict[str, Any] = {
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "skip_special_tokens": not effective_enable_thinking,
            }
            if top_k > 0:
                sampling_kwargs["top_k"] = top_k
            if min_p > 0:
                sampling_kwargs["min_p"] = min_p
            if max_input_tokens > 0:
                sampling_kwargs["truncate_prompt_tokens"] = max_input_tokens
            stop_token_ids = self.inference_cfg.get("stop_token_ids")
            if isinstance(stop_token_ids, list) and stop_token_ids:
                sampling_kwargs["stop_token_ids"] = [int(x) for x in stop_token_ids]
            try:
                sampling_params = self._sampling_params_cls(**sampling_kwargs)
            except TypeError as exc:
                if "truncate_prompt_tokens" not in str(exc):
                    raise
                sampling_kwargs.pop("truncate_prompt_tokens", None)
                sampling_params = self._sampling_params_cls(**sampling_kwargs)
            outputs = model.generate(list(prompts), sampling_params=sampling_params, use_tqdm=False)
            texts: List[str] = []
            for item in outputs:
                if not getattr(item, "outputs", None):
                    texts.append("")
                    continue
                texts.append(str(item.outputs[0].text))
            return texts

        if self._torch is None:
            raise RuntimeError("Torch backend state is not initialized")
        model = self._model
        torch = self._torch

        max_input_tokens = resolve_effective_max_input_tokens(self.inference_cfg)
        tokenize_kwargs: Dict[str, Any] = {
            "return_tensors": "pt",
            "padding": True,
        }
        if max_input_tokens > 0:
            tokenize_kwargs["truncation"] = True
            tokenize_kwargs["max_length"] = max_input_tokens

        inputs = tokenizer(list(prompts), **tokenize_kwargs)

        model_device = getattr(model, "device", None)
        if model_device is None or getattr(model_device, "type", None) in {None, "meta"}:
            try:
                model_device = next(model.parameters()).device
            except StopIteration:
                model_device = None
        if model_device is not None and getattr(model_device, "type", None) not in {"meta", None}:
            try:
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
            except Exception:
                pass

        if effective_enable_thinking:
            max_new_tokens = (
                int(max_new_tokens_override)
                if isinstance(max_new_tokens_override, int) and max_new_tokens_override > 0
                else int(self.inference_cfg.get("think_max_new_tokens", 1024))
            )
            temperature = safe_float(self.inference_cfg.get("think_temperature", 0.6), default=0.6)
            top_p = safe_float(self.inference_cfg.get("think_top_p", 0.95), default=0.95)
            if temperature <= 0.0:
                temperature = 0.6
            do_sample = True
        else:
            max_new_tokens = (
                int(max_new_tokens_override)
                if isinstance(max_new_tokens_override, int) and max_new_tokens_override > 0
                else int(self.inference_cfg.get("no_think_max_new_tokens", 512))
            )
            temperature = safe_float(self.inference_cfg.get("no_think_temperature", 0.7), default=0.7)
            top_p = safe_float(self.inference_cfg.get("no_think_top_p", 0.8), default=0.8)
            do_sample = temperature > 0.0
        top_k = int(self.inference_cfg.get("top_k", 20) or 0)
        min_p = safe_float(self.inference_cfg.get("min_p", 0.0), default=0.0)
        repetition_penalty = safe_float(self.inference_cfg.get("repetition_penalty", 1.0), default=1.0)
        use_cache = parse_bool(self.inference_cfg.get("use_cache", True), default=True)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": use_cache,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "repetition_penalty": repetition_penalty,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            if top_k > 0:
                gen_kwargs["top_k"] = top_k
            if min_p > 0:
                gen_kwargs["min_p"] = min_p

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)

        outputs: List[str] = []
        prompt_width = int(inputs["input_ids"].shape[-1])
        for row_idx in range(output_ids.shape[0]):


            generated = output_ids[row_idx][prompt_width:]
            outputs.append(tokenizer.decode(generated, skip_special_tokens=not effective_enable_thinking))
        return outputs

    def generate_from_messages_batch(
        self,
        messages_batch: Sequence[Sequence[Dict[str, str]]],
        max_new_tokens_override: Optional[int] = None,
        force_no_think: bool = False,
    ) -> List[str]:
        effective_enable_thinking = bool(self.enable_thinking and not force_no_think)
        if self.backend == "api":
            if self._api_client is None:
                raise RuntimeError("API client is not initialized")
            if effective_enable_thinking:
                max_new_tokens = (
                    int(max_new_tokens_override)
                    if isinstance(max_new_tokens_override, int) and max_new_tokens_override > 0
                    else int(self.inference_cfg.get("think_max_new_tokens", 1024))
                )
                temperature = safe_float(self.inference_cfg.get("think_temperature", 0.6), default=0.6)
                top_p = safe_float(self.inference_cfg.get("think_top_p", 0.95), default=0.95)
                if temperature <= 0.0:
                    temperature = 0.6
            else:
                max_new_tokens = (
                    int(max_new_tokens_override)
                    if isinstance(max_new_tokens_override, int) and max_new_tokens_override > 0
                    else int(self.inference_cfg.get("no_think_max_new_tokens", 512))
                )
                temperature = safe_float(self.inference_cfg.get("no_think_temperature", 0.7), default=0.7)
                top_p = safe_float(self.inference_cfg.get("no_think_top_p", 0.8), default=0.8)
            return self._api_client.generate_from_messages_batch(
                messages_batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                enable_thinking=effective_enable_thinking,
            )

        prompts = [
            self._messages_to_prompt(messages, enable_thinking_override=effective_enable_thinking)
            for messages in messages_batch
        ]
        return self._generate_batch_text(
            prompts,
            max_new_tokens_override=max_new_tokens_override,
            force_no_think=force_no_think,
        )

    def clear_cuda_cache(self) -> None:
        if self.backend in {"vllm", "api"}:
            return
        if self._torch is None:
            return
        try:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
                if hasattr(self._torch.cuda, "ipc_collect"):
                    self._torch.cuda.ipc_collect()
        except Exception:
            pass

    def shutdown(self) -> None:
        model = self._model
        torch = self._torch
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._sampling_params_cls = None
        api_client = self._api_client
        self._api_client = None
        self._ready = False
        if api_client is not None:
            try:
                api_client.shutdown()
            except Exception:
                pass
        if self.backend == "api":
            return
        if self.backend == "vllm" and model is not None:
            llm_engine = getattr(model, "llm_engine", None)
            if llm_engine is not None:
                renderer = getattr(llm_engine, "renderer", None)
                if renderer is not None and hasattr(renderer, "shutdown"):
                    try:
                        renderer.shutdown()
                    except Exception:
                        pass
                output_processor = getattr(llm_engine, "output_processor", None)
                if output_processor is not None and hasattr(output_processor, "close"):
                    try:
                        output_processor.close()
                    except Exception:
                        pass
                engine_core = getattr(llm_engine, "engine_core", None)
                if engine_core is not None and hasattr(engine_core, "shutdown"):
                    try:
                        engine_core.shutdown(timeout=float(self.inference_cfg.get("shutdown_timeout_s", 30)))
                    except Exception:
                        pass
        try:
            del model
        except Exception:
            pass
        gc.collect()
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
            except Exception:
                pass


def generate_with_oom_split_retry(
    llm: LocalRoleLLM,
    messages_batch: Sequence[Sequence[Dict[str, str]]],
    role_name: str,
    logger: EventLogger,
    enable_split_retry: bool,
    max_depth: int,
    max_new_tokens_override: Optional[int] = None,
    force_no_think: bool = False,
    depth: int = 0,
) -> List[str]:
    try:
        return llm.generate_from_messages_batch(
            messages_batch,
            max_new_tokens_override=max_new_tokens_override,
            force_no_think=force_no_think,
        )
    except RuntimeError as exc:
        if not is_cuda_oom_error(exc):
            raise

        llm.clear_cuda_cache()
        batch_size = len(messages_batch)
        logger.event(
            "oom",
            "batch_generation_oom",
            level="WARNING",
            role=role_name,
            batch_size=batch_size,
            depth=depth,
            error=truncate_text(str(exc), max_chars=400),
        )

        if (not enable_split_retry) or batch_size <= 1 or depth >= max_depth:
            raise

        mid = batch_size // 2
        if mid <= 0 or mid >= batch_size:
            raise

        left_outputs = generate_with_oom_split_retry(
            llm=llm,
            messages_batch=messages_batch[:mid],
            role_name=role_name,
            logger=logger,
            enable_split_retry=enable_split_retry,
            max_depth=max_depth,
            max_new_tokens_override=max_new_tokens_override,
            force_no_think=force_no_think,
            depth=depth + 1,
        )
        right_outputs = generate_with_oom_split_retry(
            llm=llm,
            messages_batch=messages_batch[mid:],
            role_name=role_name,
            logger=logger,
            enable_split_retry=enable_split_retry,
            max_depth=max_depth,
            max_new_tokens_override=max_new_tokens_override,
            force_no_think=force_no_think,
            depth=depth + 1,
        )
        return left_outputs + right_outputs


def iter_micro_batches(rows: Sequence[Dict[str, Any]], micro_batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    if micro_batch_size <= 0:
        micro_batch_size = 1
    for start in range(0, len(rows), micro_batch_size):
        yield list(rows[start : start + micro_batch_size])


def build_group_prediction_base_row(row: Dict[str, Any], role_output_keys: Sequence[str]) -> Dict[str, Any]:
    group_id = str(row.get("group_id") or "")
    sample_id = (
        str(row.get("sample_id") or "")
        or str(parse_sample_id_from_group_id(group_id) or "")
        or str(row.get("id") or "")
        or str(row.get("original_id") or "")
        or group_id
    )
    return {
        "group_id": group_id,
        "sample_id": sample_id,
        "original_id": str(row.get("original_id") or ""),
        "source": str(row.get("source") or "unknown"),
        "language": str(row.get("language") or "en"),
        "split": str(row.get("split") or "train"),
        "local_group_id": int(row.get("local_group_id") or 1),
        "focus_text": str(row.get("focus_text") or f"<focus>{str(row.get('group_text') or '')}</focus>"),
        "group_text": str(row.get("group_text") or ""),
        "canonical_target": str(row.get("canonical_target") or "IMPLICIT_TARGET"),
        "escalation_flag": bool(row.get("escalation_flag")),
        "phase2_predictions": {key: [] for key in role_output_keys},
    }


def resolve_attempt_max_new_tokens(
    inference_cfg: Dict[str, Any],
    enable_thinking: bool,
    attempt: int,
    force_no_think_retry: bool = False,
) -> int:
    use_thinking_profile = enable_thinking and not (force_no_think_retry and attempt > 0)
    if use_thinking_profile:
        base = int(inference_cfg.get("think_max_new_tokens", 1024))
        retry = int(inference_cfg.get("think_retry_max_new_tokens", base))
    else:
        base = int(inference_cfg.get("no_think_max_new_tokens", 512))
        retry = int(inference_cfg.get("no_think_retry_max_new_tokens", base))
    return base if attempt <= 0 else retry


def process_rows(
    rows: Sequence[Dict[str, Any]],
    mode: str,
    role_specs: Dict[str, str],
    phase2_cfg: Dict[str, Any],
    inference_cfg: Dict[str, Any],
    paths: Paths,
    zh_lexicon_terms: Sequence[Dict[str, str]],
    role_prompt_overrides: Dict[str, Dict[str, str]],
    logger: EventLogger,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Set[str], Dict[str, Any]]:
    micro_batch_size = int(max(1, phase2_cfg.get("micro_batch_size", 4)))
    progress_interval = int(max(1, phase2_cfg.get("progress_interval", 200)))
    max_parse_retries = int(max(0, phase2_cfg.get("max_parse_retries", 1)))
    allow_fallback = parse_bool(phase2_cfg.get("allow_qwen_fallback_to_heuristic", True), default=True)
    require_cuda = parse_bool(inference_cfg.get("require_cuda", True), default=True)
    checkpoint_interval_groups = int(max(1, phase2_cfg.get("checkpoint_save_interval_groups", 200)))
    eta_log_interval_seconds = int(max(300, phase2_cfg.get("eta_log_interval_seconds", 3600)))
    resume_from_checkpoint = parse_bool(phase2_cfg.get("resume_from_checkpoint", True), default=True)
    enable_partial_candidate_export = parse_bool(phase2_cfg.get("enable_partial_candidate_export", True), default=True)
    enable_oom_split_retry = parse_bool(inference_cfg.get("enable_oom_split_retry", True), default=True)
    oom_split_retry_max_depth = int(max(0, inference_cfg.get("oom_split_retry_max_depth", 4)))
    clear_cuda_cache_per_batch = parse_bool(inference_cfg.get("clear_cuda_cache_per_batch", True), default=True)
    force_no_think_retry = parse_bool(inference_cfg.get("force_no_think_retry", True), default=True)

    checkpoint_path = paths.run_root / "phase2_checkpoint.json"
    role_order = list(role_specs.keys())
    role_output_keys = [ROLE_OUTPUT_KEY_MAP.get(role, role) for role in role_order]
    total_groups = len(rows)
    total_role_groups = total_groups * len(role_order)
    group_id_signature = build_group_id_signature(rows)
    run_started_at_iso = utc_now()

    grouped_rows: List[Dict[str, Any]] = [build_group_prediction_base_row(row, role_output_keys=role_output_keys) for row in rows]
    parse_fail_by_role: Dict[str, int] = {role: 0 for role in role_specs.keys()}
    lexicon_hit_group_ids: Set[str] = set()
    processed_by_role: Dict[str, int] = {role: 0 for role in role_specs.keys()}
    resumed_from_checkpoint = False

    checkpoint_state = load_checkpoint_state(checkpoint_path) if resume_from_checkpoint else None
    latest_eta_iso: Optional[str] = None
    eta_last_sampled_at: Optional[str] = None
    if checkpoint_state:
        checkpoint_started_at = checkpoint_state.get("started_at")
        if isinstance(checkpoint_started_at, str) and checkpoint_started_at:
            run_started_at_iso = checkpoint_started_at

        checkpoint_latest_eta = normalize_string(checkpoint_state.get("latest_eta_iso"))
        if checkpoint_latest_eta:
            latest_eta_iso = checkpoint_latest_eta

        checkpoint_eta_sampled_at = normalize_string(checkpoint_state.get("eta_last_sampled_at"))
        if checkpoint_eta_sampled_at:
            eta_last_sampled_at = checkpoint_eta_sampled_at

        saved_role_order = checkpoint_state.get("role_order")
        saved_signature = normalize_string(checkpoint_state.get("group_id_signature"))
        saved_rows = checkpoint_state.get("group_rows")

        compatible = (
            isinstance(saved_role_order, list)
            and [str(x) for x in saved_role_order] == role_order
            and saved_signature == group_id_signature
            and isinstance(saved_rows, list)
            and len(saved_rows) == len(grouped_rows)
        )

        if compatible:
            restored_rows: List[Dict[str, Any]] = []
            restore_ok = True
            for idx, base_row in enumerate(grouped_rows):
                saved_row = saved_rows[idx]
                if not isinstance(saved_row, dict):
                    restore_ok = False
                    break
                if str(saved_row.get("group_id") or "") != str(base_row.get("group_id") or ""):
                    restore_ok = False
                    break
                merged = dict(base_row)
                saved_preds = saved_row.get("phase2_predictions")
                if isinstance(saved_preds, dict):
                    merged["phase2_predictions"] = {
                        key: saved_preds.get(key) if isinstance(saved_preds.get(key), list) else []
                        for key in role_output_keys
                    }
                restored_rows.append(merged)

            if restore_ok:
                grouped_rows = restored_rows
                resumed_from_checkpoint = True

                saved_parse_fail = checkpoint_state.get("parse_fail_by_role")
                if isinstance(saved_parse_fail, dict):
                    for role in parse_fail_by_role.keys():
                        parse_fail_by_role[role] = int(max(0, int(saved_parse_fail.get(role, 0))))

                saved_lexicon_hits = checkpoint_state.get("lexicon_hit_group_ids")
                if isinstance(saved_lexicon_hits, list):
                    lexicon_hit_group_ids = {str(x) for x in saved_lexicon_hits if str(x)}

                saved_processed = checkpoint_state.get("processed_by_role")
                if isinstance(saved_processed, dict):
                    for role in processed_by_role.keys():
                        try:
                            raw_count = int(saved_processed.get(role, 0))
                        except (TypeError, ValueError):
                            raw_count = 0
                        processed_by_role[role] = max(0, min(len(grouped_rows), raw_count))

                logger.event(
                    "checkpoint",
                    "resume_loaded",
                    checkpoint_path=str(checkpoint_path),
                    processed_by_role=processed_by_role,
                )
            else:
                logger.event(
                    "checkpoint",
                    "resume_rejected_row_mismatch",
                    level="WARNING",
                    checkpoint_path=str(checkpoint_path),
                )
        else:
            logger.event(
                "checkpoint",
                "resume_rejected_incompatible",
                level="WARNING",
                checkpoint_path=str(checkpoint_path),
                saved_role_order=saved_role_order,
                saved_signature=saved_signature,
                expected_signature=group_id_signature,
            )

    def save_checkpoint(reason: str) -> None:
        state = {
            "version": 1,
            "run_id": paths.run_root.name,
            "started_at": run_started_at_iso,
            "updated_at": utc_now(),
            "mode": mode,
            "role_order": role_order,
            "total_groups": len(grouped_rows),
            "total_role_groups": total_role_groups,
            "processed_by_role": dict(processed_by_role),
            "parse_fail_by_role": dict(parse_fail_by_role),
            "lexicon_hit_group_ids": sorted(lexicon_hit_group_ids),
            "group_id_signature": group_id_signature,
            "group_rows": grouped_rows,
            "latest_eta_iso": latest_eta_iso,
            "eta_last_sampled_at": eta_last_sampled_at,
            "completed": all(processed_by_role.get(role, 0) >= len(grouped_rows) for role in role_order),
            "reason": reason,
        }
        try:
            write_checkpoint_state(checkpoint_path, state)
        except Exception as exc:
            logger.event(
                "checkpoint",
                "save_failed",
                level="WARNING",
                checkpoint_path=str(checkpoint_path),
                reason=reason,
                error=str(exc),
            )
            return

        processed_total = int(sum(int(v) for v in processed_by_role.values()))
        if enable_partial_candidate_export and processed_total > 0:
            try:
                partial_group_count, partial_candidate_count, partial_phase4_count = write_partial_candidate_outputs(
                    candidates_root=paths.run_root / "candidates",
                    phase4_root=paths.run_root / "phase4_inputs",
                    group_rows=grouped_rows,
                    role_names=list(role_specs.keys()),
                    parse_fail_by_role=parse_fail_by_role,
                )
                logger.event(
                    "export",
                    "partial_candidates_written",
                    reason=reason,
                    partial_groups=partial_group_count,
                    partial_candidates=partial_candidate_count,
                    partial_phase4=partial_phase4_count,
                )
            except Exception as exc:
                logger.event(
                    "export",
                    "partial_candidates_failed",
                    level="WARNING",
                    reason=reason,
                    error=truncate_text(str(exc), max_chars=400),
                )

    if not resumed_from_checkpoint:
        save_checkpoint(reason="initialized")

    llm: Optional[LocalRoleLLM] = None
    if mode in {"qwen_local", "vllm_local", "api"}:
        llm = LocalRoleLLM(model_path=paths.generator_model_path, inference_cfg=inference_cfg, prompt_template=DEFAULT_ROLE_PROMPT)
        try:
            llm.initialize()
            logger.event(
                "role_generation",
                "model_backend_ready",
                mode=mode,
                backend=llm.backend,
                cuda_available=llm.cuda_available,
                visible_cuda_devices=llm.visible_cuda_devices,
                model_device=llm.model_device,
            )
            logger.event(
                "role_generation",
                "sampling_profile",
                enable_thinking=llm.enable_thinking,
                think_max_new_tokens=int(inference_cfg.get("think_max_new_tokens", 1024)),
                no_think_max_new_tokens=int(inference_cfg.get("no_think_max_new_tokens", 512)),
                micro_batch_size=micro_batch_size,
                tensor_parallel_size=int(inference_cfg.get("tensor_parallel_size", 2)),
                gpu_memory_utilization=safe_float(inference_cfg.get("gpu_memory_utilization", 0.85), default=0.85),
            )
            if llm.enable_thinking and total_role_groups >= 1000:
                logger.event(
                    "role_generation",
                    "thinking_mode_enabled_warning",
                    level="WARNING",
                    total_groups=total_groups,
                    total_role_groups=total_role_groups,
                    think_max_new_tokens=int(inference_cfg.get("think_max_new_tokens", 1024)),
                    message="Thinking mode is enabled for large-scale Phase2 extraction. This can degrade throughput dramatically.",
                )
        except Exception as exc:
            if require_cuda and not allow_fallback:
                raise
            if require_cuda:
                raise
            logger.event("role_generation", "model_backend_unavailable_fallback", level="WARNING", mode=mode, message=str(exc))
            llm = None

    processed_role_groups = int(sum(int(v) for v in processed_by_role.values()))
    next_checkpoint_boundary = ((processed_role_groups // checkpoint_interval_groups) + 1) * checkpoint_interval_groups
    last_eta_log_mono = time.monotonic()

    def on_batch_finished(role_name: str, processed_count_for_role: int) -> None:
        nonlocal processed_role_groups
        nonlocal next_checkpoint_boundary
        nonlocal last_eta_log_mono
        nonlocal latest_eta_iso
        nonlocal eta_last_sampled_at

        processed_by_role[role_name] = max(0, min(len(grouped_rows), int(processed_count_for_role)))
        processed_role_groups = int(sum(int(v) for v in processed_by_role.values()))

        now_mono = time.monotonic()
        should_sample_eta = (now_mono - last_eta_log_mono >= eta_log_interval_seconds) or (
            processed_role_groups >= total_role_groups
        )
        if should_sample_eta:
            eta_iso = estimate_eta_iso(run_started_at_iso, processed=processed_role_groups, total=total_role_groups)
            sample_ts = utc_now()
            if eta_iso is None and processed_role_groups >= total_role_groups:
                eta_iso = sample_ts
            latest_eta_iso = eta_iso
            eta_last_sampled_at = sample_ts
            logger.event(
                "progress",
                "hourly_eta_sample",
                role=role_name,
                processed_role_groups=processed_role_groups,
                total_role_groups=total_role_groups,
                eta_iso=eta_iso,
                sampled_at=sample_ts,
            )
            last_eta_log_mono = now_mono

        if processed_role_groups >= next_checkpoint_boundary:
            save_checkpoint(reason=f"interval_{processed_role_groups}")
            next_checkpoint_boundary = ((processed_role_groups // checkpoint_interval_groups) + 1) * checkpoint_interval_groups

    for role_name in role_specs.keys():
        output_key = ROLE_OUTPUT_KEY_MAP.get(role_name, role_name)
        role_counter = int(processed_by_role.get(role_name, 0))

        if role_counter >= len(grouped_rows):
            logger.event("checkpoint", "role_already_completed", role=role_name, processed=role_counter)
            continue

        for batch_start in range(role_counter, len(grouped_rows), micro_batch_size):
            batch_rows = grouped_rows[batch_start : batch_start + micro_batch_size]

            if mode in {"heuristic", "stub"} or llm is None:
                for row in batch_rows:
                    group_id = str(row.get("group_id") or "")
                    if mode == "stub":
                        row["phase2_predictions"][output_key] = []
                    else:
                        row["phase2_predictions"][output_key] = build_heuristic_role_array_prediction(row=row, role_name=role_name)
                    role_counter += 1
                    if role_counter == 1 or role_counter % progress_interval == 0:
                        logger.event(
                            "role_generation",
                            "progress",
                            role=role_name,
                            index=role_counter,
                            group_id=group_id,
                            mode=mode,
                        )
                on_batch_finished(role_name=role_name, processed_count_for_role=role_counter)
                continue

            pending_indices: List[int] = list(range(len(batch_rows)))
            pending_errors: List[Optional[str]] = [None for _ in batch_rows]

            for attempt in range(max_parse_retries + 1):
                if not pending_indices:
                    break

                snapshot = list(pending_indices)
                messages_batch: List[List[Dict[str, str]]] = []
                force_no_think_this_attempt = bool(llm.enable_thinking and force_no_think_retry and attempt > 0)
                system_content = build_generation_system_message(
                    enable_thinking=bool(llm.enable_thinking),
                    force_no_think=force_no_think_this_attempt,
                    model_name=llm.model_path.name,
                )

                for local_idx in snapshot:
                    row = batch_rows[local_idx]
                    language = str(row.get("language") or "en")
                    template = select_role_prompt_template(
                        language=language,
                        role_name=role_name,
                        role_prompt_overrides=role_prompt_overrides,
                    )
                    lexicon_knowledge = ""
                    if role_name == "cultural_bystander" and language == "zh":
                        hits = lookup_zh_lexicon_hits(row=row, lexicon_terms=zh_lexicon_terms)
                        if hits:
                            lexicon_hit_group_ids.add(str(row.get("group_id") or ""))
                        lexicon_knowledge = build_lexicon_knowledge_block(hits)

                    retry_note = pending_errors[local_idx] if attempt > 0 else None
                    if retry_note and force_no_think_this_attempt and is_qwen35_family_model(llm.model_path.name):
                        retry_note = None

                    prompt = render_role_prompt_text(
                        template=template,
                        focus_text=str(row.get("focus_text") or f"<focus>{str(row.get('group_text') or '')}</focus>"),
                        group_text=str(row.get("group_text") or ""),
                        lexicon_knowledge=lexicon_knowledge,
                        retry_note=retry_note,
                        target_hint=normalize_string(row.get("canonical_target")) if normalize_string(row.get("canonical_target")) != "IMPLICIT_TARGET" else "",
                        language=language,
                    )
                    messages_batch.append(
                        [
                            {
                                "role": "system",
                                "content": system_content,
                            },
                            {"role": "user", "content": prompt},
                        ]
                    )

                try:
                    generation_max_new_tokens = resolve_attempt_max_new_tokens(
                        inference_cfg=inference_cfg,
                        enable_thinking=bool(llm.enable_thinking),
                        attempt=attempt,
                        force_no_think_retry=force_no_think_retry,
                    )
                    outputs = generate_with_oom_split_retry(
                        llm=llm,
                        messages_batch=messages_batch,
                        role_name=role_name,
                        logger=logger,
                        enable_split_retry=enable_oom_split_retry,
                        max_depth=oom_split_retry_max_depth,
                        max_new_tokens_override=generation_max_new_tokens,
                        force_no_think=force_no_think_this_attempt,
                    )
                except Exception as exc:
                    if not allow_fallback:
                        raise
                    for local_idx in snapshot:
                        row = batch_rows[local_idx]
                        row["phase2_predictions"][output_key] = []
                        parse_fail_by_role[role_name] = int(parse_fail_by_role.get(role_name, 0)) + 1
                        logger.event(
                            "oom",
                            "batch_generation_failed_fallback",
                            level="WARNING",
                            role=role_name,
                            group_id=str(row.get("group_id") or ""),
                            error=truncate_text(str(exc), max_chars=500),
                        )
                    pending_indices = []
                    break

                if len(outputs) != len(snapshot):
                    raise RuntimeError("Batch generation output size mismatch")

                next_pending: List[int] = []
                for out_idx, local_idx in enumerate(snapshot):
                    row = batch_rows[local_idx]
                    raw_output = outputs[out_idx]
                    language = str(row.get("language") or "en")






                    try:
                        parsed = extract_json_array_candidate(raw_output)
                        normalized = normalize_prediction_array_by_language(
                            parsed=parsed,
                            language=language,
                            focus_text=str(row.get("focus_text") or ""),
                            group_text=str(row.get("group_text") or ""),
                        )





                        if role_name == "cultural_bystander" and language == "zh":
                            group_id = str(row.get("group_id") or "")

                            if group_id in lexicon_hit_group_ids:
                                for item in normalized:

                                    if not str(item.get("rationale", "")).startswith("[Lexicon Hit]"):
                                        item["rationale"] = f"[Lexicon Hit] {item.get('rationale', '')}"


                        row["phase2_predictions"][output_key] = normalized
                        pending_errors[local_idx] = None

                    except Exception as exc:
                        pending_errors[local_idx] = str(exc)
                        next_pending.append(local_idx)
                        if attempt >= max_parse_retries:
                            row["phase2_predictions"][output_key] = []
                            parse_fail_by_role[role_name] = int(parse_fail_by_role.get(role_name, 0)) + 1
                            logger.event(
                                "parse",
                                "role_array_parse_failed",
                                level="WARNING",
                                role=role_name,
                                group_id=str(row.get("group_id") or ""),
                                error=str(exc),
                                raw_output=truncate_text(raw_output, max_chars=1200),
                            )
                pending_indices = next_pending

            for row in batch_rows:
                role_counter += 1
                if role_counter == 1 or role_counter % progress_interval == 0:
                    logger.event(
                        "role_generation",
                        "progress",
                        role=role_name,
                        index=role_counter,
                        group_id=str(row.get("group_id") or ""),
                        mode=mode,
                    )

            if llm is not None and clear_cuda_cache_per_batch:
                llm.clear_cuda_cache()

            on_batch_finished(role_name=role_name, processed_count_for_role=role_counter)

        save_checkpoint(reason=f"role_completed_{role_name}")

    save_checkpoint(reason="completed")

    completed_at_iso = utc_now()
    elapsed_seconds = 0.0
    try:
        elapsed_seconds = max(0.0, (datetime.fromisoformat(completed_at_iso) - datetime.fromisoformat(run_started_at_iso)).total_seconds())
    except Exception:
        elapsed_seconds = 0.0

    process_meta = {
        "checkpoint_path": str(checkpoint_path),
        "resumed_from_checkpoint": bool(resumed_from_checkpoint),
        "started_at": run_started_at_iso,
        "completed_at": completed_at_iso,
        "elapsed_seconds": elapsed_seconds,
        "processed_role_groups": int(sum(int(v) for v in processed_by_role.values())),
        "total_role_groups": int(total_role_groups),
        "throughput_role_groups_per_second": round(
            (int(sum(int(v) for v in processed_by_role.values())) / elapsed_seconds), 6
        )
        if elapsed_seconds > 0
        else 0.0,
        "expected_finish_eta_iso": latest_eta_iso,
        "eta_last_sampled_at": eta_last_sampled_at,
    }

    return grouped_rows, parse_fail_by_role, lexicon_hit_group_ids, process_meta


def _validate_group_predictions_array(items: Any, language: str) -> bool:
    if not isinstance(items, list):
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        if language == "zh":
            required = {"rationale", "target", "argument", "group", "hateful"}
        else:
            required = {"rationale", "target", "argument", "harmful"}
        if set(item.keys()) != required:
            return False
    return True


def quality_gates(group_rows: Sequence[Dict[str, Any]], expected_roles: Sequence[str]) -> Dict[str, Any]:
    missing_group_id = 0
    invalid_split = 0
    missing_phase2_predictions = 0
    invalid_phase2_predictions = 0

    expected_output_keys = {ROLE_OUTPUT_KEY_MAP.get(role, role) for role in expected_roles}

    for row in group_rows:
        if not normalize_string(row.get("group_id")):
            missing_group_id += 1
        if str(row.get("split")) not in {"train", "dev", "test"}:
            invalid_split += 1

        language = str(row.get("language") or "en")
        preds = row.get("phase2_predictions")
        if not isinstance(preds, dict):
            missing_phase2_predictions += 1
            continue

        if set(preds.keys()) != expected_output_keys:
            invalid_phase2_predictions += 1
            continue

        for key in expected_output_keys:
            if not _validate_group_predictions_array(preds.get(key), language=language):
                invalid_phase2_predictions += 1
                break

    passed = (
        missing_group_id == 0
        and invalid_split == 0
        and missing_phase2_predictions == 0
        and invalid_phase2_predictions == 0
    )

    return {
        "total_groups": len(group_rows),
        "missing_group_id": missing_group_id,
        "invalid_split": invalid_split,
        "missing_phase2_predictions": missing_phase2_predictions,
        "invalid_phase2_predictions": invalid_phase2_predictions,
        "expected_roles": list(expected_roles),
        "passed": passed,
    }


def build_summary(
    group_rows: Sequence[Dict[str, Any]],
    parse_fail_by_role: Dict[str, int],
    lexicon_hit_group_ids: Set[str],
) -> Dict[str, Any]:
    split_dist = dict(sorted(Counter(str(r.get("split")) for r in group_rows).items()))
    lang_dist = dict(sorted(Counter(str(r.get("language")) for r in group_rows).items()))
    source_dist = dict(sorted(Counter(str(r.get("source")) for r in group_rows).items()))

    total_role_predictions: Dict[str, int] = {"victim": 0, "moderator": 0, "bystander": 0}
    nonempty_groups: Dict[str, int] = {"victim": 0, "moderator": 0, "bystander": 0}
    for row in group_rows:
        preds = row.get("phase2_predictions") if isinstance(row.get("phase2_predictions"), dict) else {}
        for role_key in ["victim", "moderator", "bystander"]:
            role_items = preds.get(role_key)
            if isinstance(role_items, list):
                total_role_predictions[role_key] += len(role_items)
                if role_items:
                    nonempty_groups[role_key] += 1

    parse_fail_total = int(sum(int(v) for v in parse_fail_by_role.values()))
    return {
        "group_count": len(group_rows),
        "split_distribution": split_dist,
        "language_distribution": lang_dist,
        "source_distribution": source_dist,
        "total_role_predictions": total_role_predictions,
        "nonempty_group_count_by_role": nonempty_groups,
        "model_parse_fail_count": parse_fail_total,
        "model_parse_fail_by_role": dict(sorted(parse_fail_by_role.items())),
        "zh_bystander_lexicon_hit_groups": len(lexicon_hit_group_ids),
    }


def build_compat_role_rows(
    group_rows: Sequence[Dict[str, Any]],
    role_names: Sequence[str],
    parse_fail_by_role: Dict[str, int],
) -> List[Dict[str, Any]]:
    """导出兼容 all_roles 视图，同时保留最小可用的 Phase3/Phase4 关键字段。"""
    rows: List[Dict[str, Any]] = []

    def harmful_to_direction(raw_value: Any) -> str:
        label = normalize_hate_label(raw_value)
        if label == 1:
            return "harmful"
        if label == 0:
            return "non-harmful"
        return "unknown"

    def build_predicted_tuple(item: Dict[str, Any], language: str) -> Dict[str, Any]:
        harmful_key = "hateful" if language == "zh" else "harmful"
        harmful = normalize_hate_label(item.get(harmful_key))
        harmful_direction = harmful_to_direction(item.get(harmful_key))
        return {
            "target": normalize_string(item.get("target")) or "IMPLICIT_TARGET",
            "argument": normalize_string(item.get("argument")),
            "group": normalize_string(item.get("group")) or "unknown",
            "harmful": harmful,
            "harmful_direction": harmful_direction,
        }

    for row in group_rows:
        group_id = str(row.get("group_id") or "")
        sample_id = str(row.get("sample_id") or parse_sample_id_from_group_id(group_id) or row.get("original_id") or group_id)
        original_id = str(row.get("original_id") or sample_id)
        source = str(row.get("source") or "unknown")
        language = str(row.get("language") or "en")
        split = str(row.get("split") or "train")
        local_group_id = int(row.get("local_group_id") or 1)
        focus_text = str(row.get("focus_text") or f"<focus>{str(row.get('group_text') or '')}</focus>")
        group_text = str(row.get("group_text") or "")
        canonical_target = str(row.get("canonical_target") or "IMPLICIT_TARGET")
        escalation_flag = bool(row.get("escalation_flag"))
        preds_by_role = row.get("phase2_predictions") if isinstance(row.get("phase2_predictions"), dict) else {}

        for role_name in role_names:
            role_key = ROLE_OUTPUT_KEY_MAP.get(role_name, role_name)
            role_items = preds_by_role.get(role_key, [])
            if not role_items:
                continue

            for idx, item in enumerate(role_items, start=1):
                if not isinstance(item, dict):
                    continue
                predicted_tuple = build_predicted_tuple(item=item, language=language)
                rationale = normalize_string(item.get("rationale")) or ""
                role_candidate_id = f"{group_id}:{role_name}:{idx}"
                input_package = {
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "original_id": original_id,
                    "source": source,
                    "split": split,
                    "language": language,
                    "local_group_id": local_group_id,
                    "focus_text": focus_text,
                    "group_text": group_text,
                    "canonical_target": canonical_target,
                    "source_target": canonical_target if canonical_target != "IMPLICIT_TARGET" else None,
                    "source_argument": group_text or None,
                    "source_harmful": None,
                    "segmenter_confidence": 0.0,
                    "escalation_flag": escalation_flag,
                }
                phase4_role_hypothesis = {
                    "task": role_name,
                    "role": role_name,
                    "role_candidate_id": role_candidate_id,
                    "group_id": group_id,
                    "sample_id": sample_id,
                    "tuple": predicted_tuple,
                    "rationale": rationale,
                    "uncertainty_flag": False,
                    "escalation_flag": escalation_flag,
                }
                rows.append({
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "original_id": original_id,
                    "source": source,
                    "split": split,
                    "role": role_name,
                    "language": language,
                    "local_group_id": local_group_id,
                    "focus_text": focus_text,
                    "group_text": group_text,
                    "canonical_target": canonical_target,
                    "escalation_flag": escalation_flag,
                    "role_candidate_id": role_candidate_id,
                    "prediction": item,
                    "predicted_tuple": predicted_tuple,
                    "rationale": rationale,
                    "uncertainty_flag": False,
                    "fallback_used": False,
                    "model_trace": {"parse_fail": False, "phase2_parse_fail_count_for_role": int(parse_fail_by_role.get(role_name, 0))},
                    "input_package": input_package,
                    "phase4_role_hypothesis": phase4_role_hypothesis,
                })
    return rows


def build_phase4_role_rows(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cand in candidates:
        payload = cand.get("phase4_role_hypothesis") if isinstance(cand.get("phase4_role_hypothesis"), dict) else {}
        if not payload:
            payload = {
                "task": str(cand.get("role") or "unknown"),
                "role": str(cand.get("role") or "unknown"),
                "role_candidate_id": str(cand.get("role_candidate_id") or ""),
                "group_id": str(cand.get("group_id") or ""),
                "sample_id": str(cand.get("sample_id") or ""),
                "tuple": cand.get("predicted_tuple") if isinstance(cand.get("predicted_tuple"), dict) else {},
                "rationale": str(cand.get("rationale") or ""),
                "uncertainty_flag": bool(cand.get("uncertainty_flag")),
                "escalation_flag": bool(cand.get("escalation_flag")),
            }
        rows.append(
            {
                "sample_id": str(cand.get("sample_id") or ""),
                "group_id": str(cand.get("group_id") or ""),
                "role": str(cand.get("role") or "unknown"),
                "role_candidate_id": str(cand.get("role_candidate_id") or ""),
                "source": str(cand.get("source") or "unknown"),
                "split": str(cand.get("split") or "train"),
                "language": str(cand.get("language") or "unknown"),
                "role_hypothesis": payload,
            }
        )
    return rows


def write_partial_candidate_outputs(
    candidates_root: Path,
    phase4_root: Optional[Path],
    group_rows: Sequence[Dict[str, Any]],
    role_names: Sequence[str],
    parse_fail_by_role: Dict[str, int],
) -> Tuple[int, int, int]:
    partial_group_rows: List[Dict[str, Any]] = []
    for row in group_rows:
        preds = row.get("phase2_predictions") if isinstance(row.get("phase2_predictions"), dict) else {}
        if any(isinstance(items, list) and items for items in preds.values()):
            partial_group_rows.append(row)

    compat_rows = build_compat_role_rows(group_rows=partial_group_rows, role_names=role_names, parse_fail_by_role=parse_fail_by_role)
    phase4_role_rows = build_phase4_role_rows(compat_rows)

    write_jsonl(candidates_root / "group_predictions.partial.jsonl", partial_group_rows)
    write_jsonl(candidates_root / "all_roles.partial.jsonl", compat_rows)
    write_jsonl(candidates_root / "phase4_role_hypotheses.partial.jsonl", phase4_role_rows)

    if phase4_root is not None:
        write_jsonl(phase4_root / "phase4_role_hypotheses.partial.jsonl", phase4_role_rows)

    return len(partial_group_rows), len(compat_rows), len(phase4_role_rows)


def write_candidate_outputs(
    candidates_root: Path,
    phase4_root: Optional[Path],
    group_rows: Sequence[Dict[str, Any]],
    role_names: Sequence[str],
    parse_fail_by_role: Dict[str, int],
) -> Tuple[int, int]:
    compat_rows = build_compat_role_rows(group_rows=group_rows, role_names=role_names, parse_fail_by_role=parse_fail_by_role)
    phase4_role_rows = build_phase4_role_rows(compat_rows)

    write_jsonl(candidates_root / "group_predictions.jsonl", group_rows)
    write_jsonl(candidates_root / "all_roles.jsonl", compat_rows)
    write_jsonl(candidates_root / "phase4_role_hypotheses.jsonl", phase4_role_rows)

    if phase4_root is not None:
        write_jsonl(phase4_root / "phase4_role_hypotheses.jsonl", phase4_role_rows)

    for role_name in role_names:
        role_rows = [c for c in compat_rows if c.get("role") == role_name]
        write_jsonl(candidates_root / "roles" / f"{role_name}.jsonl", role_rows)

    for split in ["train", "dev", "test"]:
        split_group_rows = [g for g in group_rows if g.get("split") == split]
        write_jsonl(candidates_root / "by_split" / f"group_{split}.jsonl", split_group_rows)

    for lang, split in LANGUAGE_SPLIT_ORDER:
        split_group_rows = [
            g for g in group_rows if str(g.get("language") or "") == lang and str(g.get("split") or "") == split
        ]
        write_jsonl(candidates_root / "by_lang_split" / f"group_{lang}_{split}.jsonl", split_group_rows)
        write_jsonl(candidates_root / f"phase2_{lang}_{split}.jsonl", split_group_rows)

    return len(compat_rows), len(phase4_role_rows)


def run_phase2(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    cli_args = args or parse_args()
    workspace_root = Path(__file__).resolve().parents[2]
    config_path = Path(cli_args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)
    apply_cli_overrides(cfg, cli_args)
    paths = build_paths(args=cli_args, cfg=cfg, workspace_root=workspace_root)
    run_id = paths.run_root.name

    logger = EventLogger(
        run_id=run_id,
        log_path=paths.run_root / "logs" / "events.jsonl",
        level=str(cfg.get("logging", {}).get("level", "INFO")),
    )

    try:
        phase2_cfg = cfg.get("phase2", {})
        inference_cfg = cfg.get("inference", {})
        runtime_cfg = cfg.get("runtime", {})
        mode = str(phase2_cfg.get("mode", "qwen_local"))
        if mode not in {"heuristic", "qwen_local", "vllm_local", "api", "stub"}:
            raise ValueError(f"Unsupported mode: {mode}")

        role_specs = parse_role_specs(phase2_cfg)
        role_prompt_overrides = load_role_prompt_overrides(paths=paths, logger=logger)

        zh_lexicon_path = resolve_zh_lexicon_path(paths=paths, cfg=cfg)
        zh_lexicon_terms = load_zh_annotated_lexicon(zh_lexicon_path)

        logger.event(
            "bootstrap",
            "run_started",
            config_path=str(paths.config_path),
            output_root=str(paths.run_root),
            model_root=str(paths.model_root),
            generator_model_path=str(paths.generator_model_path),
            role_names=list(role_specs.keys()),
            role_prompt_override_roles=sorted(role_prompt_overrides.keys()),
            zh_lexicon_path=str(zh_lexicon_path),
            zh_lexicon_terms=len(zh_lexicon_terms),
        )

        all_rows, loaded_paths = load_phase1_rows_by_language_split(paths=paths, cfg=cfg, logger=logger)
        logger.event("load", "input_loaded", groups=len(all_rows), loaded_paths=loaded_paths)

        allowed_splits = set(parse_allowed_splits(cfg))
        allowed_languages = set(parse_allowed_languages(cfg))
        filtered_rows = [
            r
            for r in all_rows
            if str(r.get("split")) in allowed_splits and str(r.get("language") or "en").lower() in allowed_languages
        ]

        max_samples_per_split = runtime_cfg.get("max_samples_per_split")
        if isinstance(max_samples_per_split, int) and max_samples_per_split > 0:
            filtered_rows = limit_rows_per_language_split(filtered_rows, max_per_split=max_samples_per_split)

        max_samples = runtime_cfg.get("max_samples")
        if isinstance(max_samples, int) and max_samples > 0:
            filtered_rows = filtered_rows[:max_samples]
        logger.event(
            "load",
            "filtered_input",
            groups=len(filtered_rows),
            splits=sorted(allowed_splits),
            languages=sorted(allowed_languages),
            max_samples_per_split=max_samples_per_split if isinstance(max_samples_per_split, int) else None,
        )

        process_phase2_cfg = dict(phase2_cfg)
        process_phase2_cfg["progress_interval"] = runtime_cfg.get("progress_interval", 200)
        if "checkpoint_save_interval_groups" in runtime_cfg:
            process_phase2_cfg["checkpoint_save_interval_groups"] = runtime_cfg.get("checkpoint_save_interval_groups")
        if "eta_log_interval_seconds" in runtime_cfg:
            process_phase2_cfg["eta_log_interval_seconds"] = runtime_cfg.get("eta_log_interval_seconds")
        if "resume_from_checkpoint" in runtime_cfg:
            process_phase2_cfg["resume_from_checkpoint"] = runtime_cfg.get("resume_from_checkpoint")

        group_rows, parse_fail_by_role, lexicon_hit_group_ids, process_meta = process_rows(
            rows=filtered_rows,
            mode=mode,
            role_specs=role_specs,
            phase2_cfg=process_phase2_cfg,
            inference_cfg=inference_cfg,
            paths=paths,
            zh_lexicon_terms=zh_lexicon_terms,
            role_prompt_overrides=role_prompt_overrides,
            logger=logger,
        )
        logger.event("checkpoint", "process_rows_completed", **process_meta)

        quality = quality_gates(group_rows=group_rows, expected_roles=list(role_specs.keys()))
        write_json(paths.run_root / "reports" / "quality_gates.json", quality)
        logger.event("quality", "quality_report_written", **quality)

        strict_mode = parse_bool(cfg.get("quality", {}).get("strict", False), default=False)
        if strict_mode and not quality.get("passed", False):
            raise RuntimeError(f"Phase2 quality gate failed under strict mode: {quality}")

        summary = build_summary(
            group_rows=group_rows,
            parse_fail_by_role=parse_fail_by_role,
            lexicon_hit_group_ids=lexicon_hit_group_ids,
        )
        write_json(paths.run_root / "reports" / "summary.json", summary)
        logger.event("report", "summary_written", **summary)

        compat_candidate_count = 0
        phase4_role_hypotheses_count = 0

        if not cli_args.dry_run:
            compat_candidate_count, phase4_role_hypotheses_count = write_candidate_outputs(
                candidates_root=paths.run_root / "candidates",
                phase4_root=paths.run_root / "phase4",
                group_rows=group_rows,
                role_names=list(role_specs.keys()),
                parse_fail_by_role=parse_fail_by_role,
            )

            latest_candidates_root = paths.output_root / "latest" / "candidates"
            write_candidate_outputs(
                candidates_root=latest_candidates_root,
                phase4_root=paths.output_root / "latest" / "phase4",
                group_rows=group_rows,
                role_names=list(role_specs.keys()),
                parse_fail_by_role=parse_fail_by_role,
            )

            logger.event(
                "export",
                "candidates_written",
                group_rows=len(group_rows),
                compat_candidates=compat_candidate_count,
                phase4_role_hypotheses=phase4_role_hypotheses_count,
            )
        else:
            logger.event("export", "dry_run_skip_large_writes")

        run_summary = {
            "run_id": run_id,
            "created_at": utc_now(),
            "config_path": str(paths.config_path),
            "input_jsonl": str(paths.input_jsonl),
            "loaded_phase1_inputs": loaded_paths,
            "output_root": str(paths.run_root),
            "mode": mode,
            "model_path": str(paths.generator_model_path),
            "prompt_mode": "built_in_6_templates_with_file_override" if role_prompt_overrides else "built_in_6_templates",
            "role_prompt_overrides": {
                role_name: sorted(lang_map.keys())
                for role_name, lang_map in role_prompt_overrides.items()
            },
            "zh_lexicon_path": str(zh_lexicon_path),
            "roles": list(role_specs.keys()),
            "filters": {
                "splits": sorted(allowed_splits),
                "languages": sorted(allowed_languages),
                "max_samples": max_samples if isinstance(max_samples, int) and max_samples > 0 else None,
                "max_samples_per_split": (
                    runtime_cfg.get("max_samples_per_split")
                    if isinstance(runtime_cfg.get("max_samples_per_split"), int)
                    else None
                ),
            },
            "counts": {
                "input_groups": len(filtered_rows),
                "output_group_rows": len(group_rows),
                "output_compat_candidates": compat_candidate_count,
                "output_phase4_role_hypotheses": phase4_role_hypotheses_count,
            },
            "processing": process_meta,
            "quality": quality,
            "summary": summary,
        }
        summary_path = paths.run_root / "run_summary.json"
        write_json(summary_path, run_summary)
        logger.event("bootstrap", "run_completed", summary_path=str(summary_path))
        return run_summary
    finally:
        logger.close()


def main() -> None:
    run_phase2()


if __name__ == "__main__":
    main()

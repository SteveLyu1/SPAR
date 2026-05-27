from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List

import spacy


@dataclass(frozen=True)
class SentenceInfo:
    sentence_id: int
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class ParsedSyntax:
    conll: str
    sentences: List[SentenceInfo]


MODEL_BY_LANGUAGE = {
    "en": "en_core_web_sm",
    "zh": "zh_core_web_sm",
}


def _normalize_language(language: str) -> str:
    lang = str(language or "").strip().lower()
    if lang.startswith("zh"):
        return "zh"
    return "en"


@lru_cache(maxsize=2)
def get_nlp(language: str):
    normalized = _normalize_language(language)
    model_name = MODEL_BY_LANGUAGE[normalized]
    return spacy.load(model_name)


def parse_text_to_conll(text: str, language: str) -> ParsedSyntax:
    normalized = _normalize_language(language)
    nlp = get_nlp(normalized)
    doc = nlp(text)

    lines: List[str] = []
    sentences: List[SentenceInfo] = []
    header = "ID\tTEXT\tLEMMA\tPOS\tTAG\tFEATS\tHEAD\tDEPREL\tDEPS\tMISC"

    for sentence_index, sent in enumerate(doc.sents, start=1):
        sentence_text = sent.text.strip()
        if not sentence_text:
            continue
        sentence_info = SentenceInfo(
            sentence_id=sentence_index,
            text=sentence_text,
            start_char=sent.start_char,
            end_char=sent.end_char,
        )
        sentences.append(sentence_info)

        lines.append(f"# sent_id = {sentence_index}")
        lines.append(f"# text = {sentence_text}")
        lines.append(header)

        token_index_map = {token.i: local_idx for local_idx, token in enumerate(sent, start=1)}
        for local_idx, token in enumerate(sent, start=1):
            head_idx = 0 if token.head == token else token_index_map.get(token.head.i, 0)
            row = [
                str(local_idx),
                token.text,
                token.lemma_,
                token.pos_,
                token.tag_,
                "_",
                str(head_idx),
                token.dep_,
                "_",
                "_",
            ]
            lines.append("\t".join(row))
        lines.append("")

    return ParsedSyntax(conll="\n".join(lines).strip(), sentences=sentences)

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from pydantic import Field

from moatrader.canonical.models import ContractModel
from moatrader.evidence.models import EvidenceCard


MOAT_QUESTIONS = (
    "무엇이 고객이 공급자를 바꾸기 어렵게 하는가 switching cost customer retention qualification",
    "무엇이 신규 경쟁자의 진입을 막는가 entry barrier regulation scale",
    "무엇이 경쟁사보다 높은 가격을 가능하게 하는가 pricing power brand patent",
    "무엇이 지속적인 원가 우위를 만드는가 cost advantage scale yield sourcing",
    "고객 또는 공급자 집중이 만드는 취약점은 무엇인가 concentration vulnerability",
    "시장점유율이 지속되는 근거는 무엇인가 market share persistence retention",
    "현재 경쟁우위를 파괴할 반대 증거는 무엇인가 competition substitution technology risk",
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*|[가-힣]{2,}|\d+(?:\.\d+)?")


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


class RetrievalHit(ContractModel):
    question: str
    evidence_id: str
    source_chunk_id: str
    score: float = Field(ge=0.0)


class RetrievalResult(ContractModel):
    hits: list[RetrievalHit] = Field(default_factory=list)
    chunk_relevance: dict[str, float] = Field(default_factory=dict)


class EvidenceRetriever:
    def __init__(self, *, top_k_per_question: int = 5, k1: float = 1.5, b: float = 0.75) -> None:
        self.top_k = top_k_per_question
        self.k1 = k1
        self.b = b

    def retrieve(
        self,
        cards: list[EvidenceCard],
        questions: tuple[str, ...] = MOAT_QUESTIONS,
    ) -> RetrievalResult:
        if not cards:
            return RetrievalResult()
        docs = [
            _tokens(" ".join([card.evidence_type.value, card.fact, *card.mechanism, card.segment or ""]))
            for card in cards
        ]
        frequencies = [Counter(doc) for doc in docs]
        avg_length = sum(len(doc) for doc in docs) / len(docs) or 1.0
        document_frequency: Counter[str] = Counter()
        for doc in docs:
            document_frequency.update(set(doc))
        hits: list[RetrievalHit] = []
        chunk_scores: dict[str, float] = defaultdict(float)
        for question in questions:
            query_tokens = _tokens(question)
            scored: list[tuple[float, int]] = []
            for index, (doc, frequency) in enumerate(zip(docs, frequencies, strict=True)):
                score = 0.0
                for token in query_tokens:
                    tf = frequency[token]
                    if not tf:
                        continue
                    df = document_frequency[token]
                    idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
                    denominator = tf + self.k1 * (1 - self.b + self.b * len(doc) / avg_length)
                    score += idf * tf * (self.k1 + 1) / denominator
                card = cards[index]
                score *= 0.5 + 0.5 * card.strength
                score *= 0.5 + 0.5 * card.reliability
                if score > 0:
                    scored.append((score, index))
            for score, index in sorted(scored, reverse=True)[: self.top_k]:
                card = cards[index]
                hits.append(
                    RetrievalHit(
                        question=question,
                        evidence_id=card.evidence_id,
                        source_chunk_id=card.source_chunk_id,
                        score=score,
                    )
                )
                chunk_scores[card.source_chunk_id] = max(chunk_scores[card.source_chunk_id], score)
        return RetrievalResult(hits=hits, chunk_relevance=dict(chunk_scores))


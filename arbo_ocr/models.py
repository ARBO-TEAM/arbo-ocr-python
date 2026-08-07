from dataclasses import dataclass, field
from typing import Any

from .exceptions import OcrError


@dataclass(frozen=True)
class LineResult:
    """One recognized text line. `polygon` is a list of {"x": float, "y": float}
    points, in the order arboOCR reports them (clockwise from top-left-ish).

    `words` holds one {"text": str, "score": float, "polygon": [...]} dict per
    word (per character for CJK), and is only populated when the Engine was
    given `word_boxes=True` — arboOCR omits the JSON key entirely otherwise,
    so an empty list is the normal, successful default. Kept as raw dicts for
    the same reason `polygon` is: it mirrors the JSON one-to-one and needs no
    extra result class."""

    text: str
    score: float
    det_score: float
    polygon: list[dict[str, float]]
    words: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "LineResult":
        return LineResult(
            text=str(data.get("text", "")),
            score=float(data.get("score", 0.0)),
            det_score=float(data.get("detScore", 0.0)),
            polygon=data.get("polygon", []),
            words=data.get("words") or [],
        )


@dataclass(frozen=True)
class PageResult:
    """Full-page OCR result — mirrors arboOCR's PagePrediction. Empty `lines`
    is a normal, successful result (no text found), not an error."""

    backend: str
    image: str
    elapsed_ms: float
    lines: list[LineResult]

    @staticmethod
    def from_json(raw: str) -> "PageResult":
        import json

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise OcrError(
                f"arboocr_demo --json produced unparseable output: {raw[:500]!r}"
            ) from e

        if not isinstance(data, dict) or not isinstance(data.get("lines"), list):
            raise OcrError(
                f"arboocr_demo --json produced unparseable output: {raw[:500]!r}"
            )

        lines = [LineResult.from_dict(line) for line in data["lines"]]
        return PageResult(
            backend=str(data.get("backend", "")),
            image=str(data.get("image", "")),
            elapsed_ms=float(data.get("elapsedMs", 0.0)),
            lines=lines,
        )

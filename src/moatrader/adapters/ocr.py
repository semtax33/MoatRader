from __future__ import annotations

import json
import os
import sysconfig
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from statistics import fmean
from typing import Any, Iterable, Protocol


@dataclass(frozen=True, slots=True)
class PdfOcrBlock:
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class PdfOcrResult:
    blocks: tuple[PdfOcrBlock, ...]
    dpi: int
    mean_confidence: float
    engine: str


class PdfOcrAdapter(Protocol):
    @property
    def name(self) -> str: ...

    def extract_page(self, page: Any, *, dpi: int) -> PdfOcrResult: ...


_WINDOWS_DLL_HANDLES: list[Any] = []


def _installed_version(*distribution_names: str) -> str | None:
    for name in distribution_names:
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return None


def _configure_windows_gpu_dlls() -> None:
    """Expose pip-installed NVIDIA runtime DLLs to Paddle on Windows."""

    if os.name != "nt" or _WINDOWS_DLL_HANDLES:
        return
    nvidia_root = sysconfig.get_paths()["purelib"]
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return
    from pathlib import Path

    root = Path(nvidia_root) / "nvidia"
    if not root.is_dir():
        return
    dll_directories = sorted(path for path in root.glob("*/bin") if path.is_dir())
    for directory in dll_directories:
        _WINDOWS_DLL_HANDLES.append(add_dll_directory(str(directory)))
    if dll_directories:
        os.environ["PATH"] = os.pathsep.join(
            [*(str(path) for path in dll_directories), os.environ.get("PATH", "")]
        )


class PaddlePdfOcrAdapter:
    """Lazy PaddleOCR 3.x adapter with PDF-coordinate output.

    The implementation follows Synalyst's reviewed Korean PP-OCRv5 setup but
    keeps the dependency optional for installations that ingest only digital
    filings.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        language: str = "korean",
        ocr_version: str = "PP-OCRv5",
        cpu_threads: int = 6,
        detection_model_name: str | None = None,
        recognition_model_name: str | None = None,
        pipeline: Any | None = None,
    ) -> None:
        if not 1 <= cpu_threads <= 64:
            raise ValueError("PaddleOCR cpu_threads must be between 1 and 64")
        if detection_model_name is None and recognition_model_name is None:
            if language == "korean" and ocr_version == "PP-OCRv5":
                detection_model_name = "PP-OCRv5_server_det"
                recognition_model_name = "korean_PP-OCRv5_mobile_rec"
        elif detection_model_name is None or recognition_model_name is None:
            raise ValueError(
                "PaddleOCR detection and recognition model names must be supplied together"
            )
        if pipeline is None and find_spec("paddleocr") is None:
            raise RuntimeError(
                "PaddleOCR is optional; install MoatRader with the 'ocr' extra "
                "before enabling the paddle IR OCR engine"
            )
        self.device = device
        self.language = language
        self.ocr_version = ocr_version
        self.cpu_threads = cpu_threads
        self.detection_model_name = detection_model_name
        self.recognition_model_name = recognition_model_name
        self.pipeline = pipeline
        self.runtime_versions = {
            "paddleocr": _installed_version("paddleocr"),
            "paddlepaddle": _installed_version("paddlepaddle-gpu", "paddlepaddle"),
        }

    @property
    def name(self) -> str:
        return f"paddleocr-v3:{self.ocr_version}:{self.language}"

    def _ensure_pipeline(self) -> Any:
        if self.pipeline is not None:
            return self.pipeline
        _configure_windows_gpu_dlls()
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PaddleOCR is optional; install MoatRader with the 'ocr' extra "
                "before enabling the paddle IR OCR engine"
            ) from exc
        model_options = (
            {
                "text_detection_model_name": self.detection_model_name,
                "text_recognition_model_name": self.recognition_model_name,
            }
            if self.detection_model_name is not None
            else {"lang": self.language, "ocr_version": self.ocr_version}
        )
        self.pipeline = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=self.device,
            cpu_threads=self.cpu_threads,
            **model_options,
        )
        return self.pipeline

    def extract_page(self, page: Any, *, dpi: int) -> PdfOcrResult:
        if dpi < 72:
            raise ValueError("OCR DPI must be at least 72")
        try:
            import fitz
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Paddle IR OCR requires PyMuPDF and numpy") from exc
        matrix = page.get_pixmap(
            dpi=dpi,
            colorspace=fitz.csRGB,
            alpha=False,
        )
        image = np.frombuffer(matrix.samples, dtype=np.uint8).reshape(
            matrix.height,
            matrix.width,
            matrix.n,
        )
        results = self._ensure_pipeline().predict(input=image)
        blocks = self._blocks(results, dpi=dpi)
        ordered = tuple(
            sorted(blocks, key=lambda item: (item.bbox[1], item.bbox[0], item.text))
        )
        return PdfOcrResult(
            blocks=ordered,
            dpi=dpi,
            mean_confidence=(
                fmean(block.confidence for block in ordered) if ordered else 0.0
            ),
            engine=self.name,
        )

    @classmethod
    def _blocks(cls, results: Iterable[Any], *, dpi: int) -> list[PdfOcrBlock]:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Paddle IR OCR requires numpy") from exc
        scale = 72.0 / dpi
        blocks: list[PdfOcrBlock] = []
        for result in results:
            payload = cls._payload(result)
            texts = list(payload.get("rec_texts", ()))
            scores = list(payload.get("rec_scores", ()))
            boxes = list(payload.get("rec_boxes", ()))
            if not boxes and payload.get("rec_polys") is not None:
                boxes = list(payload["rec_polys"])
            for text, score, box in zip(texts, scores, boxes, strict=False):
                normalized = str(text).strip()
                if not normalized:
                    continue
                array = np.asarray(box, dtype=float)
                if array.ndim == 1 and array.size >= 4:
                    x0, y0, x1, y1 = map(float, array[:4])
                elif array.ndim == 2 and array.shape[1] >= 2:
                    x0 = float(array[:, 0].min())
                    y0 = float(array[:, 1].min())
                    x1 = float(array[:, 0].max())
                    y1 = float(array[:, 1].max())
                else:
                    continue
                blocks.append(
                    PdfOcrBlock(
                        text=normalized,
                        bbox=(x0 * scale, y0 * scale, x1 * scale, y1 * scale),
                        confidence=max(0.0, min(1.0, float(score))),
                    )
                )
        return blocks

    @staticmethod
    def _payload(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            payload: Any = result
        elif hasattr(result, "json"):
            payload = result.json
            if callable(payload):
                payload = payload()
        elif hasattr(result, "res"):
            payload = result.res
        else:
            try:
                payload = dict(result)
            except (TypeError, ValueError) as exc:
                raise ValueError("unsupported PaddleOCR result object") from exc
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            raise ValueError("unsupported PaddleOCR result payload")
        nested = payload.get("res", payload)
        if not isinstance(nested, dict):
            raise ValueError("PaddleOCR result has no mapping payload")
        return nested

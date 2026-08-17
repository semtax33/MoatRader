from __future__ import annotations

import gzip
import io
import json
import threading
import time
from dataclasses import dataclass
from email.message import Message
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class HttpResponse:
    url: str
    status_code: int
    headers: dict[str, str]
    content: bytes

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8-sig"))


class HttpClient(Protocol):
    def get(
        self,
        url: str,
        *,
        query: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        max_bytes: int | None = None,
    ) -> HttpResponse: ...

    def post_form(
        self,
        url: str,
        *,
        form: dict[str, Any],
        headers: dict[str, str] | None = None,
        max_bytes: int | None = None,
    ) -> HttpResponse: ...


class HttpRequestError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RequestRateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.interval = 1.0 / requests_per_second
        self._next_at = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_at - now)
            if delay:
                time.sleep(delay)
            self._next_at = max(now, self._next_at) + self.interval


class ResilientHttpClient:
    """Small dependency-free HTTP client with rate limiting and bounded retries."""

    def __init__(
        self,
        *,
        user_agent: str,
        requests_per_second: float,
        timeout_seconds: float = 60.0,
        max_retries: int = 4,
        default_max_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("user_agent is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if default_max_bytes <= 0:
            raise ValueError("default_max_bytes must be positive")
        self.user_agent = user_agent.strip()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.default_max_bytes = default_max_bytes
        self.rate_limiter = RequestRateLimiter(requests_per_second)

    def get(
        self,
        url: str,
        *,
        query: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        max_bytes: int | None = None,
    ) -> HttpResponse:
        request_url = self._with_query(url, query)
        return self._request(
            request_url,
            method="GET",
            headers=headers,
            max_bytes=max_bytes,
        )

    def post_form(
        self,
        url: str,
        *,
        form: dict[str, Any],
        headers: dict[str, str] | None = None,
        max_bytes: int | None = None,
    ) -> HttpResponse:
        body = urlencode(
            [(key, str(value)) for key, value in form.items() if value is not None]
        ).encode("utf-8")
        return self._request(
            url,
            method="POST",
            body=body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                **(headers or {}),
            },
            max_bytes=max_bytes,
        )

    def _request(
        self,
        request_url: str,
        *,
        method: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        max_bytes: int | None = None,
    ) -> HttpResponse:
        safe_url = self.redact_url(request_url)
        limit = max_bytes or self.default_max_bytes
        request_headers = {
            "User-Agent": self.user_agent,
            "Accept": "*/*",
            "Accept-Encoding": "gzip",
            **(headers or {}),
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.rate_limiter.wait()
            try:
                request = Request(
                    request_url,
                    data=body,
                    headers=request_headers,
                    method=method,
                )
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    response_headers = self._headers(response.headers)
                    content = self._read_bounded(response, limit, safe_url)
                    if response_headers.get("content-encoding", "").lower() == "gzip":
                        with gzip.GzipFile(fileobj=io.BytesIO(content)) as compressed:
                            content = compressed.read(limit + 1)
                        if len(content) > limit:
                            raise HttpRequestError(f"decompressed response exceeds {limit} bytes: {safe_url}")
                    return HttpResponse(
                        url=safe_url,
                        status_code=int(getattr(response, "status", 200)),
                        headers=response_headers,
                        content=content,
                    )
            except HTTPError as exc:
                last_error = exc
                if attempt >= self.max_retries or not self._retryable(exc.code):
                    raise HttpRequestError(
                        f"HTTP {exc.code} while fetching {safe_url}", status_code=exc.code
                    ) from exc
                self._backoff(attempt, exc.headers)
            except (URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self._backoff(attempt, None)
        assert last_error is not None
        raise HttpRequestError(
            f"request failed after {self.max_retries + 1} attempt(s): {safe_url}: {last_error}"
        ) from last_error

    @staticmethod
    def _with_query(url: str, query: dict[str, Any] | None) -> str:
        if not query:
            return url
        parts = urlsplit(url)
        existing = parse_qsl(parts.query, keep_blank_values=True)
        values = existing + [(key, str(value)) for key, value in query.items() if value is not None]
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(values), parts.fragment))

    @staticmethod
    def redact_url(url: str) -> str:
        parts = urlsplit(url)
        secret_keys = {"crtfc_key", "api_key", "apikey", "token", "authorization"}
        query = [
            (key, "REDACTED" if key.lower() in secret_keys else value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
        ]
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))

    @staticmethod
    def _headers(headers: Message | Any) -> dict[str, str]:
        return {str(key).lower(): str(value) for key, value in headers.items()}

    @staticmethod
    def _read_bounded(response: Any, limit: int, safe_url: str) -> bytes:
        length = response.headers.get("Content-Length")
        if length:
            try:
                if int(length) > limit:
                    raise HttpRequestError(f"response Content-Length exceeds {limit} bytes: {safe_url}")
            except ValueError:
                pass
        content = response.read(limit + 1)
        if len(content) > limit:
            raise HttpRequestError(f"response exceeds {limit} bytes: {safe_url}")
        return content

    @staticmethod
    def _retryable(status_code: int) -> bool:
        return status_code in {408, 409, 425, 429} or status_code >= 500

    @staticmethod
    def _backoff(attempt: int, headers: Message | None) -> None:
        retry_after = headers.get("Retry-After") if headers else None
        try:
            delay = float(retry_after) if retry_after is not None else 1.5 * (2**attempt)
        except ValueError:
            delay = 1.5 * (2**attempt)
        time.sleep(min(30.0, max(0.0, delay)))

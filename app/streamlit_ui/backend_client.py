"""Thin clients share server predictions; no duplicate model or face crop code."""
import requests


def _result(response):
    result = response.json()
    if not 200 <= response.status_code < 300:
        raise ValueError(str(result.get("detail", "API request failed")))
    return result


def analyze_upload(base, data, content_type="image/png", *, explain=False, transport=requests):
    return _result(transport.post(base.rstrip("/") + "/analyze", data=data,
                   headers={"Content-Type": content_type}, params={"explain": str(explain).lower()}, timeout=90))


def analyze_url(base, url, *, explain=False, transport=requests):
    return _result(transport.post(base.rstrip("/") + "/analyze", json={"url": url, "explain": explain}, timeout=90))


def recent_history(base, *, transport=requests):
    return _result(transport.get(base.rstrip("/") + "/history", timeout=10))

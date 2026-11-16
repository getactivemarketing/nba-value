# backend/src/edge/social/poster.py
"""Where a rendered post goes: Blotato (X) or nowhere (dry run).

post() takes a thread: texts[i] is post i, images[i] its PNGs. A result that is not ok says
whether a retry is safe: a failure before Blotato's create-post call (an image upload) is
retryable; a failed or unanswered create-post call is not, because the post may exist anyway
and the owner's rule is never to double-post.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from src.services.social import blotato

log = logging.getLogger(__name__)


@dataclass
class PostResult:
    ok: bool
    submission_id: str | None
    media_urls: list[str]
    post_url: str | None
    retryable: bool = False
    error: str | None = None


@dataclass
class UrlLookup:
    """Blotato's view of a submission: state is published|failed|pending|unknown."""
    state: str
    url: str | None = None
    error: str | None = None


class Poster(Protocol):
    dry_run: bool

    def post(self, texts: list[str], images: list[list[bytes]]) -> PostResult: ...

    def lookup(self, submission_id: str) -> UrlLookup: ...


@dataclass
class DryRunPoster:
    """Never touches the network. The runner stores the rendered card with status dry_run."""
    dry_run: bool = True
    calls: list = field(default_factory=list)

    def post(self, texts: list[str], images: list[list[bytes]]) -> PostResult:
        self.calls.append((texts, images))
        return PostResult(ok=True, submission_id=None, media_urls=[], post_url=None)

    def lookup(self, submission_id: str) -> UrlLookup:
        return UrlLookup("unknown")


def _lookup(submission_id: str) -> UrlLookup:
    data = blotato.get_post_status(submission_id)
    if not data:
        return UrlLookup("unknown")
    status = str(data.get("status") or "")
    if status == "published":
        return UrlLookup("published", data.get("publicUrl") or None)
    if status == "failed":
        return UrlLookup("failed", error=str(data.get("errorMessage") or "")[:300])
    return UrlLookup("pending")


class BlotatoPoster:
    """Uploads every image, posts the thread immediately, then asks once for the public URL
    (usually still in progress; the runner looks it up again on later ticks)."""
    dry_run = False

    def post(self, texts: list[str], images: list[list[bytes]]) -> PostResult:
        per_post: list[list[str]] = []
        uploaded: list[str] = []
        for i, imgs in enumerate(images or [[] for _ in texts]):
            urls = []
            for j, png in enumerate(imgs or []):
                url = blotato.upload_media(png, filename=f"truline-{i}-{j}.png")
                if not url:
                    return PostResult(False, None, uploaded, None, retryable=True, error="media upload failed")
                urls.append(url)
                uploaded.append(url)
            per_post.append(urls)
        data = blotato.post_thread(texts, schedule_at=None, media_urls_per_post=per_post)
        if not data or data.get("dry_run") or not data.get("postSubmissionId"):
            return PostResult(False, None, uploaded, None, retryable=False,
                              error="create post failed or unanswered; not retried (may have posted)")
        sid = str(data["postSubmissionId"])
        found = _lookup(sid)
        return PostResult(True, sid, uploaded, found.url if found.state == "published" else None)

    def lookup(self, submission_id: str) -> UrlLookup:
        return _lookup(submission_id)

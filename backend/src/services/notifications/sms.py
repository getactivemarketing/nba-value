"""Textbelt SMS transport for founder notifications (plain httpx)."""

import os

import httpx
import structlog

logger = structlog.get_logger()

TEXTBELT_URL = "https://textbelt.com/text"
LOW_QUOTA_WARN = 20


def send_sms(body: str) -> bool:
    """Send an SMS to the founder via Textbelt. True only on confirmed success.

    Textbelt returns HTTP 200 even for failures — success is determined by the JSON
    ``success`` field. No-ops (returns False) when TEXTBELT_API_KEY or
    PICK_ALERT_TO_NUMBER is unset so local dev never attempts a send.
    """
    key = os.getenv("TEXTBELT_API_KEY")
    to_number = os.getenv("PICK_ALERT_TO_NUMBER")
    if not all([key, to_number]):
        logger.warning("sms_not_configured")
        return False

    try:
        resp = httpx.post(
            TEXTBELT_URL,
            data={"phone": to_number, "message": body, "key": key},
            timeout=15.0,
        )
    except httpx.HTTPError as e:
        logger.warning("sms_send_failed", error=str(e))
        return False

    if resp.status_code // 100 != 2:
        logger.warning("sms_send_failed", status=resp.status_code, detail=resp.text[:200])
        return False

    try:
        payload = resp.json()
    except ValueError:
        logger.warning("sms_send_failed", detail="non-JSON response")
        return False

    if not payload.get("success"):
        logger.warning("sms_send_failed", error=payload.get("error"))
        return False

    quota = payload.get("quotaRemaining")
    if isinstance(quota, int) and quota < LOW_QUOTA_WARN:
        logger.warning("sms_quota_low", quota_remaining=quota)
    return True

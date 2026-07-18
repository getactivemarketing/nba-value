"""Twilio SMS transport for founder notifications (no Twilio SDK — plain httpx)."""

import os

import httpx
import structlog

logger = structlog.get_logger()

TWILIO_MESSAGES_URL = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"


def send_sms(body: str) -> bool:
    """Send an SMS to the founder. Returns True only on a Twilio 2xx.

    No-ops (returns False) when any required env var is unset so local
    dev and misconfigured prod never attempt a send.
    """
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    to_number = os.getenv("PICK_ALERT_TO_NUMBER")
    if not all([sid, token, from_number, to_number]):
        logger.warning("sms_not_configured")
        return False

    try:
        resp = httpx.post(
            TWILIO_MESSAGES_URL.format(sid=sid),
            auth=(sid, token),
            data={"From": from_number, "To": to_number, "Body": body},
            timeout=15.0,
        )
    except httpx.HTTPError as e:
        logger.warning("sms_send_failed", error=str(e))
        return False

    if resp.status_code // 100 != 2:
        logger.warning("sms_send_failed", status=resp.status_code, detail=resp.text[:200])
        return False
    return True

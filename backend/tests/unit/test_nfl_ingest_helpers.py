"""DB-free unit tests for pure helpers in src.services.nfl.ingest."""
from src.services.nfl.ingest import _is_dome


def test_is_dome_dome():
    assert _is_dome("dome") is True


def test_is_dome_closed():
    assert _is_dome("closed") is True


def test_is_dome_outdoors():
    assert _is_dome("outdoors") is False


def test_is_dome_open():
    assert _is_dome("open") is False


def test_is_dome_none():
    assert _is_dome(None) is False


def test_is_dome_nan():
    # Regression: pandas NaN roof values must not raise AttributeError.
    assert _is_dome(float("nan")) is False


def test_is_dome_empty_string():
    assert _is_dome("") is False

"""Test-wide guard: the social runner never reaches ESPN (or anything else) from a test. Tests that
run worker.tick against a DB would otherwise refresh headshots over the network."""
import httpx
import pytest


@pytest.fixture(autouse=True)
def _no_social_network(monkeypatch):
    from src.edge.social import runner
    transport = httpx.MockTransport(lambda request: httpx.Response(404))
    monkeypatch.setattr(runner, "_http", lambda: httpx.AsyncClient(transport=transport))

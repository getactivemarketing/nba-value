"""The site calls the Railway API directly (not via Vercel's rewrite) so the
rate limiter sees each visitor's IP; the browser needs CORS for that."""
import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from src.config import Settings

# Production sets no CORS_ORIGINS, so test the shipped defaults, not a local .env.
_defaults = Settings(_env_file=None)
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=_defaults.cors_origins,
                   allow_origin_regex=_defaults.cors_origin_regex, allow_methods=["*"])


@app.get("/api/v1/edge/games")
def _games():
    return {}


class settings:  # noqa: N801 - keeps the request paths below readable
    api_v1_prefix = "/api/v1"


@pytest.mark.parametrize("origin", ["https://www.truline.app", "https://truline.app",
                                    "https://nba-git-feature-x-samir-patels-projects-5c7af3f7.vercel.app"])
def test_site_origins_may_call_the_api(origin):
    r = TestClient(app).options(f"{settings.api_v1_prefix}/edge/games", headers={
        "Origin": origin, "Access-Control-Request-Method": "GET"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == origin


def test_other_origins_are_not_allowed():
    r = TestClient(app).options(f"{settings.api_v1_prefix}/edge/games", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in r.headers

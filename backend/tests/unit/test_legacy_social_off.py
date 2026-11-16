from src import main
from src.config import settings, Settings


def test_legacy_social_thread_not_started_by_default(monkeypatch):
    monkeypatch.setattr(settings, "legacy_social_enabled", False)
    started = []
    monkeypatch.setattr(main, "_start_social_scheduler_thread", lambda: started.append(1))
    main._maybe_start_legacy_social()
    assert started == []


def test_legacy_social_thread_starts_only_when_flag_on(monkeypatch):
    monkeypatch.setattr(settings, "legacy_social_enabled", True)
    started = []
    monkeypatch.setattr(main, "_start_social_scheduler_thread", lambda: started.append(1))
    main._maybe_start_legacy_social()
    assert started == [1]


def test_social_defaults():
    s = Settings()
    assert s.social_dry_run is True and s.social_max_per_day == 25 and s.social_max_per_hour == 6
    assert s.social_show_records is True
    assert s.legacy_social_enabled is False
    assert s.social_sports == "nba"
    assert s.social_kinds == "cheatsheet,value,blind,final,receipt"
    assert s.site_url == "https://www.truline.app"

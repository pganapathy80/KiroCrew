"""Tests for the on-disk code channel record that survives a gateway restart."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kiro_crew.slack import code_channel_store


def _orch():
    return SimpleNamespace(
        _code_channels=set(),
        _code_channel_repo_by_id={},
        _code_channel_diff_base={},
        _code_channel_origin={},
        _code_channel_session_ts={},
    )


def _populated(channel="C_CC"):
    orch = _orch()
    orch._code_channels.add(channel)
    orch._code_channel_repo_by_id[channel] = "/work/repo"
    orch._code_channel_diff_base[channel] = "abc123"
    orch._code_channel_origin[channel] = ("C_MAIN", "1700.1")
    orch._code_channel_session_ts[channel] = "1700.2"
    return orch


class TestCodeChannelStore:
    def test_round_trip_restores_every_field(self, tmp_path):
        path = tmp_path / "store.json"
        code_channel_store.save_channel(_populated(), "C_CC", path=path)

        fresh = _orch()
        assert code_channel_store.restore_into(fresh, path=path) == 1
        assert fresh._code_channels == {"C_CC"}
        assert fresh._code_channel_repo_by_id == {"C_CC": "/work/repo"}
        assert fresh._code_channel_diff_base == {"C_CC": "abc123"}
        assert fresh._code_channel_origin == {"C_CC": ("C_MAIN", "1700.1")}
        assert fresh._code_channel_session_ts == {"C_CC": "1700.2"}

    def test_restore_mutates_in_place(self, tmp_path):
        """Holders of the set (the dashboard's code-channel check) stay valid."""
        path = tmp_path / "store.json"
        code_channel_store.save_channel(_populated(), "C_CC", path=path)
        fresh = _orch()
        held = fresh._code_channels.__contains__
        code_channel_store.restore_into(fresh, path=path)
        assert held("C_CC") is True

    def test_missing_file_restores_nothing(self, tmp_path):
        fresh = _orch()
        assert code_channel_store.restore_into(fresh, path=tmp_path / "absent.json") == 0
        assert fresh._code_channels == set()

    def test_corrupt_file_restores_nothing(self, tmp_path):
        path = tmp_path / "store.json"
        path.write_text("{not json", encoding="utf-8")
        fresh = _orch()
        assert code_channel_store.restore_into(fresh, path=path) == 0
        assert fresh._code_channels == set()

    def test_malformed_record_fields_are_skipped(self, tmp_path):
        path = tmp_path / "store.json"
        path.write_text(
            '{"channels": {"C_CC": {"repo": 7, "origin": ["only-one"]}, "C_BAD": "x"}}',
            encoding="utf-8",
        )
        fresh = _orch()
        assert code_channel_store.restore_into(fresh, path=path) == 1
        assert fresh._code_channels == {"C_CC"}
        assert fresh._code_channel_repo_by_id == {}
        assert fresh._code_channel_origin == {}

    def test_forget_removes_only_that_channel(self, tmp_path):
        path = tmp_path / "store.json"
        orch = _populated("C_ONE")
        orch._code_channels.add("C_TWO")
        code_channel_store.save_channel(orch, "C_ONE", path=path)
        code_channel_store.save_channel(orch, "C_TWO", path=path)
        code_channel_store.forget_channel("C_ONE", path=path)

        fresh = _orch()
        code_channel_store.restore_into(fresh, path=path)
        assert fresh._code_channels == {"C_TWO"}

    def test_untracked_channel_is_not_saved(self, tmp_path):
        path = tmp_path / "store.json"
        code_channel_store.save_channel(_orch(), "C_NOPE", path=path)
        assert not path.exists()

    def test_default_path_is_under_the_data_home(self, tmp_path, monkeypatch):
        monkeypatch.setattr(code_channel_store, "config_dir", lambda: tmp_path)
        code_channel_store.save_channel(_populated(), "C_CC")
        assert (tmp_path / "slack-code-channels.json").exists()


class TestCodeChannelDetectionCache:
    """A failed is_code_channel lookup must not be cached as 'ordinary channel'."""

    @pytest.mark.asyncio
    async def test_failed_lookup_is_retried_definite_answers_are_cached(self, monkeypatch):
        from kiro_crew.slack import events

        monkeypatch.setattr(events, "_CODE_CHANNEL_CACHE", {})
        slack = SimpleNamespace(is_code_channel=AsyncMock(side_effect=[None, True, False]))
        orch = SimpleNamespace(slack=slack)

        # A failed lookup answers "not a code channel" for now and is not cached.
        assert await events._detect_code_channel(orch, "C_CC") is False
        assert "C_CC" not in events._CODE_CHANNEL_CACHE
        # The retry succeeds and is cached: no further Slack call for this channel.
        assert await events._detect_code_channel(orch, "C_CC") is True
        assert await events._detect_code_channel(orch, "C_CC") is True
        # An ordinary channel's definite "no" is cached too (one lookup per channel).
        assert await events._detect_code_channel(orch, "C_MAIN") is False
        assert await events._detect_code_channel(orch, "C_MAIN") is False
        assert slack.is_code_channel.await_count == 3


class TestCodeChannelStoreIsProtected:
    """The record's origin is where the gateway posts back to, tracked or not, so the
    agent must not be able to forge it, through the file tools or the shell."""

    def test_file_tools_are_refused(self):
        from kiro_crew import security

        protected = set(security.write_protected_home_paths())
        for prefix in security.crew_home_prefixes():
            assert f"{prefix}/{code_channel_store._STORE_FILENAME}" in protected

    def test_file_tools_may_still_read_it(self):
        # It holds no secret: only the write is refused.
        from kiro_crew import security

        for prefix in security.crew_home_prefixes():
            path = f"~/{prefix}/{code_channel_store._STORE_FILENAME}"
            assert security.is_sensitive_write_path(path) is True, path
            assert security.is_sensitive_path(path) is False, path

    def test_the_sandbox_seals_it_readonly_even_when_absent(self):
        # The kernel floor for a shell or interpreter write, which no text gate
        # sees. Pre-created because a mount seal cannot bind a path that is not
        # there, and the record does not exist until a code channel does.
        from kiro_crew import sandbox

        leaf = code_channel_store._STORE_FILENAME
        assert leaf in sandbox._CREW_READONLY_LEAVES
        assert leaf in sandbox._CREW_PRECREATE_READONLY_FILE_LEAVES
        assert leaf not in sandbox._CREW_SANDBOX_VISIBLE_LEAVES

    def test_an_empty_materialized_record_restores_nothing(self, tmp_path):
        # The precondition for pre-creating it: the seal's ``{}`` stub must read
        # as "no code channels", exactly like an absent file.
        from kiro_crew import sandbox

        path = tmp_path / code_channel_store._STORE_FILENAME
        path.write_bytes(sandbox._EMPTY_CEILING_DOCUMENT)
        orch = _orch()
        assert code_channel_store.restore_into(orch, path=path) == 0
        assert orch._code_channels == set() and orch._code_channel_origin == {}


class TestArchiveAndRaces:
    def test_clear_channel_state_drops_the_channel_but_keeps_the_origin(self):
        orch = _populated()
        code_channel_store.clear_channel_state(orch, "C_CC")
        assert "C_CC" not in orch._code_channels
        assert "C_CC" not in orch._code_channel_repo_by_id
        assert "C_CC" not in orch._code_channel_diff_base
        assert "C_CC" not in orch._code_channel_session_ts
        # The archive path still needs it to post the summary back.
        assert orch._code_channel_origin["C_CC"] == ("C_MAIN", "1700.1")

    def test_save_after_archive_does_not_revive_the_record(self, tmp_path):
        path = tmp_path / "store.json"
        orch = _populated()
        code_channel_store.save_channel(orch, "C_CC", path=path)
        code_channel_store.clear_channel_state(orch, "C_CC")
        code_channel_store.forget_channel("C_CC", path=path)
        # A save that was waiting when the archive ran must not bring it back.
        code_channel_store.save_channel(orch, "C_CC", path=path)
        fresh = _orch()
        assert code_channel_store.restore_into(fresh, path=path) == 0

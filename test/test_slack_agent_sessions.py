"""Tests for the agent-session and code-channel client surface.

These pin the migration from the deprecated ``assistant.threads.*`` methods to
``agents.sessions.*`` (Slack sunsets the former in Feb 2027), and the
partner-beta ``agents.conversations.*`` code-channel calls. They exercise the
real client with a recording ``_web`` stand-in so the exact wire method and
payload are observable — the interface-level mocks elsewhere cannot see which
Slack method was actually called.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from kiro_crew.config.loader import KiroCrewConfig, MessagingConfig, SlackConfig
from kiro_crew.slack import client as slack_client
from kiro_crew.slack.client import RealSlackClient


class _RecordingWeb:
    """Records api_call(method, params=/json=) and answers conversations_info."""

    def __init__(self, *, info: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._info = info or {}
        self._responses: dict[str, dict[str, Any]] = {}

    def set_response(self, method: str, resp: dict[str, Any]) -> None:
        self._responses[method] = resp

    async def api_call(
        self,
        method: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((method, params if params is not None else (json or {})))
        return self._responses.get(method, {"ok": True})

    async def conversations_info(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("conversations.info", kwargs))
        return self._info


def _client(web: _RecordingWeb) -> RealSlackClient:
    c = RealSlackClient.__new__(RealSlackClient)
    c._web = web  # type: ignore[attr-defined]
    return c


class TestSessionStatusMigration:
    @pytest.mark.asyncio
    async def test_empty_status_maps_to_active(self) -> None:
        web = _RecordingWeb()
        await _client(web).set_thread_status("C1", "1700.1", "")
        method, params = web.calls[-1]
        assert method == "agents.sessions.setStatus"
        assert params["status"] == slack_client.SESSION_STATUS_ACTIVE
        assert params["channel_id"] == "C1"
        assert params["thread_ts"] == "1700.1"

    @pytest.mark.asyncio
    async def test_nonempty_status_maps_to_processing(self) -> None:
        web = _RecordingWeb()
        await _client(web).set_thread_status("C1", "1700.1", "is using bash")
        method, params = web.calls[-1]
        assert method == "agents.sessions.setStatus"
        assert params["status"] == slack_client.SESSION_STATUS_PROCESSING

    @pytest.mark.asyncio
    async def test_no_longer_calls_deprecated_assistant_threads(self) -> None:
        web = _RecordingWeb()
        await _client(web).set_thread_status("C1", "1700.1", "working")
        await _client(web).set_thread_title("C1", "1700.1", "My session")
        methods = [m for m, _ in web.calls]
        assert not any(m.startswith("assistant.threads") for m in methods)

    @pytest.mark.asyncio
    async def test_invalid_status_coerced_to_processing(self) -> None:
        web = _RecordingWeb()
        await _client(web).set_session_status("C1", "1700.1", "bogus")
        assert web.calls[-1][1]["status"] == slack_client.SESSION_STATUS_PROCESSING

    @pytest.mark.asyncio
    async def test_channel_session_omits_thread_ts(self) -> None:
        # A code channel is one session; Slack rejects thread_ts there.
        web = _RecordingWeb()
        await _client(web).set_session_status("C1", None, slack_client.SESSION_STATUS_PROCESSING)
        assert "thread_ts" not in web.calls[-1][1]

    @pytest.mark.asyncio
    async def test_rename_uses_agents_sessions_rename(self) -> None:
        web = _RecordingWeb()
        await _client(web).set_thread_title("C1", "1700.1", "Trip prep")
        method, params = web.calls[-1]
        assert method == "agents.sessions.rename"
        assert params["title"] == "Trip prep"


class TestCodeChannels:
    @pytest.mark.asyncio
    async def test_create_returns_channel_id_and_sends_origin(self) -> None:
        web = _RecordingWeb()
        web.set_response("agents.conversations.create", {"ok": True, "channel_id": "C999"})
        cid = await _client(web).create_code_channel(
            "Fix flaky login",
            session_id="ses_1",
            origin_channel_id="C1",
            origin_message_ts="1700.1",
        )
        assert cid == "C999"
        method, body = web.calls[-1]
        assert method == "agents.conversations.create"
        assert body["origin_channel_id"] == "C1"
        assert body["session_id"] == "ses_1"

    @pytest.mark.asyncio
    async def test_context_bar_trimmed_to_five(self) -> None:
        web = _RecordingWeb()
        items = [{"key": f"k{i}", "label": f"l{i}"} for i in range(8)]
        await _client(web).set_code_channel_properties("C999", items)
        _, body = web.calls[-1]
        assert len(body["code_channel"]["context_bar_items"]) == 5

    @pytest.mark.asyncio
    async def test_set_diff_view(self) -> None:
        web = _RecordingWeb()
        web.set_response("agents.conversations.setView", {"ok": True, "view_id": "T1"})
        resp = await _client(web).set_code_channel_view(
            "C999", view_type="diff", content="diff --git a b", base_branch="main"
        )
        assert resp and resp["view_id"] == "T1"
        _, body = web.calls[-1]
        assert body["type"] == "diff"
        assert body["base_branch"] == "main"

    @pytest.mark.asyncio
    async def test_archive_with_summary(self) -> None:
        web = _RecordingWeb()
        ok = await _client(web).archive_code_channel("C999", summary_message_ts="1700.9")
        assert ok is True
        _, body = web.calls[-1]
        assert body["summary_message_ts"] == "1700.9"

    @pytest.mark.asyncio
    async def test_is_code_channel_reads_record_type(self) -> None:
        web = _RecordingWeb(
            info={"channel": {"properties": {"record_channel": {"record_type": "agent_channel"}}}}
        )
        assert await _client(web).is_code_channel("C999") is True

    @pytest.mark.asyncio
    async def test_is_code_channel_false_for_plain_channel(self) -> None:
        web = _RecordingWeb(info={"channel": {"properties": {}}})
        assert await _client(web).is_code_channel("C1") is False


class TestCodeChannelsConfigFlag:
    def test_default_off(self) -> None:
        from kiro_crew.config.sections import SlackConfig

        assert SlackConfig().code_channels is False

    def test_loader_reads_flag(self, tmp_path: Any, monkeypatch: Any) -> None:
        import json

        from kiro_crew.config.loader import KiroCrewConfig

        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"slack": {"code_channels": True}}))
        monkeypatch.setattr("kiro_crew.config.loader.config_path", lambda: cfg_file)
        cfg = KiroCrewConfig.load()
        assert cfg.slack.code_channels is True


class _FakeSessions:
    def __init__(self, *, has: bool, thread_owners: dict[str, str] | None = None) -> None:
        self._has = has
        self._thread_owners = thread_owners or {}
        self.stopped: list[str] = []
        self.cleared: list[str] = []
        self.noted: list[str] = []

    def has_session(self, key: str) -> bool:
        return self._has

    def clear_queue(self, key: str) -> None:
        self.cleared.append(key)

    def get_session_for_thread(self, key: str) -> str | None:
        return self._thread_owners.get(key)

    def note_stop(self, key: str) -> bool:
        self.noted.append(key)
        return True

    async def stop_turn(self, key: str, **kwargs: Any) -> str:
        self.stopped.append(key)
        return "stopped"


class _FakeSlack:
    def __init__(self, *, created: str | None = "C999", invite_ok: bool = True) -> None:
        self.status_calls: list[tuple[str, str | None, str]] = []
        self.stopped_streams: list[tuple[str, str]] = []
        self._created = created
        self._invite_ok = invite_ok
        self.create_calls: list[str] = []
        self.invite_calls: list[tuple[str, list[str]]] = []
        self.posts: list[tuple[str, str]] = []
        self.block_posts: list[tuple[str, list[dict], str]] = []
        self.ctx_bar: list = []
        self.views: list[dict] = []
        self.archived: tuple | None = None
        self.rename_calls: list[tuple[str, str | None, str]] = []
        self.is_code_channel_result = True

    async def set_session_status(self, channel: str, thread_ts: str | None, status: str) -> None:
        self.status_calls.append((channel, thread_ts, status))

    async def stop_stream(self, channel: str, ts: str, final_text: str | None = None) -> bool:
        self.stopped_streams.append((channel, ts))
        return True

    async def create_code_channel(self, name: str, **kwargs: Any) -> str | None:
        self.create_calls.append(name)
        return self._created

    async def invite_users(self, channel_id: str, user_ids: list[str]) -> dict[str, Any]:
        self.invite_calls.append((channel_id, list(user_ids)))
        return {
            "ok": self._invite_ok,
            "invited": user_ids if self._invite_ok else [],
            "error": None if self._invite_ok else "cant_invite",
        }

    async def post_message(self, channel: str, text: str, *a: Any, **k: Any) -> str:
        self.posts.append((channel, text))
        return "1700.0"

    async def post_blocks(
        self, channel: str, blocks: list[dict], text: str, *a: Any, **k: Any
    ) -> str:
        self.block_posts.append((channel, blocks, text))
        return "1700.0"

    async def is_code_channel(self, channel_id: str) -> bool:
        return self.is_code_channel_result

    async def rename_session(self, channel: str, thread_ts: str | None, title: str) -> None:
        self.rename_calls.append((channel, thread_ts, title))

    async def set_code_channel_properties(self, channel_id: str, items: list) -> bool:
        self.ctx_bar = items
        return True

    async def set_code_channel_view(self, channel_id: str, **kw: Any) -> dict:
        self.views.append(kw)
        return {"view_id": "T1"}

    async def archive_code_channel(
        self, channel_id: str, summary_message_ts: str | None = None
    ) -> bool:
        self.archived = (channel_id, summary_message_ts)
        return True


class _FakeOrch:
    def __init__(
        self,
        *,
        has: bool,
        dm_single_session: bool = False,
        thread_owners: dict[str, str] | None = None,
        code_channels: bool = False,
        created: str | None = "C999",
        invitees: list[str] | None = None,
        invite_ok: bool = True,
    ) -> None:
        self.sessions = _FakeSessions(has=has, thread_owners=thread_owners)
        self.slack = _FakeSlack(created=created, invite_ok=invite_ok)
        self._session_tasks: dict[str, Any] = {}
        self._pending_queue: dict[str, Any] = {}
        self._handler_tasks: set[Any] = set()
        self._code_channels: set[str] = set()
        self._last_channel_id = ""
        self.slack_command = "kirocrew"
        # A single-session DM needs both the slack flag and the transport path.
        self._cfg = KiroCrewConfig(
            slack=SlackConfig(
                dm_single_session=dm_single_session,
                code_channels=code_channels,
                code_channel_invitees=invitees or [],
                code_channel_repo="",
                code_channel_context_items=[],
            ),
            messaging=MessagingConfig(use_transport=dm_single_session),
        )


class TestAgentSessionStopped:
    """Slack's stop event carries channel, thread_ts, user and streaming_message_ts."""

    @pytest.fixture(autouse=True)
    def _allowed(self, monkeypatch: Any) -> None:
        from kiro_crew.slack import events

        monkeypatch.setattr(events, "is_allowed_user", lambda uid: uid == "U1")

    @pytest.mark.asyncio
    async def test_thread_stop_cancels_and_finishes_streams(self) -> None:
        from kiro_crew.slack import events

        orch = _FakeOrch(has=True)
        await events._handle_agent_session_stopped(
            orch,
            {
                "channel": "C1",
                "thread_ts": "1700.1",
                "user": "U1",
                "streaming_message_ts": ["1700.5"],
            },
        )
        assert orch.sessions.stopped == ["1700.1"]
        assert orch.slack.stopped_streams == [("C1", "1700.5")]
        # Slack updates the session status itself after a stop.
        assert orch.slack.status_calls == []

    @pytest.mark.asyncio
    async def test_session_without_thread_keys_on_its_first_stream(self) -> None:
        from kiro_crew.slack import events

        orch = _FakeOrch(has=True)
        await events._handle_agent_session_stopped(
            orch, {"channel": "C1", "user": "U1", "streaming_message_ts": ["1700.2"]}
        )
        assert orch.sessions.stopped == ["1700.2"]
        assert orch.slack.status_calls == []

    @pytest.mark.asyncio
    async def test_unauthorized_user_cannot_stop(self, monkeypatch: Any) -> None:
        from kiro_crew.slack import events

        logged: list[dict] = []

        class _Sel:
            def log_api_access(self, **kw: Any) -> None:
                logged.append(kw)

        monkeypatch.setattr(events, "sel", lambda: _Sel())
        orch = _FakeOrch(has=True)
        await events._handle_agent_session_stopped(
            orch,
            {
                "channel": "C1",
                "thread_ts": "1700.1",
                "user": "U_STRANGER",
                "streaming_message_ts": ["1700.5"],
            },
        )
        assert orch.sessions.stopped == []
        assert orch.slack.stopped_streams == []
        assert logged and logged[-1]["outcome"] == "denied"
        assert logged[-1]["error"] == "unauthorized sender"

    @pytest.mark.asyncio
    async def test_code_channel_stop_targets_the_channel_session(self) -> None:
        from kiro_crew.slack import events

        # Every message in a code channel runs under the channel's session anchor,
        # so the stop button must stop that session, not the clicked message's ts.
        orch = _FakeOrch(has=True, code_channels=True)
        orch._code_channels = {"C_CC"}
        orch._code_channel_session_ts = {"C_CC": "1700.1"}
        await events._handle_agent_session_stopped(
            orch, {"channel": "C_CC", "user": "U1", "streaming_message_ts": ["1700.9"]}
        )
        assert orch.sessions.stopped == ["1700.1"]
        assert orch.sessions.cleared == ["1700.1"]
        assert orch.slack.stopped_streams == [("C_CC", "1700.9")]

    @pytest.mark.asyncio
    async def test_code_channel_stop_without_streams_still_stops_the_anchor(self) -> None:
        """A code channel's session is the channel, so a stop that names no stream
        still stops the anchored session rather than being dropped."""
        from kiro_crew.slack import events

        orch = _FakeOrch(has=True, code_channels=True)
        orch._code_channels = {"C_CC"}
        orch._code_channel_session_ts = {"C_CC": "1700.1"}
        await events._handle_agent_session_stopped(orch, {"channel": "C_CC", "user": "U1"})
        assert orch.sessions.stopped == ["1700.1"]

    def test_bang_stop_in_code_channel_targets_the_anchor(self) -> None:
        """``!stop`` resolves through the same helper, so a ``!stop`` anywhere in a
        code channel stops the channel's session and acks where it was sent."""
        from kiro_crew.slack import events

        orch = _FakeOrch(has=True, code_channels=True)
        orch._code_channels = {"C_CC"}
        orch._code_channel_session_ts = {"C_CC": "1700.1"}
        assert events._resolve_stop_target(orch, "C_CC", None, "1700.8") == ("1700.1", "1700.8")
        # Threaded turns run under the anchor too, so a threaded stop does as well,
        # and acks inside its thread.
        assert events._resolve_stop_target(orch, "C_CC", "1700.5", "1700.8") == (
            "1700.1",
            "1700.5",
        )
        # A code channel with no anchor yet, and any other channel, keep the
        # message's own key.
        orch._code_channel_session_ts = {}
        assert events._resolve_stop_target(orch, "C_CC", None, "1700.8") == ("1700.8", "1700.8")
        assert events._resolve_stop_target(orch, "C1", None, "1700.8") == ("1700.8", "1700.8")

    @pytest.mark.asyncio
    async def test_no_active_session_is_a_noop(self) -> None:
        from kiro_crew.slack import events

        orch = _FakeOrch(has=False)
        await events._handle_agent_session_stopped(
            orch, {"channel": "C1", "thread_ts": "1700.1", "user": "U1"}
        )
        assert orch.sessions.stopped == []
        assert orch.slack.status_calls == []

    @pytest.mark.asyncio
    async def test_single_session_dm_stops_the_flat_dm_key(self) -> None:
        """A single-session DM runs every turn under ``slack:<channel>``, so the
        stop button stops that key, the one ``!stop`` stops, and never the bare
        stream ts its event carries."""
        from kiro_crew.slack import events

        orch = _FakeOrch(has=True, dm_single_session=True)
        await events._handle_agent_session_stopped(
            orch, {"channel": "D1", "user": "U1", "streaming_message_ts": ["1700.2"]}
        )
        assert orch.sessions.stopped == ["slack:D1"]
        assert orch.sessions.cleared == ["slack:D1"]
        assert orch.sessions.noted == ["slack:D1"]
        assert orch.slack.stopped_streams == [("D1", "1700.2")]
        assert orch.slack.status_calls == []

    @pytest.mark.asyncio
    async def test_dashboard_linked_thread_stops_its_owner_session(self) -> None:
        """A DM thread that a dashboard send-to-Slack owns runs its turn under
        that owner, so the stop button stops the owner, as ``!stop`` does."""
        from kiro_crew.slack import events

        orch = _FakeOrch(
            has=True, dm_single_session=True, thread_owners={"1700.1": "dashboard:chat-7"}
        )
        await events._handle_agent_session_stopped(
            orch,
            {
                "channel": "D1",
                "thread_ts": "1700.1",
                "user": "U1",
                "streaming_message_ts": ["1700.5"],
            },
        )
        assert orch.sessions.stopped == ["dashboard:chat-7"]
        assert orch.sessions.cleared == ["dashboard:chat-7"]
        assert orch.sessions.noted == ["dashboard:chat-7"]
        assert orch.slack.stopped_streams == [("D1", "1700.5")]
        assert orch.slack.status_calls == []

    @pytest.mark.asyncio
    async def test_flat_dm_root_stop_without_thread_or_stream(self) -> None:
        """A single-session DM turn posted at channel root has no thread_ts and no
        stream, so the stop event carries neither; it still stops the flat
        ``slack:<channel>`` key rather than doing nothing."""
        from kiro_crew.slack import events

        orch = _FakeOrch(has=True, dm_single_session=True)
        await events._handle_agent_session_stopped(orch, {"channel": "D1", "user": "U1"})
        assert orch.sessions.stopped == ["slack:D1"]
        assert orch.sessions.noted == ["slack:D1"]
        assert orch.slack.stopped_streams == []
        assert orch.slack.status_calls == []

    @pytest.mark.asyncio
    async def test_threadless_stop_outside_a_flat_dm_addresses_nothing(self) -> None:
        from kiro_crew.slack import events

        orch = _FakeOrch(has=True)
        await events._handle_agent_session_stopped(orch, {"channel": "C1", "user": "U1"})
        assert orch.sessions.stopped == []
        assert orch.sessions.noted == []


class TestAgentSessionEventOrigin:
    """Lifecycle events pass the workspace origin gate before they are handled.

    ``team_id`` rides on the enclosing ``event_callback`` envelope, which is the
    Socket Mode request payload the dispatcher receives.
    """

    @pytest.fixture(autouse=True)
    def _allowed_user(self, monkeypatch: Any) -> None:
        from kiro_crew.slack import events

        monkeypatch.setattr(events, "is_allowed_user", lambda uid: uid == "U1")

    @pytest.fixture
    def logged(self, monkeypatch: Any) -> list[dict]:
        from kiro_crew.slack import events

        records: list[dict] = []

        class _Sel:
            def log_api_access(self, **kw: Any) -> None:
                records.append(kw)

        monkeypatch.setattr(events, "sel", lambda: _Sel())
        return records

    @staticmethod
    def _allowlist(monkeypatch: Any, team_ids: set[str] | None) -> None:
        from kiro_crew.slack import enterprise

        monkeypatch.setattr(enterprise, "_allowlist_configured", team_ids is not None)
        monkeypatch.setattr(enterprise, "_allowed_team_ids", set(team_ids or ()))

    @staticmethod
    async def _dispatch(orch: _FakeOrch, team_id: str, event: dict[str, Any]) -> None:
        import asyncio

        from kiro_crew.slack import events

        events._dispatch_agent_session_event(orch, {"team_id": team_id, "event": event}, event)
        await asyncio.gather(*list(orch._handler_tasks), return_exceptions=True)

    _STOP = {
        "type": "agent_session_stopped",
        "channel": "C1",
        "thread_ts": "1700.1",
        "user": "U1",
        "streaming_message_ts": ["1700.5"],
    }

    @pytest.mark.asyncio
    async def test_disallowed_workspace_stop_is_refused(
        self, monkeypatch: Any, logged: list[dict]
    ) -> None:
        self._allowlist(monkeypatch, {"T_OK"})
        orch = _FakeOrch(has=True)
        await self._dispatch(orch, "T_GONE", dict(self._STOP))
        assert orch.sessions.stopped == []
        assert orch.sessions.noted == []
        assert orch.slack.stopped_streams == []
        assert logged and logged[-1]["outcome"] == "denied"
        assert logged[-1]["operation"] == "slack.agent_session_stopped"
        assert logged[-1]["error"] == "enterprise_origin_mismatch"

    @pytest.mark.asyncio
    async def test_missing_team_id_is_refused(self, monkeypatch: Any, logged: list[dict]) -> None:
        # Refused even default-open, as a message with no workspace id is.
        self._allowlist(monkeypatch, None)
        orch = _FakeOrch(has=True)
        await self._dispatch(orch, "", dict(self._STOP))
        assert orch.sessions.stopped == []
        assert logged[-1]["outcome"] == "denied"
        assert logged[-1]["error"] == "missing_team_id"

    @pytest.mark.asyncio
    async def test_envelope_team_id_wins_over_the_event_copy(
        self, monkeypatch: Any, logged: list[dict]
    ) -> None:
        self._allowlist(monkeypatch, {"T_OK"})
        orch = _FakeOrch(has=True)
        await self._dispatch(orch, "T_GONE", dict(self._STOP, team_id="T_OK"))
        assert orch.sessions.stopped == []
        assert logged[-1]["error"] == "enterprise_origin_mismatch"

    @pytest.mark.asyncio
    async def test_allowed_workspace_still_stops(
        self, monkeypatch: Any, logged: list[dict]
    ) -> None:
        self._allowlist(monkeypatch, {"T_OK"})
        orch = _FakeOrch(has=True)
        await self._dispatch(orch, "T_OK", dict(self._STOP))
        assert orch.sessions.stopped == ["1700.1"]
        assert orch.slack.stopped_streams == [("C1", "1700.5")]
        assert logged[-1]["outcome"] == "allowed"

    @pytest.mark.asyncio
    async def test_default_open_without_allowlist_still_stops(
        self, monkeypatch: Any, logged: list[dict]
    ) -> None:
        self._allowlist(monkeypatch, None)
        orch = _FakeOrch(has=True)
        await self._dispatch(orch, "T_ANY", dict(self._STOP))
        assert orch.sessions.stopped == ["1700.1"]

    _CODE_CHANNEL_STOP = {
        "type": "agent_session_stopped",
        "channel": "C_CC",
        "user": "U1",
        "streaming_message_ts": ["1700.9"],
    }

    @staticmethod
    def _code_channel_orch() -> _FakeOrch:
        orch = _FakeOrch(has=True, code_channels=True)
        orch._code_channels = {"C_CC"}
        orch._code_channel_session_ts = {"C_CC": "1700.1"}  # type: ignore[attr-defined]
        return orch

    @pytest.mark.asyncio
    async def test_disallowed_workspace_stop_in_a_code_channel_stops_nothing(
        self, monkeypatch: Any, logged: list[dict]
    ) -> None:
        """A code channel's stop resolves to the channel's whole session, so the
        origin gate must hold there too: nothing is stopped, cleared or finished."""
        self._allowlist(monkeypatch, {"T_OK"})
        orch = self._code_channel_orch()
        await self._dispatch(orch, "T_GONE", dict(self._CODE_CHANNEL_STOP))
        assert orch.sessions.stopped == []
        assert orch.sessions.cleared == []
        assert orch.sessions.noted == []
        assert orch.slack.stopped_streams == []
        assert logged[-1]["outcome"] == "denied"
        assert logged[-1]["error"] == "enterprise_origin_mismatch"

    @pytest.mark.asyncio
    async def test_allowed_workspace_stop_in_a_code_channel_stops_the_anchor(
        self, monkeypatch: Any, logged: list[dict]
    ) -> None:
        self._allowlist(monkeypatch, {"T_OK"})
        orch = self._code_channel_orch()
        await self._dispatch(orch, "T_OK", dict(self._CODE_CHANNEL_STOP))
        assert orch.sessions.stopped == ["1700.1"]
        assert orch.slack.stopped_streams == [("C_CC", "1700.9")]

    @pytest.mark.asyncio
    async def test_disallowed_workspace_title_change_is_refused(
        self, monkeypatch: Any, logged: list[dict]
    ) -> None:
        self._allowlist(monkeypatch, {"T_OK"})
        orch = _FakeOrch(has=True)
        event = {
            "type": "agent_session_title_changed",
            "channel": "C1",
            "thread_ts": "1700.1",
            "user": "U1",
            "title": "x",
        }
        await self._dispatch(orch, "T_GONE", event)
        assert [r["outcome"] for r in logged] == ["denied"]
        assert logged[0]["operation"] == "slack.agent_session_title_changed"


class TestCodeChannelSlashCommand:
    @pytest.mark.asyncio
    async def test_flag_off_refuses_without_creating(self) -> None:
        from kiro_crew.slack import events

        orch = _FakeOrch(has=False, code_channels=False)
        replies: list[str] = []
        await events._handle_codechannel(
            orch, "U1", "Fix login", lambda t, **k: _append(replies, t)
        )
        assert orch.slack.create_calls == []
        assert "off" in replies[0].lower()

    @pytest.mark.asyncio
    async def test_flag_on_creates_and_links(self) -> None:
        from kiro_crew.slack import events

        orch = _FakeOrch(has=False, code_channels=True, created="C42")
        replies: list[str] = []
        await events._handle_codechannel(
            orch, "U1", "Fix login", lambda t, **k: _append(replies, t)
        )
        assert orch.slack.create_calls == ["Fix login"]
        assert "<#C42>" in replies[0]

    @pytest.mark.asyncio
    async def test_invites_caller_and_configured_collaborators(self) -> None:
        from kiro_crew.slack import events

        # A collaborator agent bot configured as an auto-invitee.
        orch = _FakeOrch(has=False, code_channels=True, created="C42", invitees=["UAGENT"])
        replies: list[str] = []
        await events._handle_codechannel(
            orch, "UHUMAN", "Investigate alarm", lambda t, **k: _append(replies, t)
        )
        # One invite call to the created channel with caller + collaborator, deduped.
        assert orch.slack.invite_calls == [("C42", ["UHUMAN", "UAGENT"])]
        # A Block Kit kickoff tags both participants inside the channel, and carries
        # an "Archive with summary" action button.
        assert orch.slack.block_posts and orch.slack.block_posts[0][0] == "C42"
        _blocks, _fallback = orch.slack.block_posts[0][1], orch.slack.block_posts[0][2]
        assert "<@UHUMAN>" in _fallback and "<@UAGENT>" in _fallback
        _action_ids = [
            e.get("action_id")
            for b in _blocks
            if b.get("type") == "actions"
            for e in b.get("elements", [])
        ]
        assert "code_channel_archive" in _action_ids

    @pytest.mark.asyncio
    async def test_caller_not_duplicated_when_also_in_invitees(self) -> None:
        from kiro_crew.slack import events

        orch = _FakeOrch(
            has=False, code_channels=True, created="C42", invitees=["UHUMAN", "UAGENT"]
        )
        replies: list[str] = []
        await events._handle_codechannel(
            orch, "UHUMAN", "Investigate", lambda t, **k: _append(replies, t)
        )
        assert orch.slack.invite_calls == [("C42", ["UHUMAN", "UAGENT"])]

    @pytest.mark.asyncio
    async def test_invite_failure_is_surfaced_but_channel_still_created(self) -> None:
        from kiro_crew.slack import events

        orch = _FakeOrch(
            has=False, code_channels=True, created="C42", invitees=["UAGENT"], invite_ok=False
        )
        replies: list[str] = []
        await events._handle_codechannel(
            orch, "UHUMAN", "Investigate", lambda t, **k: _append(replies, t)
        )
        assert "<#C42>" in replies[0]  # channel still created
        assert "couldn't auto-invite" in orch.slack.block_posts[0][2].lower()

    @pytest.mark.asyncio
    async def test_flag_on_reports_creation_failure(self) -> None:
        from kiro_crew.slack import events

        orch = _FakeOrch(has=False, code_channels=True, created=None)
        replies: list[str] = []
        await events._handle_codechannel(
            orch, "U1", "Fix login", lambda t, **k: _append(replies, t)
        )
        assert "could not create" in replies[0].lower()

    @pytest.mark.asyncio
    async def test_flag_on_empty_name_shows_usage(self) -> None:
        from kiro_crew.slack import events

        orch = _FakeOrch(has=False, code_channels=True)
        replies: list[str] = []
        await events._handle_codechannel(orch, "U1", "  ", lambda t, **k: _append(replies, t))
        assert orch.slack.create_calls == []
        assert "usage" in replies[0].lower()


async def _append(sink: list[str], text: str) -> None:
    sink.append(text)


def _code_channels_cfg(*, enabled: bool = True, repo: str = "") -> KiroCrewConfig:
    return KiroCrewConfig(slack=SlackConfig(code_channels=enabled, code_channel_repo=repo))


def _fake_git(monkeypatch: Any, events: Any, *, diff: str = "") -> list[list[str]]:
    """Stand in for the git subprocess the chrome runs; records each argv."""
    calls: list[list[str]] = []

    def _run(argv: list[str], **_: Any) -> SimpleNamespace:
        calls.append(argv)
        sub = argv[3:]
        if sub[:1] == ["rev-parse"]:
            out = "main\n"
        elif sub[:1] == ["diff"]:
            out = diff
        else:
            out = ""
        return SimpleNamespace(stdout=out, returncode=0)

    monkeypatch.setattr(events.subprocess, "run", _run)
    return calls


def _repo_dir(tmp_path: Any) -> str:
    repo = tmp_path / "proj"
    (repo / ".git").mkdir(parents=True)
    return str(repo)


class TestCodeChannelChrome:
    @pytest.mark.asyncio
    async def test_posts_context_bar_and_diff(self, tmp_path: Any, monkeypatch: Any) -> None:
        from kiro_crew.slack import events

        calls = _fake_git(monkeypatch, events, diff="-x = 1\n+x = 2\n")
        slack = _FakeSlack()
        orch = SimpleNamespace(
            slack=slack,
            _cfg=SimpleNamespace(slack=SimpleNamespace(code_channel_context_items=[])),
        )
        repo = _repo_dir(tmp_path)
        await events._post_code_channel_chrome(orch, "C42", repo)
        keys = [i["key"] for i in slack.ctx_bar]
        assert "repo" in keys and "branch" in keys
        assert slack.views and slack.views[0]["view_type"] == "diff"
        assert "x = 2" in slack.views[0]["content"]
        # Every git call runs against the channel's repo, never the process cwd.
        assert calls and all(c[:3] == ["git", "-C", repo] for c in calls)

    @pytest.mark.asyncio
    async def test_configured_context_items_appended_after_repo_branch(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from kiro_crew.slack import events

        _fake_git(monkeypatch, events)
        slack = _FakeSlack()
        # Three configured items would push the total to 5 (repo + branch + 3).
        extra = [
            {"key": "app", "label": "Live app", "icon": "globe", "url": "https://example.test"},
            {"key": "ci", "label": "CI: green", "icon": "terminal"},
            {
                "key": "pr",
                "label": "PR #7",
                "icon": "hierarchy",
                "url": "https://example.test/pr/7",
            },
        ]
        orch = SimpleNamespace(
            slack=slack,
            _cfg=SimpleNamespace(slack=SimpleNamespace(code_channel_context_items=extra)),
        )
        await events._post_code_channel_chrome(orch, "C42", _repo_dir(tmp_path), with_diff=False)
        keys = [i["key"] for i in slack.ctx_bar]
        # repo/branch first, then the configured items, in order.
        assert keys == ["repo", "branch", "app", "ci", "pr"]
        assert slack.ctx_bar[2]["url"] == "https://example.test"

    @pytest.mark.asyncio
    async def test_no_repo_is_noop(self) -> None:
        from kiro_crew.slack import events

        slack = _FakeSlack()
        orch = SimpleNamespace(slack=slack)
        await events._post_code_channel_chrome(orch, "C42", "/nonexistent/path/xyz")
        assert slack.ctx_bar == [] and slack.views == []


class TestArchiveCommand:
    @pytest.mark.asyncio
    async def test_archive_posts_summary_and_archives(self) -> None:
        from kiro_crew.slack import events

        orch = _FakeOrch(has=False, code_channels=True)
        orch._last_channel_id = "C42"  # type: ignore[attr-defined]
        replies: list[str] = []
        await events._handle_archive_codechannel(
            orch, "U1", "all done", lambda t, **k: _append(replies, t)
        )
        assert orch.slack.archived == ("C42", "1700.0")
        assert any("all done" in p[1] for p in orch.slack.posts)

    @pytest.mark.asyncio
    async def test_archive_off_when_flag_off(self) -> None:
        from kiro_crew.slack import events

        orch = _FakeOrch(has=False, code_channels=False)
        orch._last_channel_id = "C42"  # type: ignore[attr-defined]
        replies: list[str] = []
        await events._handle_archive_codechannel(orch, "U1", "", lambda t, **k: _append(replies, t))
        assert orch.slack.archived is None and "off" in replies[0].lower()


class TestRenameCommand:
    @pytest.mark.asyncio
    async def test_rename_renames_code_channel_without_thread_ts(self) -> None:
        from kiro_crew.slack import events

        orch = _FakeOrch(has=False, code_channels=True)
        orch._last_channel_id = "C42"  # type: ignore[attr-defined]
        orch._owned_code_channels = {"C42"}  # type: ignore[attr-defined]
        orch.slack.is_code_channel_result = True
        replies: list[str] = []
        await events._handle_rename_codechannel(
            orch, "U1", "Fix the checkout 500s", lambda t, **k: _append(replies, t)
        )
        # thread_ts must be None for a code channel (thread_ts_not_allowed otherwise).
        assert orch.slack.rename_calls == [("C42", None, "Fix the checkout 500s")]
        assert "Fix the checkout 500s" in replies[0]

    @pytest.mark.asyncio
    async def test_rename_refuses_a_code_channel_another_agent_owns(self) -> None:
        """A code channel this Kiro Crew is not in charge of keeps the title its
        owner gave it."""
        from kiro_crew.slack import events

        orch = _FakeOrch(has=False, code_channels=True)
        orch._last_channel_id = "C42"  # type: ignore[attr-defined]
        orch._owned_code_channels = set()  # type: ignore[attr-defined]
        orch.slack.is_code_channel_result = True
        replies: list[str] = []
        await events._handle_rename_codechannel(
            orch, "U1", "New title", lambda t, **k: _append(replies, t)
        )
        assert orch.slack.rename_calls == []
        assert "another agent" in replies[0].lower()

    @pytest.mark.asyncio
    async def test_rename_is_inert_with_the_flag_off(self) -> None:
        from kiro_crew.slack import events

        orch = _FakeOrch(has=False, code_channels=False)
        orch._last_channel_id = "C42"  # type: ignore[attr-defined]
        orch._owned_code_channels = {"C42"}  # type: ignore[attr-defined]
        replies: list[str] = []
        await events._handle_rename_codechannel(
            orch, "U1", "New title", lambda t, **k: _append(replies, t)
        )
        assert orch.slack.rename_calls == []
        assert "off" in replies[0].lower()

    @pytest.mark.asyncio
    async def test_rename_refuses_a_non_code_channel(self) -> None:
        from kiro_crew.slack import events

        orch = _FakeOrch(has=False, code_channels=True)
        orch._last_channel_id = "C42"  # type: ignore[attr-defined]
        orch.slack.is_code_channel_result = False
        replies: list[str] = []
        await events._handle_rename_codechannel(
            orch, "U1", "New title", lambda t, **k: _append(replies, t)
        )
        assert orch.slack.rename_calls == []
        assert "isn't a code channel" in replies[0].lower()

    @pytest.mark.asyncio
    async def test_rename_empty_title_shows_usage(self) -> None:
        from kiro_crew.slack import events

        orch = _FakeOrch(has=False, code_channels=True)
        orch._last_channel_id = "C42"  # type: ignore[attr-defined]
        replies: list[str] = []
        await events._handle_rename_codechannel(
            orch, "U1", "   ", lambda t, **k: _append(replies, t)
        )
        assert orch.slack.rename_calls == []
        assert "usage" in replies[0].lower()


class TestContextItemSanitize:
    def test_drops_malformed_and_keeps_valid_items(self) -> None:
        from kiro_crew.config.section_builders import _sanitize_context_bar_items

        raw = [
            {"key": "app", "label": "Live", "icon": "globe", "url": "https://x.test"},
            {"key": "ci", "label": "CI", "icon": "terminal"},
            {"key": "bad", "label": "Bad", "icon": "not-an-icon"},  # invalid icon
            {"key": "", "label": "No key", "icon": "folder"},  # missing key
            {"label": "No key field", "icon": "folder"},  # missing key
            "not-a-dict",
        ]
        out = _sanitize_context_bar_items(raw)
        assert [i["key"] for i in out] == ["app", "ci"]
        assert out[0]["url"] == "https://x.test"
        assert "url" not in out[1]

    def test_none_yields_empty(self) -> None:
        from kiro_crew.config.section_builders import _sanitize_context_bar_items

        assert _sanitize_context_bar_items(None) == []


class TestCodeChannelArchiveButton:
    @pytest.mark.asyncio
    async def test_archive_button_runs_archive_path(self) -> None:
        from kiro_crew.slack import interactions

        slack = _FakeSlack()
        orch = SimpleNamespace(slack=slack, _code_channels={"C42"}, _cfg=_code_channels_cfg())
        _prev = interactions._orch
        interactions._orch = orch  # type: ignore[assignment]
        try:
            await interactions._handle_code_channel_archive({"value": "C42"}, "C42", "U1")
        finally:
            interactions._orch = _prev
        # Posts a summary then archives THIS channel; stops tracking it.
        assert slack.archived == ("C42", "1700.0")
        assert "C42" not in orch._code_channels

    @pytest.mark.asyncio
    async def test_archive_button_is_inert_with_the_flag_off(self, monkeypatch: Any) -> None:
        """With ``slack.code_channels`` off the button makes no Slack call at all,
        and the refusal is audited."""
        from kiro_crew.slack import interactions

        audited: list[dict] = []

        class _Sel:
            def log_tool_invocation(self, **kw: Any) -> None:
                audited.append(kw)

        monkeypatch.setattr(interactions, "sel", lambda: _Sel())
        slack = _FakeSlack()
        calls: list[str] = []

        async def _probe(channel_id: str) -> bool:
            calls.append(channel_id)
            return True

        slack.is_code_channel = _probe  # type: ignore[method-assign]
        orch = SimpleNamespace(
            slack=slack, _code_channels={"C42"}, _cfg=_code_channels_cfg(enabled=False)
        )
        _prev = interactions._orch
        interactions._orch = orch  # type: ignore[assignment]
        try:
            await interactions._handle_code_channel_archive({"value": "C42"}, "C42", "U1")
        finally:
            interactions._orch = _prev
        assert calls == [] and slack.posts == [] and slack.archived is None
        assert "C42" in orch._code_channels
        assert audited and audited[-1]["outcome"] == "denied"
        assert audited[-1]["metadata"]["reason"] == "code_channels_off"


class TestArchiveRefusedOutsideACodeChannel:
    """Archive posts its summary only in a code channel; elsewhere nothing is posted."""

    @pytest.mark.asyncio
    async def test_command_in_an_ordinary_channel_posts_nothing(self, monkeypatch) -> None:
        from kiro_crew.slack import events

        monkeypatch.setattr(events, "_CODE_CHANNEL_CACHE", {})
        orch = _FakeOrch(has=False, code_channels=True)
        orch.slack.is_code_channel_result = False
        orch._last_channel_id = "C_MAIN"  # type: ignore[attr-defined]
        replies: list[str] = []
        await events._handle_archive_codechannel(
            orch, "U1", "done", lambda t, **k: _append(replies, t)
        )
        assert orch.slack.posts == [] and orch.slack.archived is None
        assert "inside the code channel" in replies[0]

    @pytest.mark.asyncio
    async def test_button_naming_an_ordinary_channel_posts_nothing(self) -> None:
        from kiro_crew.slack import interactions

        slack = _FakeSlack()
        slack.is_code_channel_result = False
        orch = SimpleNamespace(slack=slack, _code_channels=set(), _cfg=_code_channels_cfg())
        _prev = interactions._orch
        interactions._orch = orch  # type: ignore[assignment]
        try:
            await interactions._handle_code_channel_archive({"value": "C_MAIN"}, "C_MAIN", "U1")
        finally:
            interactions._orch = _prev
        assert slack.posts == [] and slack.archived is None


class _SlackErr(Exception):
    """Minimal stand-in carrying a subscriptable ``response`` with an error code,
    so ``_slack_error_code`` extracts it like a real SlackApiError (whose
    ``.response`` supports ``response["error"]``)."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"error": code}


class TestSlackErrorCode:
    def test_reads_the_code_off_a_slack_api_error(self) -> None:
        assert slack_client._slack_error_code(_SlackErr("missing_scope")) == "missing_scope"

    def test_empty_when_the_error_carries_no_response(self) -> None:
        assert slack_client._slack_error_code(RuntimeError("socket closed")) == ""


class TestLastCodeChannelErrorIsPerTask:
    """Two sessions whose code-channel calls fail at the same time each read their
    own Slack error code, not whichever failure landed last."""

    @pytest.mark.asyncio
    async def test_concurrent_failures_do_not_cross(self) -> None:
        import asyncio

        client = _client(_RecordingWeb())
        order: list[str] = []

        async def fail_then_read(code: str, delay: float) -> str:
            client._last_code_channel_error = code
            order.append(code)
            await asyncio.sleep(delay)  # the other task sets its code meanwhile
            return client._last_code_channel_error

        first, second = await asyncio.gather(
            fail_then_read("missing_scope", 0.02), fail_then_read("feature_disabled", 0.0)
        )
        assert order == ["missing_scope", "feature_disabled"]
        assert (first, second) == ("missing_scope", "feature_disabled")


class TestCodeChannelArchiveSummary:
    """The summary a code channel is archived with: the cached session summary,
    else a change summary from the diff baseline, else the fixed default. Shared by
    the button and a bare ``/kirocrew archive``, and never a model call."""

    @pytest.fixture(autouse=True)
    def _no_model_call(self, monkeypatch: Any) -> None:
        from kiro_crew import llm_helpers
        from kiro_crew.dashboard import chat_summary

        async def _refuse(*a: Any, **k: Any) -> Any:
            raise AssertionError("archiving must not make a model call")

        monkeypatch.setattr(llm_helpers, "run_bg_oneliner", _refuse)
        monkeypatch.setattr(chat_summary, "run_bg_oneliner", _refuse)
        monkeypatch.setattr(chat_summary, "generate_session_summary", _refuse)

    @staticmethod
    def _orch(tmp_path: Any, *, anchor: str | None = "1700.1") -> SimpleNamespace:
        from kiro_crew.history import ConversationLog

        return SimpleNamespace(
            slack=_FakeSlack(),
            _cfg=_code_channels_cfg(),
            conv_log=ConversationLog(base_dir=tmp_path / "sessions"),
            _code_channels={"C42"},
            _owned_code_channels={"C42"},
            _code_channel_session_ts={"C42": anchor} if anchor else {},
            _code_channel_repo_by_id={},
            _code_channel_diff_base={},
            _code_channel_origin={"C42": ("C_MAIN", "1600.0")},
            _last_channel_id="C42",
        )

    @staticmethod
    def _write_intents(orch: SimpleNamespace, titles: list[str]) -> None:
        key = "slack:1700.1"
        log = orch.conv_log
        log.append(key, "user", "fix the checkout bug")
        payload = {
            "intents": [
                {"title": t, "ranges": [[1, n + 1]], "status": "completed", "verified": True}
                for n, t in enumerate(titles)
            ],
            "constraints": [],
        }
        assert log.set_cached_intent_summary(key, payload, log.session_mtime(key))

    @pytest.mark.asyncio
    async def test_uses_the_cached_session_summary(self, tmp_path: Any) -> None:
        from kiro_crew.slack import events

        orch = self._orch(tmp_path)
        self._write_intents(orch, ["Fix checkout 500s", "Add a regression test"])
        summary = await events.code_channel_archive_summary(orch, "C42")
        # Most recently touched first, with the state the panel shows.
        assert summary == "Add a regression test (done); Fix checkout 500s (done)"

    @pytest.mark.asyncio
    async def test_session_summary_markup_is_escaped(self, tmp_path: Any) -> None:
        from kiro_crew.slack import events

        orch = self._orch(tmp_path)
        self._write_intents(orch, ["Ping <!channel> & <@U999>"])
        summary = await events.code_channel_archive_summary(orch, "C42")
        assert "<!channel>" not in summary and "<@U999>" not in summary
        assert "&lt;!channel&gt; &amp; &lt;@U999&gt;" in summary

    @pytest.mark.asyncio
    async def test_a_withheld_transcript_is_not_read(self, tmp_path: Any, monkeypatch: Any) -> None:
        from kiro_crew.slack import events

        orch = self._orch(tmp_path)
        self._write_intents(orch, ["Private work"])
        monkeypatch.setattr(events, "transcript_withholds_derivation", lambda log, key: True)
        summary = await events.code_channel_archive_summary(orch, "C42")
        assert summary == events.ARCHIVE_SUMMARY_DEFAULT

    @pytest.mark.asyncio
    async def test_falls_back_to_a_change_summary_from_the_baseline(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from kiro_crew.slack import events

        calls = _fake_git(
            monkeypatch,
            events,
            diff="10\t2\tsrc/app.py\n-\t-\tlogo.png\n3\t0\t<!here>.md\n",
        )
        orch = self._orch(tmp_path)
        orch._code_channel_repo_by_id = {"C42": _repo_dir(tmp_path)}
        orch._code_channel_diff_base = {"C42": "abc123"}
        summary = await events.code_channel_archive_summary(orch, "C42")
        assert summary == "Changed 3 files (+13 -2): src/app.py, logo.png, &lt;!here&gt;.md"
        # Diffed against the baseline the diff tab uses.
        assert calls[-1][3:] == ["diff", "--numstat", "abc123"]

    @pytest.mark.asyncio
    async def test_change_summary_names_a_capped_number_of_files(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from kiro_crew.slack import events

        n = events._ARCHIVE_SUMMARY_MAX_FILES + 2
        _fake_git(monkeypatch, events, diff="".join(f"1\t1\tf{i}.py\n" for i in range(n)))
        orch = self._orch(tmp_path, anchor=None)
        orch._code_channel_repo_by_id = {"C42": _repo_dir(tmp_path)}
        summary = await events.code_channel_archive_summary(orch, "C42")
        assert summary.startswith(f"Changed {n} files (+{n} -{n}): f0.py")
        assert summary.endswith(" and 2 more")
        assert f"f{events._ARCHIVE_SUMMARY_MAX_FILES}.py" not in summary

    @pytest.mark.asyncio
    async def test_summary_is_capped(self, tmp_path: Any) -> None:
        from kiro_crew.slack import events

        orch = self._orch(tmp_path)
        self._write_intents(orch, ["x" * (events._ARCHIVE_SUMMARY_MAX_CHARS * 2)])
        summary = await events.code_channel_archive_summary(orch, "C42")
        assert len(summary) == events._ARCHIVE_SUMMARY_MAX_CHARS
        assert summary.endswith("…")

    @pytest.mark.asyncio
    async def test_the_cap_never_splits_an_escaped_entity(self, tmp_path: Any) -> None:
        """The clip lands on the raw text, so every ``&`` Slack sees starts a whole
        entity rather than a fragment like ``&am``."""
        from kiro_crew.slack import events

        orch = self._orch(tmp_path)
        self._write_intents(orch, ["a&" * events._ARCHIVE_SUMMARY_MAX_CHARS])
        summary = await events.code_channel_archive_summary(orch, "C42")
        assert summary.endswith("…")
        assert "&" not in summary.replace("&amp;", "")

    @pytest.mark.asyncio
    async def test_default_when_there_is_nothing_to_summarise(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from kiro_crew.slack import events

        _fake_git(monkeypatch, events, diff="")
        orch = self._orch(tmp_path)
        orch._code_channel_repo_by_id = {"C42": _repo_dir(tmp_path)}
        summary = await events.code_channel_archive_summary(orch, "C42")
        assert summary == events.ARCHIVE_SUMMARY_DEFAULT == "Session complete."

    @pytest.mark.asyncio
    async def test_button_posts_the_summary_in_the_channel_and_the_origin(
        self, tmp_path: Any
    ) -> None:
        from kiro_crew.slack import interactions

        orch = self._orch(tmp_path)
        self._write_intents(orch, ["Fix checkout 500s"])
        _prev = interactions._orch
        interactions._orch = orch  # type: ignore[assignment]
        try:
            await interactions._handle_code_channel_archive({"value": "C42"}, "C42", "U1")
        finally:
            interactions._orch = _prev
        assert orch.slack.posts == [
            ("C42", ":white_check_mark: *Summary:* Fix checkout 500s (done)"),
            ("C_MAIN", ":white_check_mark: Resolved in <#C42>: Fix checkout 500s (done)"),
        ]
        assert orch.slack.archived == ("C42", "1700.0")

    @pytest.mark.asyncio
    async def test_bare_slash_archive_uses_the_same_summary(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from kiro_crew.slack import events

        monkeypatch.setattr(events, "_CODE_CHANNEL_CACHE", {})
        orch = self._orch(tmp_path)
        self._write_intents(orch, ["Fix checkout 500s"])
        replies: list[str] = []
        await events._handle_archive_codechannel(
            orch, "U1", "  ", lambda t, **k: _append(replies, t)
        )
        assert orch.slack.posts == [
            ("C42", ":white_check_mark: *Summary:* Fix checkout 500s (done)"),
            ("C_MAIN", ":white_check_mark: Resolved in <#C42>: Fix checkout 500s (done)"),
        ]

    @pytest.mark.asyncio
    async def test_slash_archive_keeps_the_users_text(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from kiro_crew.slack import events

        monkeypatch.setattr(events, "_CODE_CHANNEL_CACHE", {})
        orch = self._orch(tmp_path)
        self._write_intents(orch, ["Fix checkout 500s"])
        replies: list[str] = []
        await events._handle_archive_codechannel(
            orch, "U1", "Shipped the fix", lambda t, **k: _append(replies, t)
        )
        assert orch.slack.posts[0] == ("C42", ":white_check_mark: *Summary:* Shipped the fix")
        assert orch.slack.posts[1][1].endswith(": Shipped the fix")

"""Unit-level checks for the code-channel handoff wiring, without any
live Slack or an end-to-end run:

- ``_format_origin_handoff`` fetches the originating thread and renders it (the
  "pull the whole thread to get issue + root cause" path), bounded and ping-free.
- ``create_code_channel_core`` captures that handoff + the origin mapping when the
  channel is opened from a triggering message.
- ``start_stream`` OMITS ``thread_ts`` for a top-level code-channel stream and
  includes it only for a real thread ts (the streaming fix).
- ``_post_summary_to_origin`` posts the outcome back to the origin and consumes the
  mapping (the loop-close).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kiro_crew.slack import events
from kiro_crew.slack.client import RealSlackClient


class _Slack:
    def __init__(self, *, thread: list[dict] | None = None, created: str | None = "C999") -> None:
        self._thread = thread or []
        self._created = created
        self.status_calls: list = []
        self.invite_calls: list = []
        self.block_posts: list = []
        self.posts: list[tuple[str, str, Any]] = []

    async def fetch_thread_replies(self, channel: str, ts: str, **kw: Any) -> list[dict]:
        return self._thread

    async def create_code_channel(self, name: str, **kw: Any) -> str | None:
        return self._created

    async def set_session_status(self, channel: str, thread_ts: str | None, status: str) -> None:
        self.status_calls.append((channel, thread_ts, status))

    async def invite_users(self, channel_id: str, user_ids: list[str]) -> dict[str, Any]:
        self.invite_calls.append((channel_id, list(user_ids)))
        return {"ok": True, "invited": user_ids, "error": None}

    async def post_blocks(self, channel: str, blocks: list, text: str, *a: Any, **k: Any) -> str:
        self.block_posts.append((channel, blocks, text))
        return "1700.0"

    async def post_message(self, channel: str, text: str, *a: Any, **k: Any) -> str:
        self.posts.append((channel, text, k.get("thread_ts")))
        return "1700.9"


def _code_channels_cfg(enabled: bool = True) -> SimpleNamespace:
    """A config whose only Slack field is the ``slack.code_channels`` flag."""
    return SimpleNamespace(slack=SimpleNamespace(code_channels=enabled, code_channel_repo=""))


def _orch(slack: _Slack, *, code_channels: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        slack=slack,
        _code_channels=set(),
        _code_channel_origin={},
        _code_channel_handoff={},
        _code_channel_repo_by_id={},
        _owner_id="U_OWNER",
        _cfg=SimpleNamespace(
            slack=SimpleNamespace(
                code_channels=code_channels,
                code_channel_invitees=[],
                code_channel_repo="",
                code_channel_context_items=[],
            )
        ),
    )


class TestFormatOriginHandoff:
    @pytest.mark.asyncio
    async def test_renders_thread_as_labelled_lines(self) -> None:
        slack = _Slack(
            thread=[
                {"user": "U_AGENT", "text": "Root cause: order 3 missing 'price'."},
                {"user": "U_HUMAN", "text": "Fix and deploy please."},
                {"text": ""},  # empty is skipped
            ]
        )
        out = await events._format_origin_handoff(_orch(slack), "C_MAIN", "1700.1")
        assert "U_AGENT: Root cause: order 3 missing 'price'." in out
        assert "U_HUMAN: Fix and deploy please." in out
        # no <@id> mention form -> injected context must not ping anyone
        assert "<@" not in out

    @pytest.mark.asyncio
    async def test_bounded(self) -> None:
        slack = _Slack(thread=[{"user": "U1", "text": "x" * 9000}])
        out = await events._format_origin_handoff(_orch(slack), "C", "1")
        assert len(out) <= events._HANDOFF_MAX_CHARS + len("\n… (handoff truncated)")
        assert out.endswith("(handoff truncated)")

    @pytest.mark.asyncio
    async def test_empty_thread_returns_empty(self) -> None:
        assert await events._format_origin_handoff(_orch(_Slack(thread=[])), "C", "1") == ""


class TestSeedingOnCreate:
    @pytest.mark.asyncio
    async def test_origin_and_handoff_captured(self, monkeypatch: Any) -> None:
        # Neutralize the always-on config persist (not under test here).
        async def _noop(*a: Any, **k: Any) -> None:
            return None

        monkeypatch.setattr(events, "run_config_write", _noop)
        slack = _Slack(thread=[{"user": "U_AGENT", "text": "Root cause: missing price"}])
        orch = _orch(slack)
        res = await events.create_code_channel_core(
            orch,
            "Fix the orders service",
            "U_HUMAN",
            origin_channel_id="C_MAIN",
            origin_message_ts="1700.1",
        )
        assert res["ok"] and res["channel_id"] == "C999"
        # origin recorded for share-back, handoff captured for first-turn seeding
        assert orch._code_channel_origin["C999"] == ("C_MAIN", "1700.1")
        assert "Root cause: missing price" in orch._code_channel_handoff["C999"]

    @pytest.mark.asyncio
    async def test_per_channel_repo_bound_and_overrides_global(self, monkeypatch: Any) -> None:
        async def _noop(*a: Any, **k: Any) -> None:
            return None

        monkeypatch.setattr(events, "run_config_write", _noop)
        # global default is set, but the per-task repo must win and bind per channel
        orch = _orch(_Slack())
        orch._cfg.slack.code_channel_repo = "/global/default-repo"
        res = await events.create_code_channel_core(
            orch, "task", "U_HUMAN", repo="/work/payments-service"
        )
        assert res["ok"]
        assert orch._code_channel_repo_by_id["C999"] == "/work/payments-service"

    @pytest.mark.asyncio
    async def test_no_repo_falls_back_to_global(self, monkeypatch: Any) -> None:
        async def _noop(*a: Any, **k: Any) -> None:
            return None

        monkeypatch.setattr(events, "run_config_write", _noop)
        orch = _orch(_Slack())
        orch._cfg.slack.code_channel_repo = "/global/default-repo"
        await events.create_code_channel_core(orch, "task", "U_HUMAN")  # no repo passed
        assert orch._code_channel_repo_by_id["C999"] == "/global/default-repo"

    @pytest.mark.asyncio
    async def test_no_origin_no_handoff(self, monkeypatch: Any) -> None:
        async def _noop(*a: Any, **k: Any) -> None:
            return None

        monkeypatch.setattr(events, "run_config_write", _noop)
        orch = _orch(_Slack())
        res = await events.create_code_channel_core(orch, "adhoc", "U_HUMAN")
        assert res["ok"]
        assert orch._code_channel_handoff == {}
        assert orch._code_channel_origin == {}


class TestStartStreamThreadTs:
    def _client(self) -> tuple[RealSlackClient, list[dict]]:
        bodies: list[dict] = []

        class _Web:
            async def api_call(self, method: str, *, json: dict | None = None, **k: Any) -> dict:
                bodies.append(json or {})
                return {"ts": "1700.5"}

        c = RealSlackClient.__new__(RealSlackClient)
        c._web = _Web()  # type: ignore[attr-defined]
        c._channel_team = {}  # type: ignore[attr-defined]

        async def _noop_team(channel: str) -> None:
            return None

        c.ensure_channel_team = _noop_team  # type: ignore[assignment]
        return c, bodies

    @pytest.mark.asyncio
    async def test_omits_thread_ts_for_top_level(self) -> None:
        c, bodies = self._client()
        await c.start_stream("C_CODE", "")  # code channel: renderer passes ""
        assert "thread_ts" not in bodies[0]

    @pytest.mark.asyncio
    async def test_includes_thread_ts_when_real(self) -> None:
        c, bodies = self._client()
        await c.start_stream("C_CHAN", "1700.1")
        assert bodies[0]["thread_ts"] == "1700.1"


class TestActionReplyThreadTs:
    """Button/OPTIONS responses must post TOP-LEVEL in a code channel (one session),
    but keep threading under the message elsewhere."""

    def _set_code_channels(self, monkeypatch: Any, channels: set[str]) -> None:
        from kiro_crew.slack import interactions

        monkeypatch.setattr(
            interactions,
            "_orch",
            SimpleNamespace(_code_channels=channels, _cfg=_code_channels_cfg()),
            raising=False,
        )

    def test_code_channel_top_level(self, monkeypatch: Any) -> None:
        from kiro_crew.slack import interactions

        self._set_code_channels(monkeypatch, {"C_CODE"})
        # no genuine user thread -> top-level (empty), NOT the msg_ts fallback
        assert interactions._action_reply_thread_ts("C_CODE", "111.1", "") == ""

    def test_code_channel_honors_genuine_user_thread(self, monkeypatch: Any) -> None:
        from kiro_crew.slack import interactions

        self._set_code_channels(monkeypatch, {"C_CODE"})
        assert interactions._action_reply_thread_ts("C_CODE", "111.1", "999.9") == "999.9"

    def test_non_code_channel_threads_under_message(self, monkeypatch: Any) -> None:
        from kiro_crew.slack import interactions

        self._set_code_channels(monkeypatch, set())
        assert interactions._action_reply_thread_ts("C_PLAIN", "111.1", "") == "111.1"


class TestSummaryToOrigin:
    @pytest.mark.asyncio
    async def test_posts_back_and_consumes_mapping(self) -> None:
        slack = _Slack()
        orch = _orch(slack)
        orch._code_channel_origin["C999"] = ("C_MAIN", "1700.1")
        await events._post_summary_to_origin(orch, "C999", "fixed the KeyError")
        assert slack.posts and slack.posts[0][0] == "C_MAIN"
        assert "fixed the KeyError" in slack.posts[0][1]
        assert slack.posts[0][2] == "1700.1"  # threaded on the origin message
        # mapping consumed so a re-archive can't double-post
        assert "C999" not in orch._code_channel_origin

    @pytest.mark.asyncio
    async def test_noop_when_origin_unknown(self) -> None:
        slack = _Slack()
        await events._post_summary_to_origin(_orch(slack), "C999", "summary")
        assert slack.posts == []


class TestMirrorPostThreadInCodeChannel:
    """Dashboard-slot mirror posts (option clicks, replies) in a code channel.

    An option click in a code channel is routed to the linked dashboard slot, whose
    mirror posts under the link's thread. In a code channel those posts go
    top-level; everywhere else they keep threading under the link.
    """

    def _state(self, code_channels):
        from types import SimpleNamespace

        return SimpleNamespace(_is_code_channel=set(code_channels).__contains__)

    def test_code_channel_posts_top_level(self):
        from kiro_crew.dashboard.chat_runner import _mirror_post_thread

        assert _mirror_post_thread(self._state({"C_CC"}), "C_CC", "1700.1") is None

    def test_normal_channel_keeps_thread(self):
        from kiro_crew.dashboard.chat_runner import _mirror_post_thread

        assert _mirror_post_thread(self._state({"C_CC"}), "C_MAIN", "1700.1") == "1700.1"

    def test_unwired_state_keeps_thread(self):
        from types import SimpleNamespace

        from kiro_crew.dashboard.chat_runner import _mirror_post_thread

        assert _mirror_post_thread(SimpleNamespace(), "C_CC", "1700.1") == "1700.1"

    def test_stand_in_state_keeps_thread(self):
        """A MagicMock state answers a truthy mock, not True: no behaviour change."""
        from unittest.mock import MagicMock

        from kiro_crew.dashboard.chat_runner import _mirror_post_thread

        assert _mirror_post_thread(MagicMock(), "C_CC", "1700.1") == "1700.1"


class TestCodeChannelSurvivesRestart:
    """Creation persists the per-channel record; a fresh gateway restores it."""

    @pytest.mark.asyncio
    async def test_created_channel_restores_repo_and_origin(
        self, monkeypatch: Any, tmp_path: Any
    ) -> None:
        from kiro_crew.slack import code_channel_store

        async def _noop(*a: Any, **k: Any) -> None:
            return None

        monkeypatch.setattr(events, "run_config_write", _noop)
        monkeypatch.setattr(code_channel_store, "config_dir", lambda: tmp_path)
        orch = _orch(_Slack())
        res = await events.create_code_channel_core(
            orch,
            "task",
            "U_HUMAN",
            origin_channel_id="C_MAIN",
            origin_message_ts="1700.1",
            repo="/work/payments-service",
        )
        assert res["ok"]

        fresh = SimpleNamespace(
            _code_channels=set(),
            _code_channel_repo_by_id={},
            _code_channel_diff_base={},
            _code_channel_origin={},
        )
        code_channel_store.restore_into(fresh)
        assert "C999" in fresh._code_channels
        assert fresh._code_channel_repo_by_id["C999"] == "/work/payments-service"
        assert fresh._code_channel_origin["C999"] == ("C_MAIN", "1700.1")


class TestCodeChannelOneSession:
    """Every top-level message in a code channel continues the channel's session."""

    def _gw(self, code_channels):
        from kiro_crew.slack.gateway import GatewayOrchestrator

        gw = SimpleNamespace(
            _code_channels=set(code_channels),
            _code_channel_session_ts={},
            _cfg=_code_channels_cfg(),
        )
        gw.code_channel_session_ts = lambda ch, ts: GatewayOrchestrator.code_channel_session_ts(
            gw, ch, ts
        )
        return gw

    def test_first_top_level_message_sets_the_anchor(self) -> None:
        gw = self._gw({"C_CC"})
        assert gw.code_channel_session_ts("C_CC", "1700.1") == ("1700.1", True)
        assert gw.code_channel_session_ts("C_CC", "1700.5") == ("1700.1", False)

    def test_top_level_messages_share_one_session_ts(self) -> None:
        from kiro_crew.slack.transport_dispatch import _session_ts

        gw = self._gw({"C_CC"})
        first = _session_ts(gw, "C_CC", None, "1700.1")
        later = _session_ts(gw, "C_CC", None, "1700.9")
        assert first == later == "1700.1"

    def test_thread_reply_in_code_channel_continues_the_channel_session(self) -> None:
        """A reply threaded under any message keeps the channel's one session (its
        memory); where the reply is POSTED is decided separately and still honours
        the user's thread (_code_channel_post_ts)."""
        from kiro_crew.slack.transport_dispatch import _session_ts

        gw = self._gw({"C_CC"})
        _session_ts(gw, "C_CC", None, "1700.1")
        assert _session_ts(gw, "C_CC", "1700.3", "1700.4") == "1700.1"

    def test_normal_channel_is_unchanged(self) -> None:
        from kiro_crew.slack.transport_dispatch import _session_ts

        gw = self._gw({"C_CC"})
        assert _session_ts(gw, "C_MAIN", None, "1700.7") == "1700.7"
        assert _session_ts(gw, "C_MAIN", "1700.2", "1700.7") == "1700.2"

    def test_no_gateway_is_unchanged(self) -> None:
        from kiro_crew.slack.transport_dispatch import _session_ts

        assert _session_ts(None, "C_CC", None, "1700.7") == "1700.7"
        assert _session_ts(SimpleNamespace(), "C_CC", None, "1700.7") == "1700.7"


class TestCodeChannelWorkingContext:
    """The standing instructions a code-channel turn starts with."""

    def test_names_the_repo(self) -> None:
        assert "/work/payments-service" in events._code_channel_working_context(
            "/work/payments-service"
        )

    def test_builds_on_the_handed_off_conversation(self) -> None:
        text = events._code_channel_working_context("/r")
        assert "is your brief" in text and "rather than redoing that work" in text

    def test_deploys_only_when_asked(self) -> None:
        text = events._code_channel_working_context("/r")
        assert "Deploy only when the task or someone in the channel asks" in text

    def test_does_not_direct_the_agent_to_an_external_investigator(self) -> None:
        text = events._code_channel_working_context("/r").lower()
        assert "devops" not in text and "investigate /" not in text


class TestHandoffReadyBeforeFirstTurn:
    """Slack's opening message can start the first turn mid-setup; by then the
    channel must already be tracked with its origin, handoff and repo."""

    @pytest.mark.asyncio
    async def test_state_is_registered_before_the_first_setup_await(self, monkeypatch: Any) -> None:
        async def _noop(*a: Any, **k: Any) -> None:
            return None

        monkeypatch.setattr(events, "run_config_write", _noop)
        seen: dict[str, Any] = {}

        class _RacingSlack(_Slack):
            async def set_session_status(self, channel: str, thread_ts: Any, status: str) -> None:
                # The first await after creation: an inbound message handled here
                # sees exactly this state.
                seen["tracked"] = channel in orch._code_channels
                seen["origin"] = orch._code_channel_origin.get(channel)
                seen["handoff"] = orch._code_channel_handoff.get(channel, "")
                seen["repo"] = orch._code_channel_repo_by_id.get(channel)
                await super().set_session_status(channel, thread_ts, status)

        orch = _orch(_RacingSlack(thread=[{"user": "U_AGENT", "text": "Root cause: qty is zero"}]))
        res = await events.create_code_channel_core(
            orch,
            "task",
            "U_HUMAN",
            origin_channel_id="C_MAIN",
            origin_message_ts="1700.1",
            repo="/work/repo",
        )
        assert res["ok"]
        assert seen["tracked"] is True
        assert seen["origin"] == ("C_MAIN", "1700.1")
        assert "Root cause: qty is zero" in seen["handoff"]
        assert seen["repo"] == "/work/repo"


class TestCodeChannelTurnPreamble:
    """One builder for the working context, whichever way a turn arrives."""

    def _orch(self, channels, repo_by_id, global_repo="", origin=None):
        return SimpleNamespace(
            _code_channels=set(channels),
            _code_channel_repo_by_id=repo_by_id,
            _code_channel_origin=origin or {},
            _cfg=SimpleNamespace(
                slack=SimpleNamespace(code_channel_repo=global_repo, code_channels=True)
            ),
        )

    def test_code_channel_with_repo_gets_context(self, monkeypatch: Any) -> None:
        orch = self._orch({"C_CC"}, {"C_CC": "/work/repo"})
        text = events.code_channel_turn_preamble(orch, "C_CC")
        assert text.startswith("[Code channel working context]") and "/work/repo" in text

    def test_other_channel_gets_nothing(self) -> None:
        orch = self._orch({"C_CC"}, {"C_CC": "/work/repo"})
        assert events.code_channel_turn_preamble(orch, "C_MAIN") == ""

    def test_code_channel_without_any_repo_gets_nothing(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            events.KiroCrewConfig,
            "load",
            staticmethod(lambda: SimpleNamespace(slack=SimpleNamespace(code_channel_repo=""))),
        )
        orch = self._orch({"C_CC"}, {})
        assert events.code_channel_turn_preamble(orch, "C_CC") == ""


class TestRestoredRecordIsInertWithTheFlagOff:
    """A record restored from when ``slack.code_channels`` was on keeps the channel
    in ``_code_channels``, but with the flag off no code-channel behaviour runs."""

    def _restored(self, *, enabled: bool) -> SimpleNamespace:
        return SimpleNamespace(
            _code_channels={"C_CC"},
            _code_channel_repo_by_id={"C_CC": "/work/repo"},
            _code_channel_session_ts={"C_CC": "1700.1"},
            _cfg=_code_channels_cfg(enabled),
        )

    def test_the_predicate_follows_the_flag(self) -> None:
        from kiro_crew.slack.handler import is_tracked_code_channel

        assert is_tracked_code_channel(self._restored(enabled=True), "C_CC") is True
        assert is_tracked_code_channel(self._restored(enabled=False), "C_CC") is False
        assert is_tracked_code_channel(self._restored(enabled=True), "C_OTHER") is False

    def test_no_turn_context(self) -> None:
        assert events.code_channel_turn_preamble(self._restored(enabled=False), "C_CC") == ""

    def test_no_session_anchor(self) -> None:
        from kiro_crew.slack.gateway import GatewayOrchestrator

        gw = self._restored(enabled=False)
        assert GatewayOrchestrator.code_channel_session_ts(gw, "C_CC", "1700.9") == (
            "1700.9",
            False,
        )

    def test_no_stop_anchor(self) -> None:
        assert events._code_channel_stop_anchor(self._restored(enabled=False), "C_CC", None) is None

    def test_replies_keep_threading(self, monkeypatch: Any) -> None:
        from kiro_crew.slack import interactions

        monkeypatch.setattr(interactions, "_orch", self._restored(enabled=False), raising=False)
        assert interactions._action_reply_thread_ts("C_CC", "111.1", "") == "111.1"


class TestAddressingRule:
    """One rule wherever Kiro Crew answers without an @-mention."""

    def test_message_for_someone_else_is_theirs(self) -> None:
        assert events._addressed_to_someone_else("<@U0OTHER> please verify", "U0SELF")

    def test_message_mentioning_self_is_ours(self) -> None:
        assert not events._addressed_to_someone_else("<@U0SELF> and <@U0OTHER>", "U0SELF")

    def test_unaddressed_message_is_ours(self) -> None:
        assert not events._addressed_to_someone_else("can you add a test?", "U0SELF")

    def test_unknown_self_answers(self) -> None:
        assert not events._addressed_to_someone_else("<@U0OTHER> hi", "")


class TestCodeChannelOwnership:
    """Always-on only where this Kiro Crew is in charge of the code channel."""

    def _orch(self, agents: Any) -> SimpleNamespace:
        return SimpleNamespace(
            slack=SimpleNamespace(code_channel_agent_ids=AsyncMock(return_value=agents))
        )

    @pytest.mark.asyncio
    async def test_channel_assigned_to_this_bot_alone_is_owned(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(events, "validated_self_user_id", lambda: "U_SELF")
        assert await events._owns_detected_code_channel(self._orch(["U_SELF"]), "C1") is True

    @pytest.mark.asyncio
    async def test_channel_shared_with_other_agents_is_not_owned(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(events, "validated_self_user_id", lambda: "U_SELF")
        orch = self._orch(["U_OTHER_AGENT", "U_SELF"])
        assert await events._owns_detected_code_channel(orch, "C1") is False

    @pytest.mark.asyncio
    async def test_unknown_assignment_keeps_it_owned(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(events, "validated_self_user_id", lambda: "U_SELF")
        assert await events._owns_detected_code_channel(self._orch(None), "C1") is True

    @pytest.mark.asyncio
    async def test_unknown_self_keeps_it_owned(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(events, "validated_self_user_id", lambda: "")
        assert await events._owns_detected_code_channel(self._orch(["U_X", "U_Y"]), "C1") is True

    @pytest.mark.asyncio
    async def test_created_channel_is_owned_and_persisted(
        self, monkeypatch: Any, tmp_path: Any
    ) -> None:
        from kiro_crew.slack import code_channel_store

        async def _noop(*a: Any, **k: Any) -> None:
            return None

        monkeypatch.setattr(events, "run_config_write", _noop)
        monkeypatch.setattr(code_channel_store, "config_dir", lambda: tmp_path)
        orch = _orch(_Slack())
        orch._owned_code_channels = set()
        res = await events.create_code_channel_core(orch, "task", "U_HUMAN")
        assert res["ok"] and "C999" in orch._owned_code_channels

        fresh = SimpleNamespace(
            _code_channels=set(),
            _owned_code_channels=set(),
            _code_channel_repo_by_id={},
            _code_channel_diff_base={},
            _code_channel_origin={},
        )
        code_channel_store.restore_into(fresh)
        assert "C999" in fresh._owned_code_channels

    def test_archive_clears_ownership(self) -> None:
        from kiro_crew.slack import code_channel_store

        orch = SimpleNamespace(
            _code_channels={"C1"},
            _owned_code_channels={"C1"},
            _code_channel_repo_by_id={},
            _code_channel_diff_base={},
            _code_channel_session_ts={},
        )
        code_channel_store.clear_channel_state(orch, "C1")
        assert "C1" not in orch._owned_code_channels


class TestEchoShowsOnlyWhatTheUserTyped:
    """The dashboard-to-Slack echo of a code-channel turn never shows the agent-only
    working context the turn is prefixed with."""

    def test_mirror_echo_drops_the_context_block(self) -> None:
        from kiro_crew.dashboard.chat_runner import _prepare_mirror_msg

        turn = events._code_channel_working_context("/work/repo") + "Deploy the fix now"
        assert _prepare_mirror_msg(turn) == "Deploy the fix now"

    def test_ordinary_message_is_unchanged(self) -> None:
        from kiro_crew.dashboard.chat_runner import _prepare_mirror_msg

        assert _prepare_mirror_msg("hello there") == "hello there"

    def test_text_that_merely_mentions_the_header_is_unchanged(self) -> None:
        from kiro_crew.slack.code_channel_store import strip_code_channel_context

        text = "why does it say [Code channel working context]?"
        assert strip_code_channel_context(text) == text

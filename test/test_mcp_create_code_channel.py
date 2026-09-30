"""The create_code_channel MCP tool: its advertised shape, its handler's identity
and governance gates, the gateway endpoint that backs it, and the shared Slack-side
core it delegates to.

The tool lets Kiro Crew open a Slack code channel mid-turn (the same path as the
``/kirocrew codechannel`` slash command). Because it creates a channel and invites
people, it is held to the same strict-identity and channel-agent containment bar as
send_message, and the endpoint is internal-secret only.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew.dashboard.handlers import api_create_code_channel
from kiro_crew.mcp_tools.messaging import HANDLERS, create_code_channel, schemas
from kiro_crew.validation import (
    CREATE_CODE_CHANNEL_SCHEMA,
    MCP_CORE_SCHEMAS,
    ValidationError,
    validate_tool_args,
)

# ── Advertisement / registry agreement ──


def _advertised() -> dict:
    for tool in schemas():
        if tool["name"] == "create_code_channel":
            return tool
    raise AssertionError("create_code_channel is not advertised")


def test_tool_is_advertised_and_handled() -> None:
    assert "create_code_channel" in HANDLERS
    tool = _advertised()
    assert tool["inputSchema"]["required"] == ["name"]


def test_advertised_properties_are_all_validated() -> None:
    known = {spec.name for spec in CREATE_CODE_CHANNEL_SCHEMA.fields}
    advertised = set(_advertised()["inputSchema"]["properties"])
    missing = sorted(advertised - known)
    assert not missing, f"advertised but unvalidatable, so those calls fail: {missing}"


def test_schema_is_registered() -> None:
    assert MCP_CORE_SCHEMAS["create_code_channel"] is CREATE_CODE_CHANNEL_SCHEMA


def test_validation_rejects_missing_or_empty_name() -> None:
    with pytest.raises(ValidationError):
        validate_tool_args({}, CREATE_CODE_CHANNEL_SCHEMA)
    with pytest.raises(ValidationError):
        validate_tool_args({"name": "x" * 100_000}, CREATE_CODE_CHANNEL_SCHEMA)
    cleaned = validate_tool_args({"name": "Fix the orders service"}, CREATE_CODE_CHANNEL_SCHEMA)
    assert cleaned["name"] == "Fix the orders service"


# ── Tool handler: identity + governance gates ──


@contextlib.contextmanager
def _tool_mcp_core(
    *, strict="slack:1712793600.1", deny=None, gov_msg=None, gov_chan=None, post=None
):
    """Patch the mcp_core seams create_code_channel resolves at call time. Handlers
    reach them as attributes of mcp_core, so rebinding on the module intercepts."""
    from kiro_crew import mcp_core

    with contextlib.ExitStack() as stack:
        yield {
            "strict": stack.enter_context(
                patch.object(mcp_core, "_resolve_session_key_strict", return_value=strict)
            ),
            "deny": stack.enter_context(
                patch.object(mcp_core, "_deny_channel_agent_messaging", return_value=deny)
            ),
            "gov_msg": stack.enter_context(
                patch.object(mcp_core, "_vet_messaging_governance", return_value=gov_msg)
            ),
            "gov_chan": stack.enter_context(
                patch.object(mcp_core, "_vet_channel_governance", return_value=gov_chan)
            ),
            "post": stack.enter_context(
                patch.object(
                    mcp_core,
                    "_post",
                    return_value=(post if post is not None else {"ok": True, "channel_id": "C1"}),
                )
            ),
        }


def test_handler_requires_a_name() -> None:
    with _tool_mcp_core() as m:
        out = create_code_channel("create_code_channel", {"name": "   "})
        assert out.startswith("Error:")
        m["post"].assert_not_called()


def test_handler_refuses_without_a_strict_identity() -> None:
    """A lenient identity is an ancestor walk: a sub-agent would create a channel as
    its parent's session."""
    with _tool_mcp_core(strict="") as m:
        out = create_code_channel("create_code_channel", {"name": "task"})
        assert out.startswith("Error:")
        assert "verify caller identity" in out
        m["post"].assert_not_called()


def test_handler_blocks_a_channel_agent() -> None:
    with _tool_mcp_core(strict="channel:C9:agent1", deny="Error: channel agents cannot") as m:
        out = create_code_channel("create_code_channel", {"name": "task"})
        assert out == "Error: channel agents cannot"
        m["post"].assert_not_called()


def test_handler_honours_messaging_governance_denial() -> None:
    with _tool_mcp_core(gov_msg="messaging disabled") as m:
        out = create_code_channel("create_code_channel", {"name": "task"})
        assert out == "Error: messaging disabled"
        m["post"].assert_not_called()


def test_handler_vets_the_slack_transport() -> None:
    with _tool_mcp_core(gov_chan="slack denied") as m:
        out = create_code_channel("create_code_channel", {"name": "task"})
        assert out == "Error: slack denied"
        m["post"].assert_not_called()


def test_handler_posts_under_the_verified_key_and_reports_success() -> None:
    with _tool_mcp_core(strict="slack:abc", post={"ok": True, "channel_id": "C123"}) as m:
        out = create_code_channel("create_code_channel", {"name": "task"})
        assert out == "Code channel created: <#C123>"
        # The request is sent under the key the strict gate returned, never re-resolved.
        m["post"].assert_called_once()
        args, kwargs = m["post"].call_args
        assert args[0] == "/api/create-code-channel"
        assert args[1] == {"name": "task", "repo": ""}
        assert kwargs["session_key"] == "slack:abc"


def test_handler_surfaces_a_creation_error() -> None:
    with _tool_mcp_core(post={"ok": False, "error": "code_channels_off"}):
        out = create_code_channel("create_code_channel", {"name": "task"})
        assert out.startswith("Error:")
        assert "code_channels_off" in out


def test_handler_steers_in_place_when_already_in_a_code_channel() -> None:
    """The loop-guard refusal is surfaced as guidance, not a failure: no ``Error:``
    prefix (so it is not logged as a tool failure or retried), and it tells the agent
    to keep working in the channel it is already in."""
    with _tool_mcp_core(
        post={
            "ok": False,
            "code": "already_in_code_channel",
            "error": "could not create code channel: already_in_code_channel",
        }
    ):
        out = create_code_channel("create_code_channel", {"name": "task"})
        assert not out.startswith("Error:")
        assert "already in a code channel" in out.lower()


def test_handler_notes_a_partial_invite() -> None:
    with _tool_mcp_core(
        post={
            "ok": True,
            "channel_id": "C1",
            "invite_ok": False,
            "invite_error": "cant_invite_self",
        }
    ):
        out = create_code_channel("create_code_channel", {"name": "task"})
        assert out.startswith("Code channel created: <#C1>")
        assert "cant_invite_self" in out


# ── Gateway endpoint: internal-secret only, error mapping ──


def _make_app(state, *, internal_auth: bool) -> web.Application:
    @web.middleware
    async def _mw(request, handler):
        if internal_auth:
            request["internal_auth"] = True
        return await handler(request)

    app = web.Application(middlewares=[_mw])
    app.router.add_post("/api/create-code-channel", api_create_code_channel)
    app["state"] = state
    return app


@pytest.mark.asyncio
async def test_endpoint_refuses_a_non_internal_caller() -> None:
    state = SimpleNamespace(_create_code_channel=AsyncMock())
    app = _make_app(state, internal_auth=False)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/api/create-code-channel", json={"name": "task"})
        assert resp.status == 403
        assert (await resp.json())["code"] == "auth_required"
    state._create_code_channel.assert_not_awaited()


@pytest.mark.asyncio
async def test_endpoint_requires_a_name() -> None:
    state = SimpleNamespace(_create_code_channel=AsyncMock())
    app = _make_app(state, internal_auth=True)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/api/create-code-channel", json={})
        assert resp.status == 400
        assert (await resp.json())["code"] == "name_required"


@pytest.mark.asyncio
async def test_endpoint_503_when_slack_not_wired() -> None:
    state = SimpleNamespace(_create_code_channel=None)
    app = _make_app(state, internal_auth=True)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/api/create-code-channel", json={"name": "task"})
        assert resp.status == 503
        assert (await resp.json())["code"] == "slack_unavailable"


@pytest.mark.asyncio
async def test_endpoint_success_calls_callback_with_empty_caller() -> None:
    callback = AsyncMock(return_value={"ok": True, "channel_id": "C42", "invite_ok": True})
    state = SimpleNamespace(_create_code_channel=callback)
    app = _make_app(state, internal_auth=True)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/api/create-code-channel", json={"name": "task"})
        assert resp.status == 200
        assert (await resp.json())["channel_id"] == "C42"
    # The MCP caller is a session, not a Slack user; the orchestrator resolves the
    # human to invite (the owner), so the endpoint passes an empty caller id. The
    # third arg is the caller's X-Session-Key (empty here — no header in this test).
    callback.assert_awaited_once_with("task", "", "", "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,status",
    [
        ("code_channels_off", 503),
        ("slack_unavailable", 503),
        ("create_failed", 502),
        ("already_in_code_channel", 409),
    ],
)
async def test_endpoint_maps_core_errors(error, status) -> None:
    state = SimpleNamespace(
        _create_code_channel=AsyncMock(return_value={"ok": False, "error": error})
    )
    app = _make_app(state, internal_auth=True)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/api/create-code-channel", json={"name": "task"})
        assert resp.status == status
        assert (await resp.json())["code"] == error


# ── Shared Slack-side core ──


def _make_orch(*, code_channels=True, slack=None, invitees=None, repo=""):
    cfg = SimpleNamespace(
        slack=SimpleNamespace(
            code_channels=code_channels,
            code_channel_invitees=invitees or [],
            code_channel_repo=repo,
        )
    )
    return SimpleNamespace(_cfg=cfg, slack=slack, _code_channels=set(), _owner_id="U_OWNER")


@pytest.mark.asyncio
async def test_core_off_when_code_channels_disabled() -> None:
    from kiro_crew.slack.events import create_code_channel_core

    orch = _make_orch(code_channels=False, slack=MagicMock())
    result = await create_code_channel_core(orch, "task", "U1")
    assert result == {"ok": False, "channel_id": None, "error": "code_channels_off"}


@pytest.mark.asyncio
async def test_core_requires_a_name() -> None:
    from kiro_crew.slack.events import create_code_channel_core

    orch = _make_orch(slack=MagicMock())
    result = await create_code_channel_core(orch, "  ", "U1")
    assert result["error"] == "name_required"


@pytest.mark.asyncio
async def test_core_reports_create_failure() -> None:
    from kiro_crew.slack.events import create_code_channel_core

    slack = MagicMock()
    slack.create_code_channel = AsyncMock(return_value=None)
    orch = _make_orch(slack=slack)
    result = await create_code_channel_core(orch, "task", "U1")
    assert result["error"] == "create_failed"


@pytest.mark.asyncio
async def test_core_happy_path_creates_invites_and_tracks(monkeypatch) -> None:
    from kiro_crew.slack import events

    monkeypatch.setattr(events, "run_config_write", AsyncMock())
    slack = MagicMock()
    slack.create_code_channel = AsyncMock(return_value="C777")
    slack.set_session_status = AsyncMock()
    slack.invite_users = AsyncMock(return_value={"ok": True})
    slack.post_blocks = AsyncMock()
    orch = _make_orch(slack=slack, invitees=["U_AGENT"])

    result = await events.create_code_channel_core(orch, "Fix it", "U_HUMAN")

    assert result["ok"] is True
    assert result["channel_id"] == "C777"
    assert "C777" in orch._code_channels
    slack.create_code_channel.assert_awaited_once()
    # Caller + configured invitees, deduped and in order.
    assert slack.invite_users.call_args.args[1] == ["U_HUMAN", "U_AGENT"]
    slack.post_blocks.assert_awaited_once()


# ── Gateway orchestrator delegation + owner fallback ──


@pytest.mark.asyncio
async def test_orchestrator_falls_back_to_owner_for_the_invite(monkeypatch) -> None:
    from kiro_crew.slack import events
    from kiro_crew.slack.gateway import GatewayOrchestrator

    captured = {}

    async def _fake_core(orch, name, caller_id, *, session_id=None, **_kw):
        captured["name"] = name
        captured["caller_id"] = caller_id
        return {"ok": True, "channel_id": "C1"}

    monkeypatch.setattr(events, "create_code_channel_core", _fake_core)
    fake = SimpleNamespace(_owner_id="U_OWNER")
    result = await GatewayOrchestrator.create_code_channel_for_task(fake, "task", "")
    assert result["ok"] is True
    assert captured["caller_id"] == "U_OWNER"


@pytest.mark.asyncio
async def test_orchestrator_passes_an_explicit_caller_through(monkeypatch) -> None:
    from kiro_crew.slack import events
    from kiro_crew.slack.gateway import GatewayOrchestrator

    captured = {}

    async def _fake_core(orch, name, caller_id, *, session_id=None, **_kw):
        captured["caller_id"] = caller_id
        return {"ok": True, "channel_id": "C1"}

    monkeypatch.setattr(events, "create_code_channel_core", _fake_core)
    fake = SimpleNamespace(_owner_id="U_OWNER")
    await GatewayOrchestrator.create_code_channel_for_task(fake, "task", "U_CALLER")
    assert captured["caller_id"] == "U_CALLER"


# ── Origin link: agent path carries the triggering message ──


@pytest.mark.asyncio
async def test_endpoint_forwards_the_caller_session_key() -> None:
    callback = AsyncMock(return_value={"ok": True, "channel_id": "C42", "invite_ok": True})
    state = SimpleNamespace(_create_code_channel=callback)
    app = _make_app(state, internal_auth=True)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/api/create-code-channel",
            json={"name": "task"},
            headers={"X-Session-Key": "slack:1717.42"},
        )
        assert resp.status == 200
    # The verified session key travels as the third positional arg so the
    # orchestrator can resolve this turn's triggering message for the origin link.
    callback.assert_awaited_once_with("task", "", "slack:1717.42", "")


@pytest.mark.asyncio
async def test_orchestrator_passes_a_remembered_origin_through(monkeypatch) -> None:
    from kiro_crew.slack import events
    from kiro_crew.slack.gateway import GatewayOrchestrator

    captured = {}

    async def _fake_core(
        orch,
        name,
        caller_id,
        *,
        origin_channel_id=None,
        origin_message_ts=None,
        session_id=None,
        repo=None,
    ):
        captured["origin_channel_id"] = origin_channel_id
        captured["origin_message_ts"] = origin_message_ts
        captured["session_id"] = session_id
        return {"ok": True, "channel_id": "C1"}

    monkeypatch.setattr(events, "create_code_channel_core", _fake_core)
    fake = SimpleNamespace(
        _owner_id="U_OWNER",
        _session_origin_msg={"slack:1717.42": ("C_ORIGIN", "1717.42")},
        _resolve_session_code_channel=lambda _sk: "",
    )
    await GatewayOrchestrator.create_code_channel_for_task(fake, "task", "", "slack:1717.42")
    assert captured["origin_channel_id"] == "C_ORIGIN"
    assert captured["origin_message_ts"] == "1717.42"
    # The verified session key is also Slack's idempotency key on create.
    assert captured["session_id"] == "slack:1717.42"


@pytest.mark.asyncio
async def test_orchestrator_omits_origin_when_session_unknown(monkeypatch) -> None:
    from kiro_crew.slack import events
    from kiro_crew.slack.gateway import GatewayOrchestrator

    captured = {"called": False, "origin_channel_id": "SENTINEL"}

    async def _fake_core(
        orch,
        name,
        caller_id,
        *,
        origin_channel_id=None,
        origin_message_ts=None,
        session_id=None,
        repo=None,
    ):
        captured["called"] = True
        captured["origin_channel_id"] = origin_channel_id
        return {"ok": True, "channel_id": "C1"}

    monkeypatch.setattr(events, "create_code_channel_core", _fake_core)
    fake = SimpleNamespace(
        _owner_id="U_OWNER", _session_origin_msg={}, _resolve_session_code_channel=lambda _sk: ""
    )
    # A session_key with no remembered origin creates the channel without an origin.
    await GatewayOrchestrator.create_code_channel_for_task(fake, "task", "", "slack:unknown")
    assert captured["called"] is True
    assert captured["origin_channel_id"] is None


@pytest.mark.asyncio
async def test_orchestrator_refuses_when_session_already_in_a_code_channel(monkeypatch) -> None:
    """Loop guard: the agent must not open a code channel from inside one. A code
    channel is always-on and Slack seeds it with the triggering message, so without
    this the "open a code channel" instruction re-fires the tool every turn and fans
    out unboundedly. Refuse with a stable code; do not call the create core."""
    from kiro_crew.slack import events
    from kiro_crew.slack.gateway import GatewayOrchestrator

    called = {"core": False}

    async def _fake_core(*_a, **_kw):
        called["core"] = True
        return {"ok": True, "channel_id": "C_SHOULD_NOT_HAPPEN"}

    monkeypatch.setattr(events, "create_code_channel_core", _fake_core)
    # The caller's session already resolves to a known code channel.
    fake = SimpleNamespace(
        _owner_id="U_OWNER",
        _session_origin_msg={},
        _resolve_session_code_channel=lambda sk: "C_CURRENT" if sk == "slack:in.cc" else "",
    )
    result = await GatewayOrchestrator.create_code_channel_for_task(fake, "task", "", "slack:in.cc")
    assert result == {"ok": False, "channel_id": "C_CURRENT", "error": "already_in_code_channel"}
    assert called["core"] is False


def test_remember_session_origin_stores_and_is_bounded() -> None:
    from kiro_crew.slack.gateway import GatewayOrchestrator

    fake = SimpleNamespace(_session_origin_msg={})
    GatewayOrchestrator.remember_session_origin(fake, "slack:1", "C_A", "1.0")
    assert fake._session_origin_msg["slack:1"] == ("C_A", "1.0")
    # Missing any of the three is a no-op (not a partial store).
    GatewayOrchestrator.remember_session_origin(fake, "slack:2", "", "2.0")
    assert "slack:2" not in fake._session_origin_msg
    # Bounded: far more than the cap evicts the oldest, never grows without bound.
    for i in range(400):
        GatewayOrchestrator.remember_session_origin(fake, f"k{i}", "C", f"{i}.0")
    assert len(fake._session_origin_msg) <= 256


@pytest.mark.asyncio
async def test_core_retries_without_origin_on_invalid_origin(monkeypatch) -> None:
    from kiro_crew.slack import events

    monkeypatch.setattr(events, "run_config_write", AsyncMock())
    slack = MagicMock()
    # First call (with origin) fails as Slack would on invalid_origin_link (None);
    # the retry without origin succeeds.
    slack.create_code_channel = AsyncMock(side_effect=[None, "C_RETRY"])
    slack.set_session_status = AsyncMock()
    slack.invite_users = AsyncMock(return_value={"ok": True})
    slack.post_blocks = AsyncMock()
    orch = _make_orch(slack=slack)

    result = await events.create_code_channel_core(
        orch, "task", "U1", origin_channel_id="C_STALE", origin_message_ts="9999.0"
    )

    assert result["ok"] is True
    assert result["channel_id"] == "C_RETRY"
    assert slack.create_code_channel.await_count == 2
    # The retry drops the origin params (still private).
    retry_kwargs = slack.create_code_channel.await_args_list[1].kwargs
    assert retry_kwargs.get("origin_channel_id") is None
    assert retry_kwargs.get("origin_message_ts") is None
    assert retry_kwargs.get("is_private") is True


@pytest.mark.asyncio
async def test_core_forwards_session_id_as_idempotency_key() -> None:
    from kiro_crew.slack.events import create_code_channel_core

    slack = MagicMock()
    slack.create_code_channel = AsyncMock(return_value=None)  # short-circuit after create
    orch = _make_orch(slack=slack)
    await create_code_channel_core(orch, "task", "U1", session_id="ses_abc")
    assert slack.create_code_channel.await_args.kwargs.get("session_id") == "ses_abc"


@pytest.mark.asyncio
async def test_core_surfaces_a_documented_slack_error_code() -> None:
    from kiro_crew.slack.events import create_code_channel_core

    slack = MagicMock()
    slack.create_code_channel = AsyncMock(return_value=None)
    # The client captured Slack's documented code on the swallowed failure.
    slack._last_code_channel_error = "missing_scope"
    orch = _make_orch(slack=slack)
    result = await create_code_channel_core(orch, "task", "U1")
    # It surfaces the specific code, not the generic create_failed.
    assert result["error"] == "missing_scope"

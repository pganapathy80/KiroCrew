"""The set_code_channel_view MCP tool: its advertised shape, its handler's identity
and governance gates, the gateway endpoint that backs it, and the orchestrator's
channel resolution.

The tool lets Kiro Crew publish a view tab (agents.conversations.setView) into the
code channel its session is working in. Because it posts to Slack it is held to the
same strict-identity + channel-agent containment bar as create_code_channel, and the
orchestrator refuses to publish unless the caller's session resolves to a KNOWN code
channel — so the agent can never target a channel it names.
"""

from __future__ import annotations

import contextlib
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew.config.loader import KiroCrewConfig, SlackConfig
from kiro_crew.dashboard.handlers import api_set_code_channel_view
from kiro_crew.mcp_tools.messaging import HANDLERS, schemas, set_code_channel_view
from kiro_crew.validation import (
    MCP_CORE_SCHEMAS,
    SET_CODE_CHANNEL_VIEW_SCHEMA,
    ValidationError,
    validate_tool_args,
)

# ── Advertisement / registry agreement ──


def _advertised() -> dict:
    for tool in schemas():
        if tool["name"] == "set_code_channel_view":
            return tool
    raise AssertionError("set_code_channel_view is not advertised")


def test_tool_is_advertised_and_handled() -> None:
    assert "set_code_channel_view" in HANDLERS
    tool = _advertised()
    assert tool["inputSchema"]["required"] == ["view_type"]


def test_advertised_properties_are_all_validated() -> None:
    known = {spec.name for spec in SET_CODE_CHANNEL_VIEW_SCHEMA.fields}
    advertised = set(_advertised()["inputSchema"]["properties"])
    missing = sorted(advertised - known)
    assert not missing, f"advertised but unvalidatable, so those calls fail: {missing}"


def test_schema_is_registered() -> None:
    assert MCP_CORE_SCHEMAS["set_code_channel_view"] is SET_CODE_CHANNEL_VIEW_SCHEMA


def test_validation_rejects_a_bad_view_type() -> None:
    with pytest.raises(ValidationError):
        validate_tool_args({"view_type": "pdf"}, SET_CODE_CHANNEL_VIEW_SCHEMA)
    cleaned = validate_tool_args(
        {"view_type": "html", "content": "<b>hi</b>"}, SET_CODE_CHANNEL_VIEW_SCHEMA
    )
    assert cleaned["view_type"] == "html"


# ── Tool handler: identity + governance gates ──


@contextlib.contextmanager
def _tool_mcp_core(
    *, strict="slack:1712793600.1", deny=None, gov_msg=None, gov_chan=None, post=None
):
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
                    mcp_core, "_post", return_value=(post if post is not None else {"ok": True})
                )
            ),
        }


def test_handler_rejects_a_bad_view_type() -> None:
    with _tool_mcp_core() as m:
        out = set_code_channel_view("set_code_channel_view", {"view_type": "pdf"})
        assert out.startswith("Error:")
        m["post"].assert_not_called()


def test_handler_refuses_without_a_strict_identity() -> None:
    with _tool_mcp_core(strict="") as m:
        out = set_code_channel_view("set_code_channel_view", {"view_type": "html"})
        assert out.startswith("Error:")
        assert "verify caller identity" in out
        m["post"].assert_not_called()


def test_handler_blocks_a_channel_agent() -> None:
    with _tool_mcp_core(strict="channel:C9:agent1", deny="Error: channel agents cannot") as m:
        out = set_code_channel_view("set_code_channel_view", {"view_type": "html"})
        assert out == "Error: channel agents cannot"
        m["post"].assert_not_called()


def test_handler_honours_messaging_governance_denial() -> None:
    with _tool_mcp_core(gov_msg="messaging disabled") as m:
        out = set_code_channel_view("set_code_channel_view", {"view_type": "html"})
        assert out == "Error: messaging disabled"
        m["post"].assert_not_called()


def test_handler_vets_the_slack_transport() -> None:
    with _tool_mcp_core(gov_chan="slack denied") as m:
        out = set_code_channel_view("set_code_channel_view", {"view_type": "html"})
        assert out == "Error: slack denied"
        m["post"].assert_not_called()


def test_handler_posts_under_the_verified_key_and_reports_success() -> None:
    with _tool_mcp_core(strict="slack:abc", post={"ok": True}) as m:
        out = set_code_channel_view(
            "set_code_channel_view",
            {"view_type": "html", "content": "<h1>ok</h1>", "name": "Status", "view_key": "k1"},
        )
        assert "html view" in out
        m["post"].assert_called_once()
        args, kwargs = m["post"].call_args
        assert args[0] == "/api/set-code-channel-view"
        assert args[1]["view_type"] == "html"
        assert args[1]["content"] == "<h1>ok</h1>"
        assert kwargs["session_key"] == "slack:abc"


def test_handler_surfaces_a_view_error() -> None:
    with _tool_mcp_core(post={"ok": False, "error": "not_a_code_channel"}):
        out = set_code_channel_view("set_code_channel_view", {"view_type": "html"})
        assert out.startswith("Error:")
        assert "not_a_code_channel" in out


# ── Gateway endpoint: internal-secret only, error mapping ──


def _make_app(state, *, internal_auth: bool) -> web.Application:
    @web.middleware
    async def _mw(request, handler):
        if internal_auth:
            request["internal_auth"] = True
        return await handler(request)

    app = web.Application(middlewares=[_mw])
    app.router.add_post("/api/set-code-channel-view", api_set_code_channel_view)
    app["state"] = state
    return app


@pytest.mark.asyncio
async def test_endpoint_refuses_a_non_internal_caller() -> None:
    state = SimpleNamespace(_set_code_channel_view=AsyncMock())
    app = _make_app(state, internal_auth=False)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/api/set-code-channel-view", json={"view_type": "html"})
        assert resp.status == 403
        assert (await resp.json())["code"] == "auth_required"
    state._set_code_channel_view.assert_not_awaited()


@pytest.mark.asyncio
async def test_endpoint_rejects_a_bad_view_type() -> None:
    state = SimpleNamespace(_set_code_channel_view=AsyncMock())
    app = _make_app(state, internal_auth=True)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/api/set-code-channel-view", json={"view_type": "pdf"})
        assert resp.status == 400
        assert (await resp.json())["code"] == "invalid_view_type"


@pytest.mark.asyncio
async def test_endpoint_503_when_slack_not_wired() -> None:
    state = SimpleNamespace(_set_code_channel_view=None)
    app = _make_app(state, internal_auth=True)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/api/set-code-channel-view", json={"view_type": "html"})
        assert resp.status == 503
        assert (await resp.json())["code"] == "slack_unavailable"


@pytest.mark.asyncio
async def test_endpoint_forwards_session_and_view_args() -> None:
    callback = AsyncMock(return_value={"ok": True})
    state = SimpleNamespace(_set_code_channel_view=callback)
    app = _make_app(state, internal_auth=True)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/api/set-code-channel-view",
            json={"view_type": "html", "content": "<b>x</b>", "name": "S", "view_key": "k"},
            headers={"X-Session-Key": "slack:1717.42"},
        )
        assert resp.status == 200
    callback.assert_awaited_once_with("slack:1717.42", "html", "<b>x</b>", None, "S", "k")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,status",
    [
        ("not_a_code_channel", 409),
        ("slack_unavailable", 503),
        ("code_channels_off", 503),
        ("view_failed", 502),
    ],
)
async def test_endpoint_maps_errors(error, status) -> None:
    state = SimpleNamespace(
        _set_code_channel_view=AsyncMock(return_value={"ok": False, "error": error})
    )
    app = _make_app(state, internal_auth=True)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/api/set-code-channel-view", json={"view_type": "html"})
        assert resp.status == status
        assert (await resp.json())["code"] == error


# ── Gateway orchestrator: channel resolution + refusal ──


def _cfg(*, enabled: bool = True) -> KiroCrewConfig:
    return KiroCrewConfig(slack=SlackConfig(code_channels=enabled))


@pytest.mark.asyncio
async def test_orchestrator_is_inert_with_the_flag_off() -> None:
    """A code channel restored from when the flag was on is not publishable once
    it is off."""
    from kiro_crew.slack.gateway import GatewayOrchestrator

    sessions = SimpleNamespace(
        get_slack_link=lambda k: (None, "C_CODE"),
        get_origin_link=lambda k: None,
    )
    fake = SimpleNamespace(
        slack=AsyncMock(), sessions=sessions, _code_channels={"C_CODE"}, _cfg=_cfg(enabled=False)
    )
    fake._resolve_session_code_channel = types.MethodType(
        GatewayOrchestrator._resolve_session_code_channel, fake
    )
    result = await GatewayOrchestrator.set_code_channel_view_for_session(
        fake, "slack:1", "html", content="<b>x</b>"
    )
    assert result == {"ok": False, "error": "code_channels_off"}
    fake.slack.set_code_channel_view.assert_not_awaited()


@pytest.mark.asyncio
async def test_orchestrator_refuses_when_session_is_not_a_code_channel() -> None:
    from kiro_crew.slack.gateway import GatewayOrchestrator

    # Session resolves to an ordinary channel (not in _code_channels) -> refused.
    sessions = SimpleNamespace(
        get_slack_link=lambda k: (None, "C_ORDINARY"),
        get_origin_link=lambda k: None,
    )
    fake = SimpleNamespace(slack=AsyncMock(), sessions=sessions, _code_channels=set(), _cfg=_cfg())
    fake._resolve_session_code_channel = types.MethodType(
        GatewayOrchestrator._resolve_session_code_channel, fake
    )
    result = await GatewayOrchestrator.set_code_channel_view_for_session(
        fake, "slack:1", "html", content="<b>x</b>"
    )
    assert result == {"ok": False, "error": "not_a_code_channel"}
    fake.slack.set_code_channel_view.assert_not_awaited()


@pytest.mark.asyncio
async def test_orchestrator_publishes_to_the_resolved_code_channel() -> None:
    from kiro_crew.slack.gateway import GatewayOrchestrator

    sessions = SimpleNamespace(
        get_slack_link=lambda k: (None, "C_CODE"),
        get_origin_link=lambda k: None,
    )
    slack = AsyncMock()
    slack.set_code_channel_view = AsyncMock(return_value={"view_id": "V1"})
    fake = SimpleNamespace(slack=slack, sessions=sessions, _code_channels={"C_CODE"}, _cfg=_cfg())
    fake._resolve_session_code_channel = types.MethodType(
        GatewayOrchestrator._resolve_session_code_channel, fake
    )
    result = await GatewayOrchestrator.set_code_channel_view_for_session(
        fake, "slack:1", "html", content="<b>x</b>", name="Status"
    )
    assert result["ok"] is True
    slack.set_code_channel_view.assert_awaited_once()
    assert slack.set_code_channel_view.await_args.args[0] == "C_CODE"
    assert slack.set_code_channel_view.await_args.kwargs["view_type"] == "html"


@pytest.mark.asyncio
async def test_orchestrator_surfaces_the_slack_error_on_failure() -> None:
    from kiro_crew.slack.gateway import GatewayOrchestrator

    sessions = SimpleNamespace(
        get_slack_link=lambda k: (None, "C_CODE"),
        get_origin_link=lambda k: None,
    )
    slack = AsyncMock()
    slack.set_code_channel_view = AsyncMock(return_value=None)  # Slack failure, swallowed
    slack._last_code_channel_error = "missing_scope"
    fake = SimpleNamespace(slack=slack, sessions=sessions, _code_channels={"C_CODE"}, _cfg=_cfg())
    fake._resolve_session_code_channel = types.MethodType(
        GatewayOrchestrator._resolve_session_code_channel, fake
    )
    result = await GatewayOrchestrator.set_code_channel_view_for_session(
        fake, "slack:1", "diff", content="--- a"
    )
    assert result == {"ok": False, "error": "missing_scope"}

"""The publish_plan_canvas MCP tool: its advertised shape, its handler's identity
and governance gates, the gateway endpoint that backs it, and the orchestrator's
create-vs-update behaviour.

The tool lets Kiro Crew create (or update) a comment-only plan canvas in the code
channel its session is working in. The first canvas is minted via Slack's general
Canvas API (canvases.create) and attached as a comment-only view tab; a later call
with the returned canvas_id rewrites its content in place. Because it posts to Slack
it is held to the same strict-identity + channel-agent containment bar as
create_code_channel, and the orchestrator refuses unless the caller's session
resolves to a KNOWN code channel.
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
from kiro_crew.dashboard.handlers import api_publish_plan_canvas
from kiro_crew.mcp_tools.messaging import HANDLERS, publish_plan_canvas, schemas
from kiro_crew.validation import (
    MCP_CORE_SCHEMAS,
    PUBLISH_PLAN_CANVAS_SCHEMA,
    ValidationError,
    validate_tool_args,
)

# ── Advertisement / registry agreement ──


def _advertised() -> dict:
    for tool in schemas():
        if tool["name"] == "publish_plan_canvas":
            return tool
    raise AssertionError("publish_plan_canvas is not advertised")


def test_tool_is_advertised_and_handled() -> None:
    assert "publish_plan_canvas" in HANDLERS
    tool = _advertised()
    assert tool["inputSchema"]["required"] == ["content"]


def test_advertised_properties_are_all_validated() -> None:
    known = {spec.name for spec in PUBLISH_PLAN_CANVAS_SCHEMA.fields}
    advertised = set(_advertised()["inputSchema"]["properties"])
    missing = sorted(advertised - known)
    assert not missing, f"advertised but unvalidatable, so those calls fail: {missing}"


def test_schema_is_registered() -> None:
    assert MCP_CORE_SCHEMAS["publish_plan_canvas"] is PUBLISH_PLAN_CANVAS_SCHEMA


def test_validation_requires_content() -> None:
    with pytest.raises(ValidationError):
        validate_tool_args({"title": "Plan"}, PUBLISH_PLAN_CANVAS_SCHEMA)
    cleaned = validate_tool_args({"content": "# Plan"}, PUBLISH_PLAN_CANVAS_SCHEMA)
    assert cleaned["content"] == "# Plan"


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


def test_handler_requires_content() -> None:
    with _tool_mcp_core() as m:
        out = publish_plan_canvas("publish_plan_canvas", {"content": "   "})
        assert out.startswith("Error:")
        m["post"].assert_not_called()


def test_handler_refuses_without_a_strict_identity() -> None:
    with _tool_mcp_core(strict="") as m:
        out = publish_plan_canvas("publish_plan_canvas", {"content": "# Plan"})
        assert out.startswith("Error:")
        assert "verify caller identity" in out
        m["post"].assert_not_called()


def test_handler_blocks_a_channel_agent() -> None:
    with _tool_mcp_core(strict="channel:C9:agent1", deny="Error: channel agents cannot") as m:
        out = publish_plan_canvas("publish_plan_canvas", {"content": "# Plan"})
        assert out == "Error: channel agents cannot"
        m["post"].assert_not_called()


def test_handler_honours_messaging_governance_denial() -> None:
    with _tool_mcp_core(gov_msg="messaging disabled") as m:
        out = publish_plan_canvas("publish_plan_canvas", {"content": "# Plan"})
        assert out == "Error: messaging disabled"
        m["post"].assert_not_called()


def test_handler_vets_the_slack_transport() -> None:
    with _tool_mcp_core(gov_chan="slack denied") as m:
        out = publish_plan_canvas("publish_plan_canvas", {"content": "# Plan"})
        assert out == "Error: slack denied"
        m["post"].assert_not_called()


def test_handler_posts_create_under_the_verified_key() -> None:
    with _tool_mcp_core(
        strict="slack:abc", post={"ok": True, "canvas_id": "F1", "updated": False}
    ) as m:
        out = publish_plan_canvas(
            "publish_plan_canvas",
            {"content": "# Plan", "title": "Design", "view_name": "Plan"},
        )
        assert "Published plan canvas" in out
        assert "F1" in out
        m["post"].assert_called_once()
        args, kwargs = m["post"].call_args
        assert args[0] == "/api/publish-plan-canvas"
        assert args[1]["content"] == "# Plan"
        assert args[1]["title"] == "Design"
        assert "canvas_id" not in args[1]  # create path omits it
        assert kwargs["session_key"] == "slack:abc"


def test_handler_forwards_canvas_id_on_update_and_reports_updated() -> None:
    with _tool_mcp_core(post={"ok": True, "canvas_id": "F9", "updated": True}) as m:
        out = publish_plan_canvas(
            "publish_plan_canvas", {"content": "# Plan v2", "canvas_id": "F9"}
        )
        assert "Updated plan canvas" in out
        assert m["post"].call_args.args[1]["canvas_id"] == "F9"


def test_handler_surfaces_a_canvas_error() -> None:
    with _tool_mcp_core(post={"ok": False, "error": "not_a_code_channel"}):
        out = publish_plan_canvas("publish_plan_canvas", {"content": "# Plan"})
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
    app.router.add_post("/api/publish-plan-canvas", api_publish_plan_canvas)
    app["state"] = state
    return app


@pytest.mark.asyncio
async def test_endpoint_refuses_a_non_internal_caller() -> None:
    state = SimpleNamespace(_publish_plan_canvas=AsyncMock())
    app = _make_app(state, internal_auth=False)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/api/publish-plan-canvas", json={"content": "# P"})
        assert resp.status == 403
        assert (await resp.json())["code"] == "auth_required"
    state._publish_plan_canvas.assert_not_awaited()


@pytest.mark.asyncio
async def test_endpoint_requires_content() -> None:
    state = SimpleNamespace(_publish_plan_canvas=AsyncMock())
    app = _make_app(state, internal_auth=True)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/api/publish-plan-canvas", json={"content": "  "})
        assert resp.status == 400
        assert (await resp.json())["code"] == "content_required"


@pytest.mark.asyncio
async def test_endpoint_503_when_slack_not_wired() -> None:
    state = SimpleNamespace(_publish_plan_canvas=None)
    app = _make_app(state, internal_auth=True)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/api/publish-plan-canvas", json={"content": "# P"})
        assert resp.status == 503
        assert (await resp.json())["code"] == "slack_unavailable"


@pytest.mark.asyncio
async def test_endpoint_forwards_session_and_canvas_args() -> None:
    callback = AsyncMock(return_value={"ok": True, "canvas_id": "F1", "updated": False})
    state = SimpleNamespace(_publish_plan_canvas=callback)
    app = _make_app(state, internal_auth=True)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/api/publish-plan-canvas",
            json={"content": "# P", "title": "T", "canvas_id": "F1", "view_name": "Plan"},
            headers={"X-Session-Key": "slack:1717.42"},
        )
        assert resp.status == 200
        assert (await resp.json())["canvas_id"] == "F1"
    callback.assert_awaited_once_with("slack:1717.42", "# P", "T", "F1", "Plan")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,status",
    [
        ("not_a_code_channel", 409),
        ("canvas_not_owned", 409),
        ("slack_unavailable", 503),
        ("code_channels_off", 503),
        ("canvas_not_found", 502),
        ("canvas_creation_failed", 502),
    ],
)
async def test_endpoint_maps_errors(error, status) -> None:
    state = SimpleNamespace(
        _publish_plan_canvas=AsyncMock(return_value={"ok": False, "error": error})
    )
    app = _make_app(state, internal_auth=True)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/api/publish-plan-canvas", json={"content": "# P"})
        assert resp.status == status
        assert (await resp.json())["code"] == error


# ── Gateway orchestrator: channel resolution, create vs update ──


def _orch_with_session_channel(slack, *, code_channels, canvases=None, enabled=True):
    from kiro_crew.slack.gateway import GatewayOrchestrator

    sessions = SimpleNamespace(
        get_slack_link=lambda k: (None, "C_CODE"),
        get_origin_link=lambda k: None,
    )
    fake = SimpleNamespace(
        slack=slack,
        sessions=sessions,
        _code_channels=code_channels,
        _code_channel_canvases=canvases if canvases is not None else {},
        _cfg=KiroCrewConfig(slack=SlackConfig(code_channels=enabled)),
    )
    fake._resolve_session_code_channel = types.MethodType(
        GatewayOrchestrator._resolve_session_code_channel, fake
    )
    return fake


@pytest.mark.asyncio
async def test_orchestrator_refuses_when_session_is_not_a_code_channel() -> None:
    from kiro_crew.slack.gateway import GatewayOrchestrator

    slack = AsyncMock()
    fake = _orch_with_session_channel(slack, code_channels=set())  # C_CODE not registered
    result = await GatewayOrchestrator.publish_plan_canvas_for_session(fake, "slack:1", "# P")
    assert result == {"ok": False, "error": "not_a_code_channel"}
    slack.create_canvas.assert_not_awaited()


@pytest.mark.asyncio
async def test_orchestrator_creates_canvas_then_attaches_comment_only_view() -> None:
    from kiro_crew.slack.gateway import GatewayOrchestrator

    slack = AsyncMock()
    slack.create_canvas = AsyncMock(return_value="F123")
    slack.set_code_channel_view = AsyncMock(return_value={"view_id": "V1", "canvas_id": "F123"})
    fake = _orch_with_session_channel(slack, code_channels={"C_CODE"})
    result = await GatewayOrchestrator.publish_plan_canvas_for_session(
        fake, "slack:1", "# Plan", title="Design", view_name="Plan"
    )
    assert result["ok"] is True
    assert result["canvas_id"] == "F123"
    assert result["updated"] is False
    slack.create_canvas.assert_awaited_once_with(title="Design", content="# Plan")
    # attached to the resolved code channel as a comment-only canvas view
    kwargs = slack.set_code_channel_view.await_args.kwargs
    assert slack.set_code_channel_view.await_args.args[0] == "C_CODE"
    assert kwargs["view_type"] == "canvas"
    assert kwargs["canvas_id"] == "F123"
    assert kwargs["access_level"] == "comment"


@pytest.mark.asyncio
async def test_orchestrator_updates_existing_canvas_without_recreating() -> None:
    from kiro_crew.slack.gateway import GatewayOrchestrator

    slack = AsyncMock()
    slack.set_canvas_content = AsyncMock(return_value={"ok": True, "sections_changed_count": 1})
    fake = _orch_with_session_channel(slack, code_channels={"C_CODE"}, canvases={"C_CODE": {"F9"}})
    result = await GatewayOrchestrator.publish_plan_canvas_for_session(
        fake, "slack:1", "# Plan v2", canvas_id="F9"
    )
    assert result == {"ok": True, "canvas_id": "F9", "updated": True, "error": None}
    slack.set_canvas_content.assert_awaited_once_with("C_CODE", "F9", "# Plan v2")
    slack.create_canvas.assert_not_awaited()


@pytest.mark.asyncio
async def test_orchestrator_surfaces_create_failure() -> None:
    from kiro_crew.slack.gateway import GatewayOrchestrator

    slack = AsyncMock()
    slack.create_canvas = AsyncMock(return_value=None)  # canvases.create failed, swallowed
    slack._last_code_channel_error = "missing_scope"
    fake = _orch_with_session_channel(slack, code_channels={"C_CODE"})
    result = await GatewayOrchestrator.publish_plan_canvas_for_session(fake, "slack:1", "# P")
    assert result == {"ok": False, "error": "missing_scope"}
    slack.set_code_channel_view.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "canvases",
    [{}, {"C_OTHER": {"F9"}}],
    ids=["never-created", "created-in-another-channel"],
)
async def test_orchestrator_refuses_to_update_a_canvas_it_did_not_create(canvases) -> None:
    """canvas_id comes from the agent, and the bot token can write canvases far
    beyond this channel, so only a canvas this tool created here is rewritten."""
    from kiro_crew.slack.gateway import GatewayOrchestrator

    slack = AsyncMock()
    fake = _orch_with_session_channel(slack, code_channels={"C_CODE"}, canvases=canvases)
    result = await GatewayOrchestrator.publish_plan_canvas_for_session(
        fake, "slack:1", "# Overwrite", canvas_id="F9"
    )
    assert result == {"ok": False, "error": "canvas_not_owned"}
    slack.set_canvas_content.assert_not_awaited()
    slack.create_canvas.assert_not_awaited()


@pytest.mark.asyncio
async def test_orchestrator_can_update_the_canvas_it_just_created() -> None:
    from kiro_crew.slack.gateway import GatewayOrchestrator

    slack = AsyncMock()
    slack.create_canvas = AsyncMock(return_value="F123")
    slack.set_code_channel_view = AsyncMock(return_value={"view_id": "V1"})
    slack.set_canvas_content = AsyncMock(return_value={"ok": True})
    fake = _orch_with_session_channel(slack, code_channels={"C_CODE"})
    with patch("kiro_crew.slack.gateway.save_channel") as save:
        created = await GatewayOrchestrator.publish_plan_canvas_for_session(
            fake, "slack:1", "# Plan"
        )
    assert fake._code_channel_canvases == {"C_CODE": {"F123"}}
    save.assert_called_once_with(fake, "C_CODE")
    updated = await GatewayOrchestrator.publish_plan_canvas_for_session(
        fake, "slack:1", "# Plan v2", canvas_id=created["canvas_id"]
    )
    assert updated["ok"] is True and updated["updated"] is True


@pytest.mark.asyncio
async def test_orchestrator_is_inert_with_the_flag_off() -> None:
    from kiro_crew.slack.gateway import GatewayOrchestrator

    slack = AsyncMock()
    fake = _orch_with_session_channel(
        slack, code_channels={"C_CODE"}, canvases={"C_CODE": {"F9"}}, enabled=False
    )
    result = await GatewayOrchestrator.publish_plan_canvas_for_session(
        fake, "slack:1", "# P", canvas_id="F9"
    )
    assert result == {"ok": False, "error": "code_channels_off"}
    slack.set_canvas_content.assert_not_awaited()
    slack.create_canvas.assert_not_awaited()


# ── Client: create_canvas hits canvases.create and returns the id ──


@pytest.mark.asyncio
async def test_create_canvas_calls_canvases_create_and_returns_id() -> None:
    from kiro_crew.slack.client import RealSlackClient

    client = RealSlackClient.__new__(RealSlackClient)
    client._last_code_channel_error = ""
    client._web = SimpleNamespace(api_call=AsyncMock(return_value={"ok": True, "canvas_id": "F42"}))
    cid = await client.create_canvas(title="Plan", content="# hi")
    assert cid == "F42"
    method, kwargs = client._web.api_call.await_args.args[0], client._web.api_call.await_args.kwargs
    assert method == "canvases.create"
    assert kwargs["json"]["document_content"] == {"type": "markdown", "markdown": "# hi"}
    assert kwargs["json"]["title"] == "Plan"


@pytest.mark.asyncio
async def test_create_canvas_records_error_and_returns_none_on_failure() -> None:
    from kiro_crew.slack.client import RealSlackClient

    client = RealSlackClient.__new__(RealSlackClient)
    client._last_code_channel_error = ""
    err = Exception("boom")
    err.response = {"error": "missing_scope"}  # type: ignore[attr-defined]
    client._web = SimpleNamespace(api_call=AsyncMock(side_effect=err))
    cid = await client.create_canvas(content="# hi")
    assert cid is None
    assert client._last_code_channel_error == "missing_scope"

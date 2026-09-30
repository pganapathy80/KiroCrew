"""Client-layer tests for the code-channel canvas method (setCanvasContent)
and attaching a canvas as a view via setView.

These exercise ``RealSlackClient`` directly against a recording Web stub, the
same pattern ``test_slack_agent_sessions`` uses. Creating a canvas
(``canvases.create``, Slack's general Canvas API) is covered with the
``publish_plan_canvas`` tool in ``test_mcp_publish_plan_canvas``.
"""

from __future__ import annotations

from typing import Any

import pytest

from kiro_crew.slack.client import RealSlackClient


class _RecordingWeb:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
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


def _client(web: _RecordingWeb) -> RealSlackClient:
    c = RealSlackClient.__new__(RealSlackClient)
    c._web = web  # type: ignore[attr-defined]
    c._last_code_channel_error = ""  # type: ignore[attr-defined]
    return c


class TestCanvasView:
    @pytest.mark.asyncio
    async def test_setview_canvas_sends_canvas_id_and_access_level(self) -> None:
        web = _RecordingWeb()
        web.set_response("agents.conversations.setView", {"view_id": "V1", "type": "canvas"})
        await _client(web).set_code_channel_view(
            "C1",
            view_type="canvas",
            canvas_id="F123",
            access_level="comment",
            view_key="plan",
            name="Plan",
        )
        method, body = web.calls[-1]
        assert method == "agents.conversations.setView"
        assert body["type"] == "canvas"
        assert body["canvas_id"] == "F123"
        assert body["access_level"] == "comment"
        assert body["view_key"] == "plan"

    @pytest.mark.asyncio
    async def test_non_canvas_view_omits_canvas_fields(self) -> None:
        web = _RecordingWeb()
        await _client(web).set_code_channel_view("C1", view_type="html", content="<p>hi</p>")
        _method, body = web.calls[-1]
        assert "canvas_id" not in body
        assert "access_level" not in body


class TestSetCanvasContent:
    @pytest.mark.asyncio
    async def test_sends_full_content(self) -> None:
        web = _RecordingWeb()
        web.set_response(
            "agents.conversations.setCanvasContent", {"ok": True, "sections_changed_count": 2}
        )
        out = await _client(web).set_canvas_content("C1", "F1", "# Plan\n\n1. step")
        method, body = web.calls[-1]
        assert method == "agents.conversations.setCanvasContent"
        assert body["channel"] == "C1"
        assert body["canvas_id"] == "F1"
        assert body["content"] == "# Plan\n\n1. step"
        assert out is not None and out["sections_changed_count"] == 2

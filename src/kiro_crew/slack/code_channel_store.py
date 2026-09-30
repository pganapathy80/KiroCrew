"""On-disk record of the Slack code channels this gateway works in.

Slack can say whether a channel is a code channel (``record_channel``), but not
which repo on this machine it works in, the diff baseline taken when it opened,
the origin message it posts its result back to, or the ts its one continuous
session is keyed on. Those live in the
orchestrator's in-memory maps and are lost on a gateway restart, after which the
agent in an existing code channel would not know where its code is. This store
persists exactly that per-channel record so a restart restores it.

The in-memory maps stay the source of truth; the store is written from them
(:func:`save_channel`) and read back into them once at startup
(:func:`restore_into`). A missing or corrupt file means "no records", never a
crash. One gateway process owns the file (the gateway singleton lock), but its
writes run on worker threads, so each read-modify-write holds a module lock.
The agent cannot write the file (``security._WRITE_PROTECTED_HOME_PATHS``).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from kiro_crew.atomic_write import atomic_write
from kiro_crew.config.loader import config_dir

logger = logging.getLogger(__name__)

_STORE_FILENAME = "slack-code-channels.json"

#: First line of the working-context block prepended to a code-channel turn. The
#: block runs to the first blank line; anything that shows a turn's text back to
#: people strips it with :func:`strip_code_channel_context`.
CODE_CHANNEL_CONTEXT_HEADER = "[Code channel working context]"

# Serialises save/forget: both read, modify and rewrite the whole file from worker
# threads, and two interleaved writers would lose one update or revive a record an
# archive just dropped.
_WRITE_LOCK = threading.Lock()


def _store_path() -> Path:
    return config_dir() / _STORE_FILENAME


def _load(path: Path) -> dict[str, dict[str, Any]]:
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load code channel store: %s", exc)
        return {}
    channels = data.get("channels") if isinstance(data, dict) else None
    if not isinstance(channels, dict):
        return {}
    return {str(k): v for k, v in channels.items() if isinstance(v, dict)}


def _save(path: Path, records: dict[str, dict[str, Any]]) -> None:
    try:
        atomic_write(path, json.dumps({"channels": records}, indent=2, ensure_ascii=False))
    except Exception as exc:
        logger.warning("Failed to persist code channel store: %s", exc)


def restore_into(orch: Any, path: Path | None = None) -> int:
    """Refill the orchestrator's code-channel maps from disk; returns the count.

    Mutates the existing set and dicts in place, so references other components
    already hold to them (the dashboard's code-channel check) stay valid.
    """
    records = _load(path or _store_path())
    for channel, rec in records.items():
        orch._code_channels.add(channel)
        repo = rec.get("repo")
        if isinstance(repo, str) and repo:
            orch._code_channel_repo_by_id[channel] = repo
        diff_base = rec.get("diff_base")
        if isinstance(diff_base, str):
            orch._code_channel_diff_base[channel] = diff_base
        owned = getattr(orch, "_owned_code_channels", None)
        if rec.get("owned") is True and owned is not None:
            owned.add(channel)
        session_ts = rec.get("session_ts")
        session_map = getattr(orch, "_code_channel_session_ts", None)
        if isinstance(session_ts, str) and session_ts and session_map is not None:
            session_map[channel] = session_ts
        origin = rec.get("origin")
        if (
            isinstance(origin, list)
            and len(origin) == 2
            and all(isinstance(x, str) for x in origin)
        ):
            orch._code_channel_origin[channel] = (origin[0], origin[1])
    return len(records)


def save_channel(orch: Any, channel: str, path: Path | None = None) -> None:
    """Write *channel*'s current in-memory record to disk (best-effort)."""
    if not channel or channel not in getattr(orch, "_code_channels", ()):
        return
    rec: dict[str, Any] = {}
    repo = getattr(orch, "_code_channel_repo_by_id", {}).get(channel)
    if repo:
        rec["repo"] = repo
    base_map = getattr(orch, "_code_channel_diff_base", {})
    if channel in base_map:
        rec["diff_base"] = base_map[channel]
    origin = getattr(orch, "_code_channel_origin", {}).get(channel)
    if origin:
        rec["origin"] = [origin[0], origin[1]]
    session_ts = getattr(orch, "_code_channel_session_ts", {}).get(channel)
    if session_ts:
        rec["session_ts"] = session_ts
    if channel in getattr(orch, "_owned_code_channels", ()):
        rec["owned"] = True
    target = path or _store_path()
    with _WRITE_LOCK:
        # Re-checked under the lock: an archive that ran while this save waited
        # has already forgotten the channel, and must not be undone.
        if channel not in getattr(orch, "_code_channels", ()):
            return
        records = _load(target)
        records[channel] = rec
        _save(target, records)


def forget_channel(channel: str, path: Path | None = None) -> None:
    """Drop *channel*'s record (the channel was archived)."""
    if not channel:
        return
    target = path or _store_path()
    with _WRITE_LOCK:
        records = _load(target)
        if records.pop(channel, None) is not None:
            _save(target, records)


def clear_channel_state(orch: Any, channel: str) -> None:
    """Drop *channel* from the in-memory code-channel maps (it was archived).

    The origin map is left alone: the archive path consumes it to post the
    closing summary back to the origin.
    """
    for set_name in ("_code_channels", "_owned_code_channels"):
        members: set[str] = getattr(orch, set_name, set())
        members.discard(channel)
    for name in ("_code_channel_repo_by_id", "_code_channel_diff_base", "_code_channel_session_ts"):
        getattr(orch, name, {}).pop(channel, None)


def strip_code_channel_context(text: str) -> str:
    """*text* without a leading code-channel working-context block.

    The block is instructions for the agent, not something a person typed, so an
    echo of the turn (the dashboard-to-Slack mirror) shows only what follows it.
    Text that does not start with the block is returned unchanged.
    """
    if not text or not text.startswith(CODE_CHANNEL_CONTEXT_HEADER):
        return text
    end = text.find("\n\n")
    return text[end + 2 :] if end != -1 else ""

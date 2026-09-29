#!/usr/bin/env python3
"""Write the pipeline board card's real bytes for the website's render test.

The Python gates read the card page as text. Whether the HOST puts the written values on
the page is a question about ``dashboardDocument.ts``, so that half is tested in the
website suite -- and it must test the bytes the gateway actually publishes rather than a
sample someone typed into a TypeScript file.

This runs the real provider (``build_pipeline_board``) and the real flattener
(``panel_card_data``) over two boards and writes the page plus both data sets to
``website/src/test/fixtures/pipelineBoardCard.json``. The website test re-asserts parity
against the page's own bindings, so a fixture that has gone stale fails there instead of
passing quietly.

Regenerate with::

    PYTHONPATH=src python scripts/pipeline_board_card_fixture.py

``--check`` rewrites nothing and exits non-zero when the file on disk differs, which is how
CI can tell a stale fixture from a current one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kiro_crew.pipeline_board_contract import (  # noqa: E402
    build_pipeline_board,
    card_template_path,
    panel_card_data,
    validate_judgment,
)

OUT = ROOT / "website" / "src" / "test" / "fixtures" / "pipelineBoardCard.json"


def _item(
    ident: str,
    *,
    state: str,
    pr: int | None = None,
    worker: str | None = None,
    round_: int = 1,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "item_id": ident,
        "title": "an item",
        "acceptance": {},
        "state": state,
        "verdict": None,
        "decision": "",
        "worker_session_key": worker,
        "round": round_,
        "fails": 0,
        "status": None,
        "summary": "",
        "artifacts": {},
        "pr": pr,
        "last_report_at": "2026-09-29T15:00:00+00:00",
        "created_at": "2026-09-29T14:00:00+00:00",
        "closed_at": None,
        "events": [],
    }


def _full() -> dict[str, str]:
    """A board with something in every column, an action, a tally and a metric gloss."""
    view = {
        "conductor": {
            "schema": 1,
            "slot_key": "chat-1875",
            "goal": "the board becomes the first dashboard template",
            "round": 3,
            "goal_version": 1,
            "depth": 0,
            "parent_item": None,
            "created_at": "2026-09-29T13:00:00+00:00",
            "entries": 41,
            "first_entry_at": "2026-09-29T13:00:00+00:00",
            "last_entry_at": "2026-09-29T15:30:00+00:00",
            "generation": "gen-1",
        },
        "items": [
            _item("it_0", state="open", pr=14583, worker="chat-1875", round_=3),
            _item("it_1", state="open", pr=14966, worker="chat-2176", round_=3),
            _item("it_2", state="accepted", pr=14689),
            _item("it_3", state="accepted", pr=14111),
            _item("it_4", state="rejected", worker="chat-2301"),
            _item("it_5", state="abandoned", pr=12046),
        ],
        "omitted": 2,
    }
    judgment = validate_judgment(
        {
            "lede": "Six items this round; one needs a person before CI can go green.",
            "you": {
                "it_0": "push the rebase once the base fix lands on main",
                "it_4": "decide whether the acceptance bar itself was wrong",
            },
            "notes": {
                "items": "items this board has created over its life",
                "entries": "work entries the crew log holds for it",
            },
            "checks": {"it_0": "41/47", "it_2": "161/161"},
        }
    )
    panel = build_pipeline_board(
        view,  # type: ignore[arg-type]
        judgment,
        # NOT the conductor crew's real display name, whose first word is deliberately
        # joined. This fixture is written to JSON, which carries no comment syntax and so
        # cannot carry the ``brand-ok`` marker the brand gate needs -- the joined spelling
        # would be an unexplainable product-name misspelling in a checked-in file. The
        # slugification that makes the joined form load-bearing belongs to the drawer's
        # template selection and is asserted against the real constant in
        # ``test_pipeline_board_contract_parity``; what this fixture exercises is
        # RENDERING, for which the name is just text of a realistic length.
        name="Pipeline Conductor",
        captured_at="2026-09-29 15:37 UTC",
        stale_after_seconds=900,
        now_epoch=1790696420.0,
    )
    return panel_card_data(panel)


def _hostile() -> dict[str, str]:
    """A payload from before this contract existed, plus values that cannot be read.

    Reachable rather than theoretical: a record stored in the free shape reaches the
    flattener untouched, and the operator override directory can serve an older page
    against a newer provider.
    """

    class Unreadable:
        def __str__(self) -> str:
            raise RuntimeError("this value cannot be read as text")

        __repr__ = __str__

    payload = {
        "contract_version": {"nested": "object"},
        "lede": Unreadable(),
        "since": None,
        "meta": {
            "name": ["a", "list"],
            "captured_at": "2026-09-29 15:37 UTC",
            "age_seconds": "half an hour",
            "stale_after_seconds": 900,
            "revision": Unreadable(),
        },
        "columns": [
            {"name": "open", "cards": [{"id": Unreadable(), "sub": {}, "of": 7, "you": 3}]}
        ],
        "progress": {"total": "six", "added_since": True, "segments": {"open": 1}},
        "stats": [{"k": "items", "v": Unreadable(), "note": []}, "junk"],
        "omitted": -4,
    }
    return panel_card_data(payload)  # type: ignore[arg-type]


def _excerpt(value: Any, limit: int = 160) -> str:
    """*value* as a bounded repr, so a 7 KB page does not become the whole message."""
    text = repr(value)
    return text if len(text) <= limit else f"{text[:limit]}... ({len(text)} chars)"


def _first_difference(have: Any, want: Any, path: str = "") -> tuple[str, str, str] | None:
    """The first place *have* and *want* disagree, as (where, have, want), else ``None``.

    Depth-first over mappings in the GENERATED object's key order, so the location named is
    stable between runs rather than depending on dict iteration of whichever side is bigger.
    """
    if isinstance(want, dict) and isinstance(have, dict):
        for key in want:
            if key not in have:
                return (f"{path}.{key}".lstrip("."), "<absent>", _excerpt(want[key]))
            found = _first_difference(have[key], want[key], f"{path}.{key}".lstrip("."))
            if found is not None:
                return found
        for key in have:
            if key not in want:
                return (f"{path}.{key}".lstrip("."), _excerpt(have[key]), "<absent>")
        return None
    if have != want:
        return (path or "<root>", _excerpt(have), _excerpt(want))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit non-zero if stale")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=OUT,
        help=(
            "the fixture file to read or write; defaults to the checked-in one. Exists so the "
            "test that proves --check CAN fail corrupts a COPY under tmp_path instead of the "
            "repository's own file: the two --check tests carry no xdist_group, so under -n auto "
            "they can run on separate workers while one holds the real fixture corrupted for the "
            "whole subprocess window, and a worker kill skips the restore entirely."
        ),
    )
    args = parser.parse_args()
    out: Path = args.fixture

    document = {
        # No newline normalisation here, deliberately. ``read_text`` opens with
        # ``newline=None``, so Python's universal-newline translation has already turned any
        # ``\r\n`` into ``\n`` before this sees it -- a ``.replace`` would be a mechanism with
        # no effect. What makes the comparison newline-proof is that ``--check`` compares
        # PARSED JSON rather than serialized text; what keeps the SHIPPED page byte-stable is
        # the ``eol=lf`` pin in ``.gitattributes``, which also keeps its 8192-byte budget from
        # moving by a byte per line.
        "html": card_template_path().read_text(encoding="utf-8"),
        "boards": {"full": _full(), "hostile": _hostile()},
    }
    text = json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.check:
        # COMPARED AS PARSED JSON, not as text.
        #
        # The question this check asks is whether the fixture holds what the generator produces,
        # which is about CONTENT. Comparing the serialized text instead drags every newline and
        # encoding question into a check that does not care about them: the file is written and
        # read back through text mode, whose newline translation is platform-dependent, so a
        # byte comparison can answer "stale" on a fixture that is current. That reading cost two
        # CI rounds on Windows, and the message said only "stale", which named nothing.
        #
        # So this compares objects, and when they differ it says WHICH field and shows both
        # values bounded -- a check whose failure does not locate itself is a check someone has
        # to re-derive by hand.
        if not out.exists():
            print(f"missing fixture: {out} -- generate with {Path(__file__).name}")
            return 1
        try:
            current = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"unreadable fixture: {out} ({exc})")
            return 1
        drift = _first_difference(current, document)
        if drift is not None:
            where, have, want = drift
            print(f"stale fixture: {out}")
            print(f"  differs at: {where}")
            print(f"  on disk:    {have}")
            print(f"  generated:  {want}")
            print(f"  regenerate with {Path(__file__).name}")
            return 1
        print(f"fixture current: {out}")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text)} bytes)")
    for name, data in document["boards"].items():  # type: ignore[union-attr]
        blank = sorted(k for k, v in data.items() if not v.strip())
        print(f"  {name}: {len(data)} fields, blank={blank}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Add Phase EXT (extended free open-data ingest) tickets to the existing
V12-V14 beads database WITHOUT recreating the 124 tickets already in place.

Reads the current `.beads/v12_v14_keymap.json` to learn which keys already
have beads IDs, imports the canonical ticket list from
`scripts/setup_v12_v14_beads.py`, and creates ONLY the new tickets (the ones
whose key is not yet in the keymap). Then re-wires the two existing tickets
whose dependencies changed:

    T2_4 (Build abt_v12_external.parquet)  +9 EXT_* deps (priority-1 ingest)
    T4_3 (V12 LAD search)                  +1 EXT_SURVIVOR_MERGE dep

Idempotent: safe to re-run; any ticket already in the keymap is skipped, and
adding a duplicate dependency edge is a no-op in beads.

Usage:
    python -m scripts.add_ext_open_data_tickets --dry-run   # preview
    python -m scripts.add_ext_open_data_tickets --apply     # create + rewire
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.setup_v12_v14_beads import ALL, Ticket, _bd_create

KEYMAP_PATH = Path(".beads/v12_v14_keymap.json")

# --- Dependency surgery: new edges to add to existing-keys via `bd dep add` ---
# These are the deps we added inline in the source-of-truth Ticket definitions
# (T2_4, T4_3) but which the existing beads DB doesn't yet have because those
# tickets were created before EXT_TASKS existed.
NEW_EDGES_TO_EXISTING: dict[str, list[str]] = {
    # T2_4 → priority-1 EXT_* ingest tickets
    "T2_4": [
        "EXT_UKRSTAT_RTI", "EXT_UKRSTAT_BIRTHS", "EXT_UKRSTAT_INDPROD",
        "EXT_NBU_CCI", "EXT_AIRRAID_OBLAST", "EXT_BLACKOUT_DTEK",
        "EXT_IOM_IDP", "EXT_WIKI_PV", "EXT_ORTHODOX_CAL",
    ],
    # T4_3 → wait for survivor-merge so V12.5 LAD sees the post-A/B ABT
    "T4_3": ["EXT_SURVIVOR_MERGE"],
}


def _load_keymap() -> dict[str, str]:
    if not KEYMAP_PATH.exists():
        sys.exit(f"keymap not found at {KEYMAP_PATH}; run setup_v12_v14_beads first.")
    return json.loads(KEYMAP_PATH.read_text())


def _save_keymap(keymap: dict[str, str]) -> None:
    KEYMAP_PATH.write_text(json.dumps(keymap, indent=2))


def _bd_dep_add(blocker_id: str, blocked_id: str) -> None:
    """Add a 'blocker_id blocks blocked_id' edge — i.e. blocked_id depends on
    blocker_id. Idempotent in beads (re-adding the same edge is a no-op)."""

    # `bd dep add <issue> <depends-on>` means: <issue> depends on <depends-on>
    # (so <depends-on> blocks <issue>). We want blocked_id to depend on blocker_id,
    # so blocked_id is the first arg.
    cmd = ["bd", "dep", "add", blocked_id, blocker_id]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        stderr = (out.stderr or "").lower()
        if "already exists" in stderr or "duplicate" in stderr:
            return
        sys.exit(f"bd dep add failed: {out.stderr}\nstdout: {out.stdout}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Actually create tickets and edges (default is dry-run preview)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview only (default if --apply not passed)")
    args = ap.parse_args()

    keymap = _load_keymap()
    print(f"Loaded keymap with {len(keymap)} existing key→id mappings")

    new_tickets: list[Ticket] = [t for t in ALL if t.key not in keymap]
    print(f"Found {len(new_tickets)} new tickets to create:")
    for t in new_tickets:
        print(f"  + {t.key:<28} ({t.type:<8}) {t.title[:80]}")

    new_edges_total = sum(len(v) for v in NEW_EDGES_TO_EXISTING.values())
    print(f"\nNew dependency edges to add to existing tickets: {new_edges_total}")
    for to, deps in NEW_EDGES_TO_EXISTING.items():
        print(f"  {to} → blocked by: {', '.join(deps)}")

    if not args.apply:
        print("\n(Dry-run; pass --apply to actually create tickets and edges.)")
        return 0

    print("\n=== Creating new tickets in beads (deps inline) ===", flush=True)
    for i, t in enumerate(new_tickets, 1):
        if t.parent and t.parent not in keymap:
            sys.exit(f"Parent of {t.key} ({t.parent}) not in keymap; "
                     f"create parent first.")
        for d in t.deps:
            if d not in keymap:
                sys.exit(
                    f"Dep order error: {t.key} depends on {d}, which is not "
                    f"yet in the keymap. The new_tickets list is constructed "
                    f"from ALL preserving order; reorder ALL in setup_v12_v14_beads."
                )
        bd_id = _bd_create(t, keymap)
        keymap[t.key] = bd_id
        _save_keymap(keymap)
        print(f"  [{i:>3}/{len(new_tickets)}] + {t.key:<28} → {bd_id}", flush=True)

    print("\n=== Adding new dep edges to pre-existing tickets ===", flush=True)
    edge_count = 0
    for blocked_key, blocker_keys in NEW_EDGES_TO_EXISTING.items():
        if blocked_key not in keymap:
            sys.exit(f"Existing ticket {blocked_key} not in keymap")
        blocked_id = keymap[blocked_key]
        for blocker_key in blocker_keys:
            if blocker_key not in keymap:
                sys.exit(f"New ticket {blocker_key} missing from keymap "
                         f"after creation; aborting edge wiring.")
            blocker_id = keymap[blocker_key]
            _bd_dep_add(blocker_id, blocked_id)
            edge_count += 1
            print(f"  + {blocker_key} ({blocker_id}) blocks "
                  f"{blocked_key} ({blocked_id})", flush=True)

    print(f"\n✅ Done. {len(new_tickets)} tickets created, "
          f"{edge_count} new dependency edges wired.")
    print("   Run `bd ready` to see immediately-actionable tickets.")
    print(f"   Updated keymap saved to {KEYMAP_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

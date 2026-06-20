"""02 (clean) — data-driven directory filter for the URL-discovery cache.

A real company website resolves to ONE company. A domain that resolves to many
distinct companies is, with near-certainty, a directory/aggregator (Companies
House mirrors, LEI registries, b2b data sites) or a big company repeatedly
mis-matched to small firms by name coincidence. Either way it is not the small
firm's own site.

This script:
  1. Mines data/cache/urls.json and counts distinct companies per matched domain.
  2. Auto-blocks every domain with count >= THRESHOLD; prints the full list.
  3. Persists the auto-blocklist to data/cache/auto_blocklist.json (the discovery
     runner unions it into its directory filter).
  4. Purges existing matches pointing to auto-blocked domains: the contaminated
     match is removed so the firm is re-queried fresh under the new filter. We
     only ever stored the single best match per firm (not all raw results), so we
     cannot re-pick an alternative offline — re-querying is the honest fix.

Run BEFORE resuming the sweep.
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache" / "urls.json"
AUTO_BLOCK_FILE = ROOT / "data" / "cache" / "auto_blocklist.json"
THRESHOLD = 4  # >= this many distinct companies => directory / over-matched


def main() -> None:
    cache = json.loads(CACHE.read_text())

    # 1. Count distinct companies per matched domain.
    dom_counts: Counter = Counter()
    for v in cache.values():
        m = v.get("match")
        if m and m.get("domain"):
            dom_counts[m["domain"]] += 1

    # 2. Auto-block domains at/over threshold.
    auto_blocked = {d: c for d, c in dom_counts.items() if c >= THRESHOLD}
    print(f"=== auto-blocked domains (>= {THRESHOLD} distinct companies): {len(auto_blocked)} ===")
    for dom, c in sorted(auto_blocked.items(), key=lambda x: -x[1]):
        print(f"  {c:4}  {dom}")

    # 3. Persist (union with any prior auto-blocklist).
    prior = set()
    if AUTO_BLOCK_FILE.exists():
        try:
            prior = set(json.loads(AUTO_BLOCK_FILE.read_text()))
        except json.JSONDecodeError:
            prior = set()
    full = sorted(prior | set(auto_blocked))
    AUTO_BLOCK_FILE.write_text(json.dumps(full, indent=2))
    print(f"\nauto_blocklist.json now holds {len(full)} domains")

    # 4. Purge (a) matches on auto-blocked directory domains and (b) existing
    #    zero-signal tentatives (score <= 0) — pure noise. Dropped entries revert
    #    to "unattempted" and are re-queried fresh under the gated capture logic.
    blocked_set = set(auto_blocked)
    purged_dir = 0
    purged_noise = 0
    keep = {}
    for cnum, v in cache.items():
        m = v.get("match")
        if m and m.get("domain") in blocked_set:
            purged_dir += 1
            continue
        # Zero-signal tentative: unverified match with no name overlap.
        if m and not m.get("verified") and (m.get("score") or 0) <= 0:
            purged_noise += 1
            continue
        keep[cnum] = v
    purged = purged_dir + purged_noise

    tmp = CACHE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(keep, indent=2))
    tmp.replace(CACHE)

    # Report the cleaned breakdown.
    resolved = sum(1 for v in keep.values() if v.get("match"))
    verified = sum(1 for v in keep.values() if v.get("match") and v["match"].get("verified"))
    unverified = sum(1 for v in keep.values() if v.get("match") and not v["match"].get("verified"))
    nomatch = sum(1 for v in keep.values() if not v.get("match") and not v.get("error"))
    print(f"\n=== cache after purge ===")
    print(f"  purged (directory domains)  : {purged_dir}")
    print(f"  purged (zero-signal noise)  : {purged_noise}")
    print(f"  remaining cache entries     : {len(keep)}")
    print(f"  resolved (has URL)          : {resolved}")
    print(f"    - verified (scorer)       : {verified}")
    print(f"    - unverified (tentative)  : {unverified}")
    print(f"  no-match                    : {nomatch}")
    print(f"  (purged firms drop to unattempted and will be re-queried)")


if __name__ == "__main__":
    main()

"""06 — Cluster-then-classify pipeline (Option A).

Cuts LLM spend by ~96% on the scraped segment: instead of classifying every
scraped firm, we embed them locally, cluster with HDBSCAN, classify only ~3
representatives per cluster with Gemini Flash, then propagate the cluster's
majority label to all members.

Segments (42,322 working firms):
  scraped (~27,348)  -> embed -> HDBSCAN -> classify 3 reps/cluster -> propagate
  noise (label -1)   -> needs_review=True, NO LLM call
  no-text (~14,974)  -> rules-only output,        NO LLM call

Representatives reuse src/05_classify_llm.py's validated prompt + caching, so a
rep already classified in a gold run is free on re-run.

Outputs output/classified_firms.parquet. Run:
  GEMINI_API_KEY=... python3 src/06_cluster_classify.py
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse the validated classifier (prompt, caching, parse/validate, reconcile).
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "classify_llm", ROOT / "src" / "05_classify_llm.py")
clf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clf)

from schema.taxonomy import INSUFFICIENT_EVIDENCE  # noqa: E402

WORKING = ROOT / "output" / "working.parquet"
RULES = ROOT / "output" / "rules_labels.parquet"
SCRAPES = ROOT / "data" / "cache" / "scrapes"
EMB_CACHE = ROOT / "data" / "cache" / "embeddings_minilm.npz"
OUT = ROOT / "output" / "classified_firms.parquet"

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_TEXT_CAP = 2000          # chars fed to the embedder (MiniLM truncates anyway)
TARGET_CLUSTERS = 400
PCA_DIMS = 50                  # speed + denoise HDBSCAN; cosine-style via L2-norm
PROPAGATION_DISCOUNT = 0.85
REPS_PER_CLUSTER = 3

# Pricing assumption for cost estimate (Flash tier). Confirm against billing.
PRICE_IN_PER_M = 0.30
PRICE_OUT_PER_M = 2.50
EST_IN_TOK = 1402
EST_OUT_TOK = 150
COST_PER_CALL = EST_IN_TOK / 1e6 * PRICE_IN_PER_M + EST_OUT_TOK / 1e6 * PRICE_OUT_PER_M


def _slug(cn: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(cn))


# --------------------------------------------------------------------------
# 1. Load + segment
# --------------------------------------------------------------------------
def load_segments():
    df = pd.read_parquet(WORKING)[
        ["CompanyNumber", "CompanyName", "SICCode.SicText_1"]].copy()
    df["CompanyNumber"] = df["CompanyNumber"].astype(str)
    paths = {cn: SCRAPES / f"{_slug(cn)}.txt" for cn in df["CompanyNumber"]}
    df["has_text"] = df["CompanyNumber"].map(lambda cn: paths[cn].exists())
    scraped = df[df["has_text"]].reset_index(drop=True)
    notext = df[~df["has_text"]].reset_index(drop=True)
    print(f"scraped firms : {len(scraped)}")
    print(f"no-text firms : {len(notext)}")
    return scraped, notext, paths


# --------------------------------------------------------------------------
# 2. Embed (cached)
# --------------------------------------------------------------------------
def embed(scraped: pd.DataFrame, paths: dict) -> np.ndarray:
    cns = scraped["CompanyNumber"].tolist()
    if EMB_CACHE.exists():
        z = np.load(EMB_CACHE, allow_pickle=True)
        if list(z["cns"]) == cns:
            print(f"embeddings: loaded cache ({z['emb'].shape})")
            return z["emb"]
    from sentence_transformers import SentenceTransformer
    print(f"embeddings: encoding {len(cns)} firms with {EMBED_MODEL} ...")
    texts = [paths[cn].read_text(encoding="utf-8", errors="replace")[:EMBED_TEXT_CAP]
             for cn in cns]
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(texts, batch_size=256, show_progress_bar=True,
                       normalize_embeddings=True).astype(np.float32)
    EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(EMB_CACHE, cns=np.array(cns), emb=emb)
    print(f"embeddings: encoded {emb.shape}, cached")
    return emb


# --------------------------------------------------------------------------
# 3. Cluster (HDBSCAN, tune min_cluster_size toward TARGET_CLUSTERS)
# --------------------------------------------------------------------------
def cluster(emb: np.ndarray):
    """KMeans(n_clusters=400) on the L2-normalized embeddings. KMeans guarantees
    full coverage (no noise) — chosen after HDBSCAN degenerated to 92% noise on
    this embedding space. Normalized vectors make euclidean KMeans ~ spherical
    (cosine) k-means."""
    from sklearn.cluster import MiniBatchKMeans

    X = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    km = MiniBatchKMeans(n_clusters=TARGET_CLUSTERS, random_state=0,
                         batch_size=2048, n_init=3, max_iter=200)
    labels = km.fit_predict(X)
    n_clusters = len(set(labels))
    n_noise = 0
    sizes = np.bincount(labels)
    print(f"KMeans: {n_clusters} clusters, mean {sizes.mean():.1f} firms/cluster "
          f"(min {sizes.min()}, max {sizes.max()})")
    return labels, X, n_clusters, n_noise


# --------------------------------------------------------------------------
# 4. Representatives + classify
# --------------------------------------------------------------------------
def representatives(labels: np.ndarray, X: np.ndarray) -> dict:
    """cluster_id -> list of row indices (<=3) nearest to the cluster mean."""
    reps = {}
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        idx = np.where(labels == cid)[0]
        centroid = X[idx].mean(axis=0)
        d = np.linalg.norm(X[idx] - centroid, axis=1)
        reps[cid] = idx[np.argsort(d)[:REPS_PER_CLUSTER]].tolist()
    return reps


def classify_reps(reps: dict, scraped: pd.DataFrame, api_key: str):
    """Classify each cluster's reps. Returns (rep_rows_by_cluster, calls_made)."""
    rep_rows = collections.defaultdict(list)
    calls_made = 0
    cache_dir = clf.CACHE
    total = sum(len(v) for v in reps.values())
    done = 0
    for cid, idxs in reps.items():
        for i in idxs:
            r = scraped.iloc[i]
            cn = r["CompanyNumber"]
            cache_file = cache_dir / f"{_slug(cn)}__{clf.PROMPT_VERSION}.json"
            if not cache_file.exists():
                calls_made += 1
            res = clf.classify_firm(api_key, cn, r["CompanyName"],
                                    r["SICCode.SicText_1"], "scraped")
            row = clf.parse_and_validate(res)
            rep_rows[cid].append(row)
            done += 1
            if done % 100 == 0:
                print(f"  ...{done}/{total} reps classified ({calls_made} new calls)")
    return rep_rows, calls_made


def majority_label(rows: list[dict]) -> dict | None:
    """Pick the cluster label: modal primary_niche among valid reps,
    tie-broken by highest confidence; return that rep's full label record."""
    ok = [r for r in rows if r.get("schema_ok")]
    if not ok:
        return None
    niches = [r["primary_niche"] for r in ok]
    counts = collections.Counter(niches)
    top = max(counts.values())
    candidates = [r for r in ok if counts[r["primary_niche"]] == top]
    chosen = max(candidates, key=lambda r: r["confidence"])
    mean_conf = sum(r["confidence"] for r in ok) / len(ok)
    return {
        "stack_layer": chosen["stack_layer"],
        "function": chosen["function"],
        "business_model": chosen["business_model"],
        "vertical": chosen["vertical"],
        "primary_niche": chosen["primary_niche"],
        "confidence": round(mean_conf * PROPAGATION_DISCOUNT, 3),
    }


# --------------------------------------------------------------------------
# 5/6. Assemble output rows
# --------------------------------------------------------------------------
def build_rows(scraped, notext, labels, reps, rep_rows, rules_df):
    out = []
    # cluster labels
    cluster_label = {cid: majority_label(rows) for cid, rows in rep_rows.items()}
    rep_idx = {i for idxs in reps.values() for i in idxs}

    for i, r in scraped.iterrows():
        cid = int(labels[i])
        base = {"CompanyNumber": r["CompanyNumber"],
                "CompanyName": r["CompanyName"], "cluster_id": cid}
        if cid == -1:
            out.append({**base, "stack_layer": None, "function": "",
                        "business_model": None, "vertical": "",
                        "primary_niche": INSUFFICIENT_EVIDENCE,
                        "confidence": 0.0, "needs_review": True,
                        "source": "noise"})
            continue
        lab = cluster_label.get(cid)
        if lab is None:
            out.append({**base, "stack_layer": None, "function": "",
                        "business_model": None, "vertical": "",
                        "primary_niche": INSUFFICIENT_EVIDENCE,
                        "confidence": 0.0, "needs_review": True,
                        "source": "cluster_unlabeled"})
            continue
        out.append({**base,
                    "stack_layer": lab["stack_layer"],
                    "function": "|".join(lab["function"]),
                    "business_model": lab["business_model"],
                    "vertical": "|".join(lab["vertical"]),
                    "primary_niche": lab["primary_niche"],
                    "confidence": lab["confidence"],
                    "needs_review": False,
                    "source": "cluster_rep" if i in rep_idx else "cluster_member"})

    # no-text -> rules-only
    rd = rules_df.set_index("CompanyNumber")
    for _, r in notext.iterrows():
        cn = r["CompanyNumber"]
        rr = rd.loc[cn].to_dict() if cn in rd.index else {}
        func = rr.get("function")
        out.append({
            "CompanyNumber": cn, "CompanyName": r["CompanyName"], "cluster_id": -2,
            "stack_layer": rr.get("stack_layer"),
            "function": func if isinstance(func, str) else "",
            "business_model": rr.get("business_model") or None,
            "vertical": "",
            "primary_niche": (func if isinstance(func, str) and func
                              else INSUFFICIENT_EVIDENCE),
            "confidence": float(rr.get("stack_layer_conf") or 0.0),
            "needs_review": True,
            "source": "rules_only",
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=int, default=None,
                    help="emit labels for an N-firm sample of scraped firms only "
                         "(classifies reps only for the clusters those firms touch)")
    args = ap.parse_args()

    api_key = clf._load_api_key()
    scraped, notext, paths = load_segments()
    emb = embed(scraped, paths)
    labels, X, n_clusters, n_noise = cluster(emb)
    reps = representatives(labels, X)

    out_path = OUT
    if args.checkpoint:
        # Sample N scraped firms; restrict work to the clusters they fall into.
        sample = scraped.sample(n=min(args.checkpoint, len(scraped)),
                                random_state=0).sort_index()
        keep_clusters = set(int(labels[i]) for i in sample.index)
        reps = {c: r for c, r in reps.items() if c in keep_clusters}
        print(f"checkpoint: {len(sample)} firms across {len(keep_clusters)} clusters")
        rep_rows, calls_made = classify_reps(reps, scraped, api_key)
        # build full rows then keep only the sampled scraped firms
        full = build_rows(scraped, notext, labels, reps, rep_rows,
                          pd.read_parquet(RULES))
        sample_cns = set(sample["CompanyNumber"])
        df = full[full["CompanyNumber"].isin(sample_cns)].reset_index(drop=True)
        out_path = ROOT / "output" / f"classified_checkpoint_{args.checkpoint}.parquet"
    else:
        rep_rows, calls_made = classify_reps(reps, scraped, api_key)
        df = build_rows(scraped, notext, labels, reps, rep_rows,
                        pd.read_parquet(RULES))
    df.to_parquet(out_path, index=False)

    reps_total = sum(len(v) for v in reps.values())
    cached = reps_total - calls_made
    cost = calls_made * COST_PER_CALL
    print("\n" + "=" * 60)
    print(f"clusters                 : {n_clusters}")
    print(f"mean firms / cluster     : {len(scraped)/n_clusters:.1f}")
    print(f"noise points             : {n_noise}")
    print(f"reps total               : {reps_total}")
    print(f"LLM calls: NEW           : {calls_made}")
    print(f"LLM calls: CACHED (free) : {cached}")
    print(f"est. cost (@${COST_PER_CALL:.5f}/call) : ${cost:.2f}")
    print(f"output rows              : {len(df)}  -> {out_path.relative_to(ROOT)}")
    print("\nsource breakdown:")
    for k, v in df["source"].value_counts().items():
        print(f"  {k:<16} {v}")
    print("\ntop 15 primary_niche labels by firm count:")
    for niche, cnt in df["primary_niche"].value_counts().head(15).items():
        print(f"  {niche:<32} {cnt}")


if __name__ == "__main__":
    main()

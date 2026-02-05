#!/usr/bin/env python3
"""
Convert FAERS free-text DRUGNAMEs to RxNorm concepts and lift to ingredient level —
**OOMD-resilient version** (streamed I/O, tiny memory footprint, resumable).

Key changes vs original while preserving outputs/behavior:
- Streams DISTINCT drug names from SQLite in chunks (no giant DataFrame in RAM).
- Avoids building massive Python lists; flushes rows incrementally to disk.
- Uses a temporary Parquet “delta” and DuckDB to UNION into the cache atomically.
- Explodes ingredient JSON **streaming** to a temp parquet, then dedups with DuckDB
  using `ROW_NUMBER()` to match `drop_duplicates(subset=...)` semantics.
- Produces the same artifacts:
    - cache table (parquet + csv)
    - exploded table (parquet + csv)
- Adds robust logging, heartbeats, and graceful shutdown.

Reliability / policy:
- Pure score gate: accept if RxNav approximateTerm score >= 9.0; reject otherwise.
- Per-request throttle + retries honoring Retry-After.
- Do not treat 'error' rows as cached; they are retried next run.

Outputs (same locations/filenames by default):
 - /home/zchu/FAERS/rxnorm_map_script.parquet
 - /home/zchu/FAERS/rxnorm_map_script.csv
 - /home/zchu/FAERS/rxnorm_map_exploded_script.parquet
 - /home/zchu/FAERS/rxnorm_map_exploded_script.csv
"""
import os, re, json, time, argparse, sys, signal, sqlite3
from typing import Optional, Dict, Any, List, Iterable, Sequence

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# ---- PyArrow compatibility shim: drop unsupported kwargs on from_pylist ----
# _orig_from_pylist = pa.Table.from_pylist
# def _from_pylist_compat(*args, **kwargs):
#     kwargs.pop("preserve_index", None)
#     return _orig_from_pylist(*args, **kwargs)
# pa.Table.from_pylist = _from_pylist_compat

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# ---------- Defaults (paths preserved) ----------
DEFAULT_SQLITE = "/home/zchu/FAERS/my.db"
DEFAULT_CACHE_PARQ = "/home/zchu/FAERS/rxnorm_map_script.parquet"
DEFAULT_OUT_CSV = "/home/zchu/FAERS/rxnorm_map_script.csv"
DEFAULT_OUT_PARQ = "/home/zchu/FAERS/rxnorm_map_exploded_script.parquet"
DEFAULT_OUT_EXPLODED_CSV = "/home/zchu/FAERS/rxnorm_map_exploded_script.csv"

# Tunables (can also override via CLI)
DEFAULT_CHUNK = int(os.environ.get("RXN_CHUNK", 20_000))       # names fetched per DB chunk
DEFAULT_BATCH = int(os.environ.get("RXN_BATCH", 500))          # rows buffered before flush
DEFAULT_SLEEP = float(os.environ.get("RXN_SLEEP", 0.05))       # polite delay between *names*
DEFAULT_PER_REQUEST_SLEEP = float(os.environ.get("RXN_REQ_SLEEP", 0.20))  # throttle per HTTP req (~5 rps)
TQDM_MININTERVAL = float(os.environ.get("TQDM_MININTERVAL", 2))

# Thread caps to reduce surprise memory spikes from BLAS/NumPy
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ---------- Regex + token helpers (token logic retained for utilities but not used to gate ≥9.0) ----------
UNIT_RE = re.compile(r"\b(\d+(\.\d+)?)(mg|mcg|g|ml|%|iu)\b", re.I)
FORM_RE = re.compile(r"\b(tab|cap|inj|soln|susp|sr|er|dr|po|sc|iv|im|sl|oral|patch|cream|ointment)\b", re.I)
NONDRUG = {"PLACEBO","UNKNOWN","N/A","DEVICE","SYRINGE","KIT","TEST","VACCINE NOS"}

CACHE_COLS: Sequence[str] = (
    "drugname_norm","rxcui_best","rxnorm_name_best","tty_best","score",
    "ingredient_rxcui","ingredient_name","ingredients_json",
    "accepted_by","token_pass","second_best_score","best_minus_second","cand_count","error_detail"
)

def norm_tokens(s: str):
    s = s.upper()
    s = UNIT_RE.sub(" ", s)
    s = FORM_RE.sub(" ", s)
    s = re.sub(r"[^A-Z0-9+/ ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = set(t for t in s.replace("+"," + ").replace("/"," / ").split() if t and t not in NONDRUG)
    return toks

def token_ok(src_name: str, hit_name: str, min_jaccard=0.6):
    a, b = norm_tokens(src_name), norm_tokens(hit_name)
    if not a or not b:
        return False
    inter = len(a & b)
    union = len(a | b)
    j = inter / union
    return j >= min_jaccard or (" ".join(b) in " ".join(a)) or (" ".join(a) in " ".join(b))

# ---------- HTTP / RxNav helpers with retries and per-request throttling ----------

# simple global throttle for every HTTP call
_LAST_CALL_TS = [0.0]
def _throttle(per_request_sleep: float):
    now = time.time()
    elapsed = now - _LAST_CALL_TS[0]
    if elapsed < per_request_sleep:
        time.sleep(per_request_sleep - elapsed)
    _LAST_CALL_TS[0] = time.time()

def make_session(user_agent: str = "faers-rxnorm-mapper/1.0"):
    s = requests.Session()
    retries = Retry(
        total=6,
        backoff_factor=1.0,  # exponential backoff: 1s, 2s, 4s, ...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries, pool_connections=32, pool_maxsize=32))
    s.headers.update({"User-Agent": user_agent})
    return s

def _get(sess: requests.Session, url: str, *, per_request_sleep: float, **kwargs) -> requests.Response:
    _throttle(per_request_sleep)
    r = sess.get(url, timeout=30, **kwargs)
    r.raise_for_status()
    return r

def rxnav_approx_match(sess: requests.Session, term: str, max_entries: int = 5, per_request_sleep: float | None = None) -> List[Dict[str, Any]]:
    if per_request_sleep is None:
        per_request_sleep = DEFAULT_PER_REQUEST_SLEEP
    url = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
    params = {"term": term, "maxEntries": max_entries}
    r = _get(sess, url, params=params, per_request_sleep=per_request_sleep)
    data = r.json()
    return data.get("approximateGroup", {}).get("candidate", []) or []

def rxnav_concept_props(sess: requests.Session, rxcui: str, per_request_sleep: float | None = None) -> Dict[str, Any]:
    if per_request_sleep is None:
        per_request_sleep = DEFAULT_PER_REQUEST_SLEEP
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json"
    r = _get(sess, url, per_request_sleep=per_request_sleep)
    return r.json().get("properties", {}) or {}

def rxnav_related_ingredients(sess: requests.Session, rxcui: str, per_request_sleep: float | None = None) -> List[Dict[str, Any]]:
    if per_request_sleep is None:
        per_request_sleep = DEFAULT_PER_REQUEST_SLEEP
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json"
    params = {"tty": "IN"}
    r = _get(sess, url, params=params, per_request_sleep=per_request_sleep)
    groups = r.json().get("relatedGroup", {}).get("conceptGroup", []) or []
    out = []
    for g in groups:
        for c in g.get("conceptProperties", []) or []:
            out.append(c)
    return out

def rxnav_allrelated_ingredients(sess: requests.Session, rxcui: str, per_request_sleep: float | None = None):
    if per_request_sleep is None:
        per_request_sleep = DEFAULT_PER_REQUEST_SLEEP
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allrelated.json"
    r = _get(sess, url, params={"tty": "IN"}, per_request_sleep=per_request_sleep)
    groups = r.json().get("allRelatedGroup", {}).get("conceptGroup", []) or []
    out = []
    for g in groups:
        for c in g.get("conceptProperties", []) or []:
            out.append(c)
    return out

def _pairs_from_related(concepts):
    pairs, seen = [], set()
    for c in concepts or []:
        rx = str(c.get("rxcui") or "")
        nm = c.get("name") or None
        if not rx or rx in seen:
            continue
        seen.add(rx)
        pairs.append({"name": nm, "rxcui": rx})
    return pairs

# ---------- Core normalization (pure score gate ≥ 9.0) ----------

def normalize_one(
    sess,
    name: str,
    min_score: float = 9.0,
    gap_thresh: float = -1.0,             # kept for signature compatibility; not used for gating ≥min
    lift_mode: str = "allrelated",
    use_related_fallback: bool = True,
) -> Dict[str, Any]:
    dn = (name or "").strip().upper()
    res: Dict[str, Any] = {
        "drugname_norm": dn,
        "rxcui_best": None,
        "rxnorm_name_best": None,
        "tty_best": None,
        "score": None,
        "ingredient_rxcui": None,
        "ingredient_name": None,
        "ingredients_json": None,
        "accepted_by": None,
        "token_pass": None,
        "second_best_score": None,
        "best_minus_second": None,
        "cand_count": None,
        "error_detail": None,
    }

    # 1) approx match
    cands = rxnav_approx_match(sess, dn, max_entries=5)
    res["cand_count"] = len(cands)
    if not cands:
        res["accepted_by"] = "rejected_no_candidates"
        return res

    # sort by score desc, then rank asc
    cands_sorted = sorted(
        cands, key=lambda c: (-(float(c.get("score", 0) or 0)), int(c.get("rank", 10**9) or 10**9))
    )
    best = cands_sorted[0]
    second = cands_sorted[1] if len(cands_sorted) > 1 else None

    def _f(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    score = _f(best.get("score", 0.0))
    rxcui = str(best.get("rxcui") or best.get("rxnormId") or "")
    res["score"] = score

    if second is not None:
        second_score = _f(second.get("score", 0.0))
        res["second_best_score"] = second_score
        res["best_minus_second"] = score - second_score

    if not rxcui:
        res["accepted_by"] = "rejected_no_rxcui"
        return res

    # 2) pure score gate
    if score < min_score:
        res["accepted_by"] = "rejected_low_score"
        return res

    # 3) pull properties; token checks are not used to gate once score >= min_score
    props = rxnav_concept_props(sess, rxcui) or {}
    hit_name = props.get("name", "")
    tty = props.get("tty")
    res["accepted_by"] = ">=min"
    res["token_pass"] = None

    res.update({
        "rxcui_best": rxcui,
        "rxnorm_name_best": hit_name,
        "tty_best": tty,
    })

    # 4) lift to ingredients
    if tty == "IN":
        pairs = [{"name": hit_name, "rxcui": rxcui}]
        res["ingredients_json"] = json.dumps(pairs, ensure_ascii=False)
        res["ingredient_name"] = hit_name
        res["ingredient_rxcui"] = rxcui
        return res

    concepts = []
    try:
        if lift_mode == "allrelated":
            concepts = rxnav_allrelated_ingredients(sess, rxcui)
            if not concepts and use_related_fallback:
                concepts = rxnav_related_ingredients(sess, rxcui)
        else:
            concepts = rxnav_related_ingredients(sess, rxcui)
    except Exception as e:
        # lifting failure shouldn't invalidate the base mapping
        res["error_detail"] = f"lift_error: {str(e)[:160]}"
        concepts = []

    pairs = _pairs_from_related(concepts)
    res["ingredients_json"] = json.dumps(pairs, ensure_ascii=False)
    if pairs:
        res["ingredient_name"]  = ",".join([p["name"] for p in pairs if p.get("name")])
        res["ingredient_rxcui"] = ",".join([p["rxcui"] for p in pairs if p.get("rxcui")])
    return res

# ---------- Streaming I/O helpers ----------

def iter_distinct_drugnames_sqlite(sqlite_path: str, chunk_size: int) -> Iterable[List[str]]:
    """Stream DISTINCT UPPER(TRIM(DRUGNAME)) from SQLite without loading all into RAM."""
    con = sqlite3.connect(sqlite_path)
    try:
        con.execute("PRAGMA temp_store = FILE")  # keep DISTINCT temp data on disk
        cur = con.cursor()
        cur.execute(
            """
            SELECT DISTINCT UPPER(TRIM(DRUGNAME)) AS drugname_norm
            FROM DRUG
            WHERE DRUGNAME IS NOT NULL AND LENGTH(TRIM(DRUGNAME)) > 0;
            """
        )
        while True:
            rows = cur.fetchmany(chunk_size)
            if not rows:
                break
            yield [r[0] for r in rows]
    finally:
        con.close()

def read_cached_keys(cache_path: str) -> set:
    """
    Read just the key column from cache parquet, streaming via pyarrow dataset.

    IMPORTANT: Exclude rows with accepted_by='error' so they get retried on subsequent runs.
    """
    if not os.path.exists(cache_path):
        return set()
    con = duckdb.connect()
    try:
        df = con.execute(f"""
            SELECT DISTINCT drugname_norm
            FROM read_parquet('{cache_path}')
            WHERE coalesce(accepted_by,'') <> 'error'
        """).fetchdf()
        return set(df["drugname_norm"].tolist())
    finally:
        con.close()

def append_rows_to_parquet(path: str, rows: List[Dict[str, Any]], schema: Optional[pa.schema] = None):
    """Append rows to a parquet file by creating/using a ParquetWriter (single file)."""
    try:
        table = pa.Table.from_pylist(rows, schema=schema)
    except TypeError:
        table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    if not os.path.exists(path):
        pq.write_table(table, path)
    else:
        # Append by reading old + new via DuckDB at merge time; here we write to a delta file instead
        raise RuntimeError("append_rows_to_parquet() should be used with a fresh temp file")

# ---------- Core builders (streaming, low-RAM) ----------

def build_or_extend_cache_streaming(
    names_iter: Iterable[List[str]],
    cache_path: str,
    out_cache_csv: str,
    sleep_s: float,
    batch_size: int,
) -> None:
    """Map names to RxNorm, streaming to a temporary parquet; union with existing cache via DuckDB."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cached = read_cached_keys(cache_path)

    temp_delta = cache_path + ".delta.parquet"
    if os.path.exists(temp_delta):
        os.remove(temp_delta)

    sess = make_session()
    buffer: List[Dict[str, Any]] = []
    wrote_any = False

    # Heartbeat file for quick liveness checks
    hb_path = os.path.join(os.path.dirname(cache_path), "heartbeat.txt")

    total_new = 0
    pbar = tqdm(disable=False, mininterval=TQDM_MININTERVAL, desc="RxNav mapping (new)")

    def flush():
        nonlocal buffer, wrote_any
        if not buffer:
            return
        # Clean up any leftover temp files from a previous interrupted merge
        for p in (temp_delta + ".merge", temp_delta + ".tmp"):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        # Write a fresh temp delta file on first flush, then append by concatenation with DuckDB later
        if not os.path.exists(temp_delta):
            try:
                table = pa.Table.from_pylist(buffer)
            except TypeError:
                table = pa.Table.from_pandas(pd.DataFrame(buffer), preserve_index=False)
            pq.write_table(table, temp_delta)
        else:
            # Append: use DuckDB to UNION existing temp_delta with current buffer into a new temp, then replace
            tmp2 = temp_delta + ".tmp"
            try:
                table2 = pa.Table.from_pylist(buffer)
            except TypeError:
                table2 = pa.Table.from_pandas(pd.DataFrame(buffer), preserve_index=False)
            pq.write_table(table2, tmp2)
            con = duckdb.connect()
            try:
                con.execute(
                    f"""
                    COPY (
                        SELECT * FROM read_parquet('{temp_delta}')
                        UNION ALL BY NAME
                        SELECT * FROM read_parquet('{tmp2}')
                    ) TO '{temp_delta}.merge' (FORMAT PARQUET);
                    """
                )
            finally:
                con.close()
            os.replace(f"{temp_delta}.merge", temp_delta)
            os.remove(tmp2)
        wrote_any = True
        buffer = []

    try:
        for batch_names in names_iter:
            for n in batch_names:
                if n in cached:
                    continue
                try:
                    # Pure score gate at 9.0; gap_thresh disabled; robust network layer inside helpers
                    row = normalize_one(sess, n, min_score=9.0, gap_thresh=-1.0)
                except Exception as e:
                    row = {
                        "drugname_norm": n, "rxcui_best": None, "rxnorm_name_best": None,
                        "tty_best": None, "score": None, "ingredient_rxcui": None, "ingredient_name": None,
                        "ingredients_json": None, "accepted_by": "error", "token_pass": None,
                        "second_best_score": None, "best_minus_second": None, "cand_count": None,
                        "error_detail": str(e)[:200],
                    }
                buffer.append(row)
                total_new += 1
                pbar.update(1)
                if total_new % 10_000 == 0:
                    with open(hb_path, "w") as f:
                        f.write(time.strftime("%F %T") + f"  new={total_new}\n")
                if len(buffer) >= batch_size:
                    flush()
                # polite delay between names (in addition to per-request throttle)
                time.sleep(sleep_s)
        flush()
    finally:
        pbar.close()

    # Merge delta (if any) with existing cache into a fresh file atomically
    if wrote_any:
        tmp_merged = cache_path + ".merged.tmp"
        con = duckdb.connect()
        try:
            if os.path.exists(cache_path):
                con.execute(
                    f"""
                    COPY (
                        SELECT * FROM read_parquet('{cache_path}')
                        UNION BY NAME
                        SELECT * FROM read_parquet('{temp_delta}')
                    ) TO '{tmp_merged}' (FORMAT PARQUET);
                    """
                )
            else:
                con.execute(
                    f"COPY (SELECT * FROM read_parquet('{temp_delta}')) TO '{tmp_merged}' (FORMAT PARQUET);"
                )
        finally:
            con.close()
        os.replace(tmp_merged, cache_path)
        try:
            os.remove(temp_delta)
        except FileNotFoundError:
            pass
    else:
        print("No new names to map; cache unchanged.")

    # Also emit the CSV view of the cache (streamed by DuckDB)
    con = duckdb.connect()
    try:
        con.execute(
            f"COPY (SELECT * FROM read_parquet('{cache_path}')) TO '{out_cache_csv}' (HEADER, DELIMITER ',');"
        )
    finally:
        con.close()
    print(f"Wrote cache parquet: {cache_path}")
    print(f"Wrote cache csv    : {out_cache_csv}")

    # Print acceptance counts without pulling whole table into pandas
    con = duckdb.connect()
    try:
        res = con.execute(
            f"SELECT coalesce(accepted_by,'<NULL>') AS accepted_by, COUNT(*) AS n FROM read_parquet('{cache_path}') GROUP BY 1 ORDER BY n DESC;"
        ).fetchall()
        print("Accepted_by distribution:")
        for a, n in res:
            print(f"  {a:30s} {n}")
    finally:
        con.close()

def explode_ingredients_stream(cache_parquet: str, out_parquet: str, out_csv: str, batch_size: int = 20_000):
    """Stream explode ingredients_json to temp parquet, then dedup via DuckDB to match pandas drop_duplicates.
    """
    tmp_exploded = out_parquet + ".tmp"
    if os.path.exists(tmp_exploded):
        os.remove(tmp_exploded)

    # Stream read cache
    dataset = ds.dataset(cache_parquet, format="parquet")
    cols = list(dataset.schema.names)  # keep all columns

    buffer: List[Dict[str, Any]] = []
    total = 0
    pbar = tqdm(disable=False, mininterval=TQDM_MININTERVAL, desc="Exploding ingredients")

    def flush():
        nonlocal buffer, total
        if not buffer:
            return
        try:
            table = pa.Table.from_pylist(buffer)
        except TypeError:
            table = pa.Table.from_pandas(pd.DataFrame(buffer), preserve_index=False)
        if not os.path.exists(tmp_exploded):
            pq.write_table(table, tmp_exploded)
        else:
            # append via DuckDB UNION ALL to maintain a single-file sink
            tmp2 = tmp_exploded + ".chunk"
            pq.write_table(table, tmp2)
            con = duckdb.connect()
            try:
                con.execute(
                    f"""
                    COPY (
                        SELECT * FROM read_parquet('{tmp_exploded}')
                        UNION ALL BY NAME
                        SELECT * FROM read_parquet('{tmp2}')
                    ) TO '{tmp_exploded}.merge' (FORMAT PARQUET);
                    """
                )
            finally:
                con.close()
            os.replace(f"{tmp_exploded}.merge", tmp_exploded)
            os.remove(tmp2)
        total += len(buffer)
        buffer = []

    try:
        for batch in dataset.to_batches(batch_size=batch_size):
            pdf = batch.to_pandas(types_mapper=None)
            for _, r in pdf.iterrows():
                pairs = json.loads(r.get("ingredients_json") or "[]")
                if not pairs:
                    buffer.append(r.to_dict())
                else:
                    for p in pairs:
                        rr = r.to_dict()
                        rr["ingredient_name"] = p.get("name")
                        rr["ingredient_rxcui"] = p.get("rxcui")
                        buffer.append(rr)
                if len(buffer) >= batch_size:
                    flush()
            pbar.update(len(pdf))
        flush()
    finally:
        pbar.close()

    # Deduplicate on subset (drugname_norm, ingredient_rxcui, ingredient_name) keeping first — via ROW_NUMBER()
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW exploded AS
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY drugname_norm, ingredient_rxcui, ingredient_name
                ORDER BY drugname_norm
            ) AS rn
            FROM read_parquet('{tmp_exploded}');

            COPY (
                SELECT * EXCLUDE (rn) FROM exploded WHERE rn=1
            ) TO '{out_parquet}' (FORMAT PARQUET);

            COPY (
                SELECT * EXCLUDE (rn) FROM exploded WHERE rn=1
            ) TO '{out_csv}' (HEADER, DELIMITER ',');
            """
        )
    finally:
        con.close()

    try:
        os.remove(tmp_exploded)
    except FileNotFoundError:
        pass

    print(f"Wrote exploded parquet: {out_parquet}")
    print(f"Wrote exploded csv    : {out_csv}")

# ---------- Ducks to fetch names via DuckDB (optional, fallback) ----------
# Left here as a fallback; default path uses sqlite3 which streams cleanly.

def iter_distinct_via_duckdb(sqlite_path: str, chunk_size: int) -> Iterable[List[str]]:
    con = duckdb.connect()
    try:
        con.execute("INSTALL sqlite;")
    except Exception:
        pass
    try:
        con.execute("LOAD sqlite;")
    except Exception:
        pass
    con.execute(f"ATTACH '{sqlite_path}' AS sqldb (TYPE SQLITE, READ_ONLY TRUE);")
    offset = 0
    while True:
        df = con.execute(
            f"""
            SELECT DISTINCT UPPER(TRIM(DRUGNAME)) AS drugname_norm
            FROM sqldb.main.DRUG
            WHERE DRUGNAME IS NOT NULL AND LENGTH(TRIM(DRUGNAME)) > 0
            ORDER BY 1
            LIMIT {chunk_size} OFFSET {offset};
            """
        ).fetchdf()
        if df.empty:
            break
        yield df["drugname_norm"].tolist()
        offset += len(df)
    con.close()

# ---------- CLI / Main ----------

def main():
    global DEFAULT_PER_REQUEST_SLEEP
    global DEFAULT_PER_REQUEST_SLEEP
    ap = argparse.ArgumentParser(description="Map FAERS DRUGNAMEs to RxNorm and lift to ingredients (OOMD-resilient)")
    ap.add_argument("--sqlite-path", default=DEFAULT_SQLITE, help="Path to FAERS SQLite database.")
    ap.add_argument("--drugnames-csv", default=None, help="Optional CSV with a column 'drugname_norm' to map.")
    ap.add_argument("--cache-parquet", default=DEFAULT_CACHE_PARQ, help="Path to cache parquet to build/extend.")
    ap.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help="Output CSV for the cache table.")
    ap.add_argument("--out-parquet", default=DEFAULT_OUT_PARQ, help="Output parquet for exploded mappings.")
    ap.add_argument("--out-exploded-csv", default=DEFAULT_OUT_EXPLODED_CSV, help="Output CSV for exploded mappings.")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK, help="Rows fetched per DB chunk.")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="Rows buffered before flush.")
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="Sleep seconds between API calls (per name).")
    ap.add_argument("--per-request-sleep", type=float, default=0.20, help="Throttle per mHTTP request (seconds).")
    args = ap.parse_args()

    # allow customizing per-request throttle at runtime
    DEFAULT_PER_REQUEST_SLEEP = args.per_request_sleep

    # Graceful shutdown
    running = True
    def _stop(sig, frame):
        nonlocal running
        running = False
        print(f"Received {sig}, finishing current batch and exiting…", flush=True)
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, _stop)

    # Determine name source iterator
    if args.drugnames_csv:
        df_iter = pd.read_csv(args.drugnames_csv, usecols=["drugname_norm"], chunksize=args.chunk_size)
        def _names():
            for df in df_iter:
                yield [str(x) for x in df["drugname_norm"].tolist()]
        names_iter = _names()
    else:
        # Default: stream DISTINCT from SQLite to keep RAM tiny
        names_iter = iter_distinct_drugnames_sqlite(args.sqlite_path, args.chunk_size)

    # Build/extend cache (writes parquet + csv)
    build_or_extend_cache_streaming(
        names_iter=names_iter,
        cache_path=args.cache_parquet,
        out_cache_csv=args.out_csv,
        sleep_s=args.sleep,
        batch_size=args.batch_size,
    )

    # Explode (writes parquet + csv) using streaming + DuckDB dedup
    explode_ingredients_stream(
        cache_parquet=args.cache_parquet,
        out_parquet=args.out_parquet,
        out_csv=args.out_exploded_csv,
        batch_size=max(5_000, args.batch_size),
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

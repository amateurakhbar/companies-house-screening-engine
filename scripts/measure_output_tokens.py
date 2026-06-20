"""Measure actual Gemini output (candidates) token count per call on 20 gold
firms. MODE=before uses plain system_instruction; MODE=after uses CachedContent
+ max_output_tokens=200. Picks up clf._system_prompt() live, so the rationale
change is reflected automatically once src/05 is edited."""
import os, pathlib, sys, importlib.util
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("clf", ROOT / "src" / "05_classify_llm.py")
clf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(clf)

import google.generativeai as genai

MODE = os.environ.get("MODE", "before")
api_key = clf._load_api_key()
genai.configure(api_key=api_key)

gold = pd.read_csv(clf.GOLD, dtype={"CompanyNumber": str})
gold = gold[~gold["scrape_status"].isin(clf.NO_TEXT_STATUSES)].head(20)

def firm_text(cn):
    f = clf.SCRAPES / f"{clf._slug(cn)}.txt"
    return f.read_text(encoding="utf-8", errors="replace") if f.exists() else None

if MODE == "after":
    from google.generativeai import caching
    import datetime
    cc = None
    try:
        cc = caching.CachedContent.create(
            model=clf.MODEL, system_instruction=clf._system_prompt(),
            ttl=datetime.timedelta(minutes=15))
        print(f"context cache created: {cc.name}  cached_tokens={cc.usage_metadata.total_token_count}")
        model = genai.GenerativeModel.from_cached_content(
            cached_content=cc,
            generation_config=genai.GenerationConfig(
                temperature=0.0, response_mime_type="application/json",
                max_output_tokens=200))
    except Exception as e:
        print(f"context cache FAILED ({type(e).__name__}: {e}); falling back to system_instruction")
        model = genai.GenerativeModel(
            clf.MODEL, system_instruction=clf._system_prompt(),
            generation_config=genai.GenerationConfig(
                temperature=0.0, response_mime_type="application/json",
                max_output_tokens=200))
else:
    model = genai.GenerativeModel(
        clf.MODEL, system_instruction=clf._system_prompt(),
        generation_config=genai.GenerationConfig(
            temperature=0.0, response_mime_type="application/json"))

outs = []
for _, g in gold.iterrows():
    txt = firm_text(g["CompanyNumber"])
    resp = model.generate_content(clf._user_prompt(g["CompanyName"], g["SICCode.SicText_1"], txt))
    u = resp.usage_metadata
    outs.append(u.candidates_token_count)
    cached = getattr(u, "cached_content_token_count", 0)
    print(f"  {g['CompanyName'][:32]:<32} out={u.candidates_token_count:>4}  "
          f"prompt={u.prompt_token_count:>4}  cached={cached}")

print(f"\nMODE={MODE}  n={len(outs)}  mean_output_tokens={sum(outs)/len(outs):.1f}  "
      f"min={min(outs)} max={max(outs)} total={sum(outs)}")

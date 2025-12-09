"""
main.py

FastAPI application exposing the /api/analyze endpoint.
Loads keyword CSVs at startup and serves analysis requests.

Save as:
product_matcher/backend/app/main.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

from app.matching.datastore import KeywordStore
from app.matching.matcher import Matcher

# --------
# Config paths
# --------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DIAGNOSTIC_CSV = os.path.join(DATA_DIR, "keywords_diagnostic.csv")
ENDO_CSV = os.path.join(DATA_DIR, "keywords_endo.csv")

# NEW CSV (rename your sheet to this file in data/)
SHEET1_CSV = os.path.join(DATA_DIR, "keywords_sheet1.csv")


# --------
# FastAPI app
# --------
app = FastAPI(title="Product Matching API")

# Allow CORS from local dev (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------
# Request model
# --------
class AnalyzeRequest(BaseModel):
    text: str
    category: Optional[str] = "all"   # options: "all", "diagnostic", "endo"


# --------
# Startup: load keywords
# --------
STORE = KeywordStore()

try:
    # 1. Diagnostic keywords
    if os.path.exists(DIAGNOSTIC_CSV):
        STORE.load_csv(DIAGNOSTIC_CSV, category="Diagnostic")
    else:
        print(f"[warning] Diagnostic CSV not found at {DIAGNOSTIC_CSV}")

    # 2. Endo keywords
    if os.path.exists(ENDO_CSV):
        STORE.load_csv(ENDO_CSV, category="Endo")
    else:
        print(f"[warning] Endo CSV not found at {ENDO_CSV}")

    # 3. NEW sheet keywords
    if os.path.exists(SHEET1_CSV):
        STORE.load_csv(SHEET1_CSV, category="General")
        print(f"[info] Loaded Sheet1 keywords from {SHEET1_CSV}")
    else:
        print(f"[warning] Sheet1 CSV not found at {SHEET1_CSV}")

except Exception as e:
    raise RuntimeError(f"Failed to load keyword CSVs: {e}")

# create matcher after all CSVs loaded
MATCHER = Matcher(STORE)


# --------
# Endpoints
# --------
@app.get("/api/health")
async def health():
    return {"status": "ok", "keywords_loaded": STORE.size()}


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text must be a non-empty string")

    # normalize category
    cat = (req.category or "all").strip().lower()
    if cat not in ("all", "diagnostic", "endo"):
        raise HTTPException(status_code=400, detail="category must be one of: all, diagnostic, endo")

    result = MATCHER.analyze(req.text, category_filter=cat)

    # trim large match list
    if "matches" in result and isinstance(result["matches"], list):
        result["matches"] = result["matches"][:200]

    return result

"""AVLE-C FastAPI backend.

Endpoints (all prefixed with /api):
    GET  /api/               – health
    POST /api/analyze        – run the full AVLE-C pipeline
    GET  /api/jobs           – past analyses (MongoDB)
    GET  /api/jobs/{id}      – single job
    GET  /api/ablation       – ablation table (Table 1)
    GET  /api/weights/info   – model metrics & versions
    GET  /api/blockchain     – chain status + recent records
    GET  /api/blockchain/verify/{tx_hash}
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.cors import CORSMiddleware

from avle import blockchain
from avle.config import CONFIG
from avle.docs_api import list_docs, read_doc
from avle.pipeline import analyse

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s - %(message)s")
log = logging.getLogger("avle")

# ---------------- Mongo ---------------- #
mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

# ---------------- App ---------------- #
app = FastAPI(title="AVLE-C API", version="1.0.0")
api = APIRouter(prefix="/api")


# =========================================================================== #
#  Models
# =========================================================================== #
class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    bbox: List[float] = Field(..., min_length=4, max_length=4,
                              description="[west, south, east, north]")
    date_t1_start: str
    date_t1_end:   str
    date_t2_start: str
    date_t2_end:   str
    size: int = 384
    allow_blockchain: bool = True
    use_synthetic: bool = False  # force deterministic synthetic scene (debug)


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    created_at: str


# =========================================================================== #
#  Endpoints
# =========================================================================== #
@api.get("/")
async def root():
    return {"service": "AVLE-C", "version": "1.0.0", "status": "ok"}


@api.post("/analyze")
async def analyze(req: AnalyzeRequest) -> dict:
    bbox = tuple(req.bbox)
    try:
        result = analyse(
            bbox=bbox,
            t1_start=req.date_t1_start, t1_end=req.date_t1_end,
            t2_start=req.date_t2_start, t2_end=req.date_t2_end,
            size=req.size,
            log_blockchain=req.allow_blockchain,
            allow_real=not req.use_synthetic,
        )
    except Exception as e:
        log.exception("analyse failed")
        raise HTTPException(status_code=503, detail=f"pipeline failure: {e}") from e

    doc = {
        "id":         str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request":    req.model_dump(),
        "result":     {k: v for k, v in result.items() if k != "images"},
    }
    result["id"] = doc["id"]
    result["created_at"] = doc["created_at"]
    try:
        await db.analyses.insert_one(doc.copy())
    except Exception as e:
        log.warning("failed to persist analysis to MongoDB: %s", e)
    return result


@api.get("/jobs")
async def list_jobs(limit: int = 50) -> List[dict]:
    cursor = db.analyses.find({}, {"_id": 0, "result.images": 0}) \
                        .sort("created_at", -1).limit(limit)
    return await cursor.to_list(length=limit)


@api.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    doc = await db.analyses.find_one({"id": job_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")
    return doc


@api.get("/ablation")
async def get_ablation() -> dict:
    if not CONFIG.ablation_results.exists():
        raise HTTPException(status_code=404,
                            detail="Ablation not computed yet.  Run train_all.")
    return json.loads(CONFIG.ablation_results.read_text())


@api.get("/weights/info")
async def weights_info() -> dict:
    def _read(p: Path) -> Optional[Any]:
        return json.loads(p.read_text()) if p.exists() else None

    return {
        "segmentation_unet": {
            "present": CONFIG.segmentation_weights.exists(),
            "config":  _read(CONFIG.segmentation_config),
        },
        "carbon_rf": {
            "present": CONFIG.carbon_rf_model.exists(),
            "metrics": _read(CONFIG.carbon_metrics),
        },
        "recommendation_mlp": {
            "present": CONFIG.rec_mlp_weights.exists(),
            "config":  _read(CONFIG.rec_config),
            "metrics": _read(CONFIG.rec_metrics),
        },
        "xgb_projection": {
            "bau_present":         CONFIG.xgb_bau.exists(),
            "mitigation_present":  CONFIG.xgb_mitigation.exists(),
            "ci_present":          CONFIG.xgb_lower.exists() and CONFIG.xgb_upper.exists(),
            "metrics":             _read(CONFIG.xgb_metrics),
        },
        "biome_carbon_range": CONFIG.biome_carbon_range,
        "mitigation_beta":    CONFIG.mitigation_beta,
    }


@api.get("/blockchain")
async def blockchain_status(limit: int = 25) -> dict:
    return {
        "status":  blockchain.chain_status(),
        "records": blockchain.list_records(limit=limit),
    }


@api.get("/blockchain/verify/{tx_hash}")
async def blockchain_verify(tx_hash: str) -> dict:
    res = blockchain.verify(tx_hash)
    if not res.get("valid"):
        raise HTTPException(status_code=404, detail=res.get("error", "tx not found"))
    return res


DATASET_FILES = {
    "carbon":         "carbon_training_data.csv",
    "recommendation": "recommendation_training_data.csv",
    "xgb_bau":        "xgb_bau_training_data.csv",
    "xgb_sequences":  "xgb_raw_sequences.csv",
}


@api.get("/datasets")
async def list_datasets() -> dict:
    out = {}
    for key, name in DATASET_FILES.items():
        p = CONFIG.ablation_results.parent / name
        out[key] = {
            "filename": name,
            "download_url": f"/api/datasets/{key}",
            "size_bytes": p.stat().st_size if p.exists() else None,
            "exists": p.exists(),
        }
    return {
        "files": out,
        "notes": {
            "carbon":         "50 000 synthetic samples grounded in IPCC 2006 Tier-2 biomass carbon stock ranges. Used to fit the Random Forest carbon regressor.",
            "recommendation": "30 000 synthetic samples — rule-based labels perturbed with Gaussian boundary noise (σ=0.05). Used to train the MLP recommender.",
            "xgb_bau":        "Supervised lag-feature rows derived from 500 synthetic 24-month carbon sequences. Used for XGBoost BAU + quantile models.",
            "xgb_sequences":  "The raw 500 × 24-month carbon sequences (one row per sequence, columns t0..t23).",
        },
    }


@api.get("/datasets/{key}")
async def download_dataset(key: str):
    if key not in DATASET_FILES:
        raise HTTPException(status_code=404, detail=f"unknown dataset '{key}'")
    p = CONFIG.ablation_results.parent / DATASET_FILES[key]
    if not p.exists():
        raise HTTPException(status_code=404, detail="dataset not generated yet")
    return FileResponse(str(p), media_type="text/csv",
                        filename=DATASET_FILES[key])


@api.get("/docs")
async def docs_list() -> dict:
    return {"docs": list_docs()}


@api.get("/docs/{slug}")
async def docs_one(slug: str) -> dict:
    doc = read_doc(slug)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"unknown doc '{slug}'")
    return doc


# ---------------- mount ---------------- #
app.include_router(api)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _warm():
    # Pre-load the blockchain so the first /analyze is fast
    blockchain.chain_status()
    log.info("AVLE-C API warmed up")


@app.on_event("shutdown")
async def _shutdown():
    client.close()

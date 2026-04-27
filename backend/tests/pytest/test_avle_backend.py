"""AVLE-C backend API regression tests - iteration 3 (v3 upgrades).

Changes from iter2:
- CSI metric REMOVED from ablation rows
- 10 rows per ablation regime (structured / neutral)
- Ablation family field added
- Segmentation IoU block (n, ndvi_iou, unet_iou, delta_iou)
- Projection adds xgb_rmse + linreg_rmse
- Table = 12 rows (incl. 2 segmentation rows)
- New /api/docs + /api/docs/{slug}
- Suitability gate (water/snow) with zero-carbon edge case
"""
import os
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL").rstrip("/")
API = f"{BASE_URL}/api"

ANALYZE_PAYLOAD = {
    "bbox": [-55.2, -9.8, -54.8, -9.4],
    "date_t1_start": "2019-06-01",
    "date_t1_end": "2019-08-31",
    "date_t2_start": "2023-06-01",
    "date_t2_end": "2023-08-31",
    "size": 256,
    "use_synthetic": True,
    "allow_blockchain": True,
}


@pytest.fixture(scope="session")
def session():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


@pytest.fixture(scope="session")
def analyze_result(session):
    r = session.post(f"{API}/analyze", json=ANALYZE_PAYLOAD, timeout=120)
    assert r.status_code == 200, f"analyze failed: {r.status_code} {r.text[:300]}"
    return r.json()


# ---- Health ----
def test_health(session):
    r = session.get(f"{API}/", timeout=15)
    assert r.status_code == 200
    d = r.json()
    assert d["service"] == "AVLE-C"
    assert d["status"] == "ok"


# ---- Analyze core (unchanged regressions) ----
def test_analyze_core_fields(analyze_result):
    d = analyze_result
    for key in ["avle_plus", "carbon_estimate_tco2", "carbon_ci_lower",
                "carbon_ci_upper", "recommendation", "blockchain", "images",
                "region_hash", "biome", "loss_area_ha", "suitability",
                "carbon_scenario_bau", "carbon_scenario_mitigation",
                "carbon_scenario_unchecked", "carbon_cumulative_bau",
                "carbon_cumulative_mitigation", "carbon_cumulative_unchecked"]:
        assert key in d, f"missing field {key}"

    assert d["loss_area_ha"] > 0, "expected non-zero loss area for Mato Grosso synthetic"
    assert d["carbon_estimate_tco2"] > 0, "expected non-zero carbon"

    imgs = d["images"]
    for k in ["ndvi_t1", "ndvi_t2", "loss_overlay"]:
        assert k in imgs
        assert imgs[k].startswith("data:image")


def test_analyze_suitability_ok(analyze_result):
    suit = analyze_result["suitability"]
    assert suit["suitable"] is True, f"expected suitable=True, got {suit}"
    assert suit.get("mean_ndvi", 0) > 0.1, f"mean_ndvi low: {suit}"


def test_analyze_recommendation_mc(analyze_result):
    rec = analyze_result["recommendation"]
    for k in ["action", "priority", "confidence", "class_probabilities",
              "class_probabilities_std", "mc_passes"]:
        assert k in rec, f"missing rec field {k}"
    assert rec["mc_passes"] == 25
    assert len(rec["class_probabilities_std"]) == 4
    assert rec["confidence"] < 1.0


def test_analyze_three_scenarios(analyze_result):
    d = analyze_result
    for k in ["carbon_scenario_bau", "carbon_scenario_mitigation",
              "carbon_scenario_unchecked", "carbon_cumulative_bau",
              "carbon_cumulative_mitigation", "carbon_cumulative_unchecked"]:
        assert len(d[k]) == 5

    # Cumulative monotonic non-decreasing
    for k in ["carbon_cumulative_bau", "carbon_cumulative_mitigation",
              "carbon_cumulative_unchecked"]:
        vals = d[k]
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1] - 1e-6


# ---- Suitability gate: water-only bbox -> not suitable, zero carbon ----
def test_analyze_water_bbox_unsuitable(session):
    payload = dict(ANALYZE_PAYLOAD)
    payload["bbox"] = [0.0, 0.0, 0.1, 0.1]  # middle of ocean (Gulf of Guinea)
    payload["use_synthetic"] = False  # need real NDVI to decide water
    r = session.post(f"{API}/analyze", json=payload, timeout=180)
    assert r.status_code == 200, f"analyze failed: {r.status_code} {r.text[:300]}"
    d = r.json()
    suit = d.get("suitability", {})
    assert suit.get("suitable") is False, f"expected unsuitable water bbox, got {suit}"
    assert "reason" in suit and suit["reason"], "suitability must include reason"
    assert d["carbon_estimate_tco2"] == 0, \
        f"expected 0 carbon for unsuitable region, got {d['carbon_estimate_tco2']}"


# ---- Ablation (v3: CSI removed, family added, 10 rows, segmentation block) ----
def test_ablation_structure_v3(session):
    r = session.get(f"{API}/ablation", timeout=30)
    assert r.status_code == 200
    d = r.json()
    for s in ["structured_ablation", "neutral_ablation", "segmentation",
              "projection", "recommendation", "table", "meta"]:
        assert s in d, f"missing {s}"

    assert len(d["structured_ablation"]) == 10
    assert len(d["neutral_ablation"]) == 10
    assert len(d["table"]) == 12


def test_ablation_row_fields_no_csi(session):
    r = session.get(f"{API}/ablation", timeout=30)
    d = r.json()
    allowed_families = {"baseline", "strong baseline", "avle progression",
                        "full system"}
    for regime in ["structured_ablation", "neutral_ablation"]:
        for row in d[regime]:
            # csi must be GONE
            assert "csi" not in row, f"csi should be removed but present in {regime} {row['method']}"
            # required fields
            for f in ["method", "family", "rmse", "mae", "r2", "spearman"]:
                assert f in row, f"missing {f} in {regime} {row}"
            assert row["family"] in allowed_families, \
                f"unknown family {row['family']} in {regime}"


def test_ablation_structured_full_quality(session):
    r = session.get(f"{API}/ablation", timeout=30)
    d = r.json()
    rows = {row["method"]: row for row in d["structured_ablation"]}
    # Find the AVLE-C full row (match by substring to tolerate naming)
    full = None
    for k, v in rows.items():
        if "full" in k.lower() and "ml" in k.lower():
            full = v
            break
    assert full is not None, f"no full-ML row found, methods={list(rows)}"
    assert full["r2"] > 0.9, f"AVLE-C full R² should be > 0.9, got {full['r2']}"
    assert full["rmse"] < 600, f"AVLE-C full RMSE should be < 600, got {full['rmse']}"


def test_ablation_segmentation_block(session):
    r = session.get(f"{API}/ablation", timeout=30)
    d = r.json()
    seg = d["segmentation"]
    for k in ["n", "ndvi_iou", "unet_iou", "delta_iou"]:
        assert k in seg, f"segmentation missing {k}"
    assert isinstance(seg["n"], int) and seg["n"] > 0
    # IoU in [0,1]
    assert 0.0 <= seg["ndvi_iou"] <= 1.0
    assert 0.0 <= seg["unet_iou"] <= 1.0
    # delta should be signed (can be negative or positive but finite)
    assert seg["delta_iou"] == pytest.approx(
        seg["unet_iou"] - seg["ndvi_iou"], abs=1e-6)


def test_ablation_projection_has_linreg(session):
    r = session.get(f"{API}/ablation", timeout=30)
    d = r.json()
    proj = d["projection"]
    assert "xgb_rmse" in proj
    assert "linreg_rmse" in proj
    assert "arima" in proj


# ---- Docs endpoints (new) ----
def test_docs_list(session):
    r = session.get(f"{API}/docs", timeout=15)
    assert r.status_code == 200
    d = r.json()
    assert "docs" in d
    slugs = {x["slug"] for x in d["docs"]}
    for expected in ["setup", "project-evolution", "research-paper-guide",
                     "model-training-guide", "data-guide", "interview-prep"]:
        assert expected in slugs, f"missing doc slug {expected}"
    assert len(d["docs"]) == 6


def test_docs_get_setup(session):
    r = session.get(f"{API}/docs/setup", timeout=15)
    assert r.status_code == 200
    d = r.json()
    for k in ["slug", "title", "filename", "markdown"]:
        assert k in d, f"missing {k}"
    assert d["slug"] == "setup"
    assert len(d["markdown"]) > 500, f"markdown too short: {len(d['markdown'])}"


def test_docs_unknown_404(session):
    r = session.get(f"{API}/docs/unknown-doc-xyz", timeout=15)
    assert r.status_code == 404


# ---- Blockchain ----
def test_blockchain_status(session, analyze_result):
    r = session.get(f"{API}/blockchain", timeout=15)
    assert r.status_code == 200
    d = r.json()
    assert "status" in d and "records" in d


def test_blockchain_verify(session, analyze_result):
    tx = analyze_result["blockchain"]["tx_hash"]
    r = session.get(f"{API}/blockchain/verify/{tx}", timeout=15)
    assert r.status_code == 200
    assert r.json().get("valid") is True


# ---- Datasets endpoints still work (just removed from nav) ----
def test_datasets_list(session):
    r = session.get(f"{API}/datasets", timeout=15)
    assert r.status_code == 200
    files = r.json()["files"]
    for k in ["carbon", "recommendation", "xgb_bau", "xgb_sequences"]:
        assert k in files
        assert files[k]["size_bytes"] > 0


def test_dataset_download_carbon(session):
    r = session.get(f"{API}/datasets/carbon", timeout=30)
    assert r.status_code == 200
    assert "text/csv" in r.headers.get("content-type", "")


# ---- Weights info ----
def test_weights_info(session):
    r = session.get(f"{API}/weights/info", timeout=15)
    assert r.status_code == 200
    d = r.json()
    assert d["segmentation_unet"]["present"] is True
    assert d["carbon_rf"]["present"] is True
    assert d["recommendation_mlp"]["present"] is True


# ---- Jobs ----
def test_jobs_list(session, analyze_result):
    r = session.get(f"{API}/jobs", timeout=15)
    assert r.status_code == 200
    assert isinstance(r.json(), list)

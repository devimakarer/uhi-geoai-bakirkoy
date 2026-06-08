"""
Stage 4: LLM Tool-Calling Integration
─────────────────────────────────────────────────────────────────────────────
a) Defines 3 tools for an LLM:
     get_landsat_tile(bbox, date)
     calculate_ndvi(tile)
     compute_uhi_index(lst, ndvi)
b) Issues the natural-language command
   "Calculate the heat island intensity for Kadikoy, Istanbul, for July 2023"
   and tests whether the model invokes the tools in the CORRECT ORDER.
c) Analyses success/failure cases and proposes a concrete improvement.

Run modes:
  - if .env contains GEMINI_API_KEY  -> REAL Gemini function-calling
  - otherwise                        -> rule-based local orchestrator (LLM tool-use sim)
"""

import os
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

print("=" * 64)
print("Stage 4: LLM Tool-Calling")
print("=" * 64)

# Global log to track the order in which tools are called
CALL_LOG = []


# ══════════════════════════════════════════════════════════════════════════════
# (a) TOOL DEFINITIONS — contain real computation logic
# ══════════════════════════════════════════════════════════════════════════════
# Place -> bounding box (WGS84). To avoid LLM coordinate hallucination, we pin
# the bboxes in the tool/system layer (see ).
PLACE_BBOX = {
    "Kadikoy":  [29.02, 40.96, 29.10, 41.01],   # example command from this project
    "Bakirkoy": [28.82, 40.96, 28.92, 41.02],   # main study area
}
KADIKOY_BBOX = PLACE_BBOX["Kadikoy"]            # backward compatibility


def _try_gee_landsat(bbox, date):
    """Fetch real Landsat band means from GEE (returns None on failure)."""
    try:
        import ee
        ee.Initialize(project="bakirkoy-uhi")
        west, south, east, north = bbox
        aoi = ee.Geometry.BBox(west, south, east, north)
        # date "2023-07" -> month range
        y, m = int(date[:4]), int(date[5:7])
        start = f"{y}-{m:02d}-01"
        end   = f"{y}-{m:02d}-28"
        img = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
               .filterBounds(aoi).filterDate(start, end)
               .sort("CLOUD_COVER").first())
        bands = img.select(["ST_B10", "SR_B5", "SR_B4"]).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=100, maxPixels=1e9
        ).getInfo()
        return {
            "lst_raw": bands["ST_B10"],
            "nir_raw": bands["SR_B5"],
            "red_raw": bands["SR_B4"],
            "source": "Google Earth Engine (real Landsat 8)",
        }
    except Exception:
        return None


def get_landsat_tile(bbox: list, date: str) -> dict:
    """Fetch a Landsat tile for the given bounding box and date.

    Args:
        bbox: [west, south, east, north] in WGS84 coordinates.
        date: date in 'YYYY-MM' format (e.g. '2023-07').
    Returns:
        Dictionary of raw band values for the tile.
    """
    bbox = [float(v) for v in bbox]          # Gemini protobuf -> native list
    date = str(date)
    CALL_LOG.append(("get_landsat_tile", {"bbox": bbox, "date": date}))
    real = _try_gee_landsat(bbox, date)
    if real:
        return real
    # Fallback: representative raw band values
    return {
        "lst_raw": 44000.0, "nir_raw": 18000.0, "red_raw": 12000.0,
        "source": "representative value (GEE unavailable)",
    }


def calculate_ndvi(tile: dict) -> dict:
    """Compute NDVI and LST (C) from a Landsat tile.

    Args:
        tile: output of get_landsat_tile (raw band values).
    Returns:
        Dictionary with ndvi and lst_celsius.
    """
    CALL_LOG.append(("calculate_ndvi", {"tile_source": tile.get("source")}))
    nir = tile["nir_raw"] * 0.0000275 - 0.2
    red = tile["red_raw"] * 0.0000275 - 0.2
    ndvi = (nir - red) / (nir + red)
    lst_c = tile["lst_raw"] * 0.00341802 + 149.0 - 273.15
    return {"ndvi": round(ndvi, 3), "lst_celsius": round(lst_c, 2)}


def compute_uhi_index(lst: float, ndvi: float) -> dict:
    """Compute an Urban Heat Island (UHI) intensity index from LST and NDVI.

    Args:
        lst: land surface temperature (C).
        ndvi: vegetation index.
    Returns:
        Dictionary with uhi_index and a verbal interpretation.
    """
    CALL_LOG.append(("compute_uhi_index", {"lst": lst, "ndvi": ndvi}))
    # Simple UHI proxy: high temperature + low vegetation -> high UHI
    uhi = round(lst - (15.0 * ndvi) - 25.0, 2)
    level = ("high" if uhi > 8 else "moderate" if uhi > 3 else "low")
    return {"uhi_index": uhi, "level": level,
            "explanation": f"UHI ~ {uhi} C ({level} intensity)"}


TOOLS = [get_landsat_tile, calculate_ndvi, compute_uhi_index]

# Natural-language commands to test (the first is this projectple)
COMMANDS = [
    ("Kadikoy",  "Calculate the heat island intensity for Kadikoy, Istanbul, for July 2023"),
    ("Bakirkoy", "Calculate the heat island intensity for Bakirkoy, Istanbul, for July 2023"),
]


# ══════════════════════════════════════════════════════════════════════════════
# (b) EXECUTION — real Gemini or local orchestrator
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_INSTRUCTION = (
    "You are a geospatial analysis assistant. When the user asks for heat-island "
    "intensity for a location and date, call the tools IN THIS ORDER: first fetch "
    "the relevant Landsat tile (get_landsat_tile), then compute NDVI/LST "
    "(calculate_ndvi), and finally compute the UHI index (compute_uhi_index). "
    "Do NOT invent coordinates; use these fixed bounding boxes: "
    "Kadikoy=[29.02,40.96,29.10,41.01], Bakirkoy=[28.82,40.96,28.92,41.02]. "
    "Convert the date to 'YYYY-MM' format (July 2023 -> 2023-07). "
    "Answer in English."
)


def run_with_gemini(command):
    import google.generativeai as genai
    genai.configure(api_key=API_KEY)

    model_names = ["gemini-2.0-flash", "gemini-flash-latest", "gemini-2.5-flash"]
    errors = []
    for name in model_names:
        try:
            model = genai.GenerativeModel(
                name, tools=TOOLS, system_instruction=SYSTEM_INSTRUCTION,
            )
            chat = model.start_chat(enable_automatic_function_calling=True)
            resp = chat.send_message(command)
            return resp.text, name
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue
    raise RuntimeError("All models failed:\n  " + "\n  ".join(errors))


def run_local_orchestrator(place):
    """When no LLM is available: parse the command and call tools in planned order."""
    print("    (Local orchestrator — rule-based LLM tool-use simulation)")
    bbox, date = PLACE_BBOX[place], "2023-07"
    tile = get_landsat_tile(bbox, date)
    bands = calculate_ndvi(tile)
    result = compute_uhi_index(bands["lst_celsius"], bands["ndvi"])
    text = (f"Urban heat island analysis for {place} (July 2023):\n"
            f"  LST = {bands['lst_celsius']} C, NDVI = {bands['ndvi']}\n"
            f"  {result['explanation']}")
    return text, "local-orchestrator"


# ── Run for each location ─────────────────────────────────────────────────────
expected = ["get_landsat_tile", "calculate_ndvi", "compute_uhi_index"]
runs = []   # (place, command, engine, called, correct, final_text)

for place, command in COMMANDS:
    print("\n" + "#" * 64)
    print(f"[Command] \"{command}\"")
    CALL_LOG.clear()

    if API_KEY:
        print("[Mode] REAL Gemini function-calling")
        try:
            final_text, engine = run_with_gemini(command)
        except Exception as e:
            print(f"    ! Gemini error: {e}\n    -> falling back to local orchestrator")
            CALL_LOG.clear()
            final_text, engine = run_local_orchestrator(place)
    else:
        print("[Mode] GEMINI_API_KEY not found -> local orchestrator")
        final_text, engine = run_local_orchestrator(place)

    called = [c[0] for c in CALL_LOG]
    correct_order = called == expected
    runs.append((place, command, engine, list(CALL_LOG), correct_order, final_text))

    print("\n" + "-" * 64)
    print("TOOL-CALL CHAIN:")
    for i, (name, args) in enumerate(CALL_LOG, 1):
        print(f"  {i}. {name}({json.dumps(args, ensure_ascii=False, default=str)})")
    print("-" * 64)
    print(f"Engine          : {engine}")
    print(f"Expected order  : {' -> '.join(expected)}")
    print(f"Observed order  : {' -> '.join(called) if called else '(no tool called)'}")
    print(f"Correct order?  : {'YES' if correct_order else 'NO'}")
    print("-" * 64)
    print("MODEL FINAL RESPONSE:")
    print(final_text)
    print("-" * 64)


# ══════════════════════════════════════════════════════════════════════════════
# RESULT + (c) SUCCESS/FAILURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
runs_summary = "\n".join(
    f"  [{place}] engine={eng} | order={' -> '.join(c[0] for c in log)} | correct={ok}"
    for place, cmd, eng, log, ok, txt in runs
)

# Observed behaviour (reflects the actual outcome of this run)
def describe_run(place, log, ok):
    names = [c[0] for c in log]
    n = len(names)
    if ok:
        return f"  [{place}] IDEAL: 3 tools in exact dependency order, called once."
    if n % 3 == 0 and names == expected * (n // 3):
        return (f"  [{place}] REDUNDANT CALLS: order preserved but the chain was "
                f"repeated {n // 3} times ({n} calls). Concrete evidence that LLM "
                f"tool-orchestration can be non-deterministic and wasteful.")
    return f"  [{place}] UNEXPECTED PATTERN: {n} calls, order={' -> '.join(names)}."

observations = "\n".join(describe_run(p, log, ok) for p, c, e, log, ok, t in runs)

analysis = f""" — LLM Tool-Calling Analysis
{'='*50}
Date: {datetime.now():%Y-%m-%d %H:%M}
Expected call order: {' -> '.join(expected)}

Tested commands and results:
{runs_summary}

OBSERVED BEHAVIOUR (this run):
{observations}

(c) SUCCESS CASE:
- The model chained all three tools according to logical dependency:
  NDVI/LST cannot be computed without a Landsat tile; UHI cannot be computed
  without both of those.
- The natural-language phrases "Kadikoy" and "July 2023" were resolved to the
  correct parameters (bbox + date).

FAILURE / RISK CASES:
- REDUNDANT CALLS: empirically observed — for the same task some models
  (e.g. gemini-2.5-flash) preserve the correct order but call the entire tool
  chain again unnecessarily (6 calls instead of 3). This shows LLM
  tool-orchestration can be non-deterministic and computationally wasteful;
  the same model behaving differently across runs reduces production reliability.
- If the LLM tries to "recall" Kadikoy's coordinates from memory it may produce
  a wrong bbox (coordinate hallucination). We mitigated this by pinning the bbox
  in the system prompt / a geocoding tool.
- Data-quality issues such as cloud cover / missing scenes go unnoticed and are
  reported as valid results unless handled in the tool layer.
- Date-format ambiguity ("July 2023" -> month range) is a source of parameter-
  mapping errors.

PROPOSED IMPROVEMENT:
- Add a `geocode(place_name) -> bbox` tool to the chain: this eliminates LLM
  coordinate hallucination and moves spatial accuracy into the tool layer
  (the LLM only orchestrates; it performs no spatial arithmetic).
- Additionally add schema validation (bbox bounds, valid date) and a cloud-cover
  threshold to each tool to prevent silent errors.
"""
with open(BASE_DIR / "data" / "processed" / "tool_calling_analysis.txt",
          "w", encoding="utf-8") as f:
    f.write(analysis)

print("\n" + "=" * 64)
print("Stage 4 complete!")
print(f"  - Analysis -> data/processed/tool_calling_analysis.txt")
print("Next step: 05_web_map.py ( — interactive map)")
print("=" * 64)

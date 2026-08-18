"""
Lumina Creative Tool — brain_region_lookup
Created : 2026-08-18T12:57:53
Purpose : Provides an ASCII summary, detailed lookup, and JSON export of brain regions linked to reward processing and curiosity, highlighting their overlap.
"""

"""
brain_region_lookup.py

A small, self‑contained reference for brain regions implicated in
reward processing and curiosity.  It prints an ASCII summary,
allows lookup of a specific region, reports overlap between the two
domains, and writes the full database to `brain_regions.json`.

Only the Python standard library is used.
"""

import json
import sys
from pathlib import Path
from textwrap import fill
from typing import Dict, List, Set

# ----------------------------------------------------------------------
# Static database -------------------------------------------------------
# ----------------------------------------------------------------------
# Each entry contains a short description and the categories it belongs to.
# Categories: "reward", "curiosity"
BRAIN_REGIONS: Dict[str, Dict] = {
    "ventral tegmental area (VTA)": {
        "description": "Midbrain dopaminergic nucleus that signals reward prediction errors.",
        "categories": {"reward"},
    },
    "nucleus accumbens (NAc)": {
        "description": "Core of the ventral striatum; integrates dopaminergic inputs for reward valuation.",
        "categories": {"reward"},
    },
    "orbitofrontal cortex (OFC)": {
        "description": "Prefrontal region that encodes the value of expected outcomes.",
        "categories": {"reward"},
    },
    "amygdala": {
        "description": "Limbic structure linking emotional salience to reward learning.",
        "categories": {"reward"},
    },
    "dorsal anterior cingulate cortex (dACC)": {
        "description": "Detects conflict and information‑seeking demands, linked to curiosity-driven behavior.",
        "categories": {"curiosity"},
    },
    "hippocampus": {
        "description": "Supports memory formation and novelty detection, fueling curiosity.",
        "categories": {"curiosity"},
    },
    "lateral prefrontal cortex (lPFC)": {
        "description": "Imposes top‑down control during exploratory decision‑making.",
        "categories": {"curiosity"},
    },
    "ventrolateral prefrontal cortex (vlPFC)": {
        "description": "Involved in evaluating uncertain outcomes and guiding information‑seeking.",
        "categories": {"curiosity"},
    },
    "parietal cortex (intraparietal sulcus, IPS)": {
        "description": "Tracks attentional priority and information‑gathering demands.",
        "categories": {"curiosity"},
    },
    "ventral striatum": {
        "description": "Broad region encompassing NAc; integrates reward and novelty signals.",
        "categories": {"reward", "curiosity"},
    },
    "midbrain dopaminergic system": {
        "description": "Includes VTA and substantia nigra; releases dopamine for both reward and novelty.",
        "categories": {"reward", "curiosity"},
    },
}


# ----------------------------------------------------------------------
# Helper functions ------------------------------------------------------
# ----------------------------------------------------------------------
def save_database(path: Path = Path("brain_regions.json")) -> None:
    """Write the full database to a JSON file (pretty‑printed)."""
    serializable = {
        region: {
            "description": data["description"],
            "categories": sorted(list(data["categories"])),
        }
        for region, data in BRAIN_REGIONS.items()
    }
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))
    print(f"✅ Database saved to {path.resolve()}")


def list_regions() -> None:
    """Print an ASCII table of all regions grouped by category."""
    reward = [r for r, d in BRAIN_REGIONS.items() if "reward" in d["categories"]]
    curiosity = [r for r, d in BRAIN_REGIONS.items() if "curiosity" in d["categories"]]

    def fmt(lst: List[str]) -> str:
        return "\n".join(f"  - {r}" for r in sorted(lst))

    print("\n=== Brain Regions Involved in Reward ===")
    print(fmt(reward))
    print("\n=== Brain Regions Involved in Curiosity ===")
    print(fmt(curiosity))

    overlap = set(reward) & set(curiosity)
    if overlap:
        print("\n=== Overlapping Regions (Reward ↔ Curiosity) ===")
        print("\n".join(f"  * {r}" for r in sorted(overlap)))
    else:
        print("\n(No overlapping regions found.)")


def lookup_region(name: str) -> None:
    """Show detailed info for a region (case‑insensitive fuzzy match)."""
    name_lower = name.lower()
    matches = [r for r in BRAIN_REGIONS if name_lower in r.lower()]
    if not matches:
        print(f"❓ No region matches '{name}'.")
        return
    for region in matches:
        data = BRAIN_REGIONS[region]
        print(f"\n--- {region} ---")
        print(fill(data["description"], width=80))
        cats = ", ".join(sorted(data["categories"]))
        print(f"Categories: {cats}")


def overlap_summary() -> Set[str]:
    """Return the set of regions belonging to both categories."""
    return {
        region
        for region, data in BRAIN_REGIONS.items()
        if {"reward", "curiosity"}.issubset(data["categories"])
    }


# ----------------------------------------------------------------------
# Command‑line interface ------------------------------------------------
# ----------------------------------------------------------------------
def main(argv: List[str]) -> None:
    """
    Usage:
      python brain_region_lookup.py          # print full tables + save JSON
      python brain_region_lookup.py save     # only save JSON
      python brain_region_lookup.py info <region name>  # detailed lookup
    """
    if len(argv) == 0:
        list_regions()
        save_database()
    elif argv[0] == "save":
        save_database()
    elif argv[0] == "info" and len(argv) > 1:
        lookup_region(" ".join(argv[1:]))
    else:
        print("Invalid arguments.\n" + main.__doc__)

if __name__ == "__main__":
    main(sys.argv[1:])

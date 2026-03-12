import json
from pathlib import Path
from typing import Dict, Optional


def _build_country_to_continent_map(json_path: Path) -> Dict[str, str]:
    with json_path.open("r", encoding="utf-8") as f:
        continent_to_countries = json.load(f)

    country_to_continent: Dict[str, str] = {}
    for continent, countries in continent_to_countries.items():
        for country_code in countries:
            code = country_code.upper()
            if code in country_to_continent:
                raise ValueError(f"Duplicate country code found in JSON: {code}")
            country_to_continent[code] = continent
    return country_to_continent


_JSON_PATH = Path(__file__).resolve().parent / "osv5m_10_class_iso.json"
COUNTRY_TO_CONTINENT = _build_country_to_continent_map(_JSON_PATH)


def country_to_continent(country_code: str) -> Optional[str]:
    """
    O(1) average-case lookup using a Python hash table (dict).
    Example: country_to_continent("US") -> "North_America"
    """
    return COUNTRY_TO_CONTINENT.get(country_code.upper())

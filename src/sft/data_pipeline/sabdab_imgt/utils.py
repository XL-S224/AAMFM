import datetime
from typing import Any, Iterable, List, Optional


def nan_to_empty_string(val: Any) -> str:
    if val != val or val is None:
        return ""
    return str(val)


def nan_to_none(val: Any) -> Optional[Any]:
    if val != val or val is None or val == "":
        return None
    return val


def split_sabdab_delimited_str(val: str) -> List[str]:
    if not val:
        return []
    return [s.strip() for s in val.split("|")]


def parse_sabdab_resolution(val: Any) -> Optional[float]:
    if val == "NOT" or not val or val != val:
        return None
    if isinstance(val, str) and "," in val:
        return float(val.split(",")[0].strip())
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def build_entry_id(pdbcode: str, h_chain: str, l_chain: str, ag_chains: Iterable[str]) -> str:
    ag = "".join(ag_chains)
    return f"{pdbcode}_{h_chain}_{l_chain}_{ag}"


def parse_date(value: str) -> Optional[datetime.datetime]:
    if not value:
        return None
    try:
        return datetime.datetime.strptime(value, "%m/%d/%y")
    except ValueError:
        return None

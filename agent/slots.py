"""The structured request the agent is filling in. Corrections are field-level merges."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import date

from mocks.world_seed import SERVICES

REQUIRED = ("service", "area", "date_from", "date_to", "budget_max")

SERVICE_ALIASES = {
    "plumbing": "plumber", "plumber": "plumber", "electrical": "electrician", "electrician": "electrician",
    "ac": "ac_service", "ac_service": "ac_service", "ac servicing": "ac_service", "ac repair": "ac_service",
    "cleaning": "cleaning", "cleaner": "cleaning", "carpenter": "carpenter", "pest_control": "pest_control",
    "pest control": "pest_control",
}


@dataclass
class Slots:
    service: str | None = None
    area: str | None = None
    date_from: str | None = None       # ISO date
    date_to: str | None = None         # ISO date
    earliest_hour: int | None = None   # inclusive, 24h
    latest_hour: int | None = None     # inclusive start hour
    budget_max: int | None = None      # INR
    provider_id: str | None = None     # explicit technician preference
    notes: str | None = None
    # which fields were filled from memory (so the agent can say so, and the user can override)
    from_memory: list[str] = field(default_factory=list)

    def missing(self) -> list[str]:
        return [f for f in REQUIRED if getattr(self, f) in (None, "")]

    def merge(self, updates: dict) -> list[str]:
        """Apply only the provided fields. Returns the list of fields that changed.
        Unrelated fields are preserved — this is what makes corrections safe."""
        changed = []
        names = {f.name for f in fields(self)} - {"from_memory"}
        for k, v in updates.items():
            if k not in names or v is None:
                continue
            v = normalize(k, v)
            if getattr(self, k) != v:
                setattr(self, k, v)
                changed.append(k)
                if k in self.from_memory:
                    self.from_memory.remove(k)
        return changed

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Slots":
        return cls(**d)


def normalize(key: str, value):
    if key == "service":
        v = str(value).strip().lower()
        v = SERVICE_ALIASES.get(v, v)
        if v not in SERVICES:
            raise ValueError(f"unknown service '{value}'. Known: {SERVICES}")
        return v
    if key in ("date_from", "date_to"):
        return date.fromisoformat(str(value)[:10]).isoformat()
    if key in ("earliest_hour", "latest_hour"):
        h = int(value)
        if not 0 <= h <= 23:
            raise ValueError("hour must be 0-23")
        return h
    if key == "budget_max":
        return int(float(str(value).replace(",", "").replace("₹", "")))
    if key == "area":
        return str(value).strip().title().replace("Hsr", "HSR")
    return value

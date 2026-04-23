"""Framework for loading, caching, and joining external data signals.

Every external source (FX rates, weather, holidays, search trends, ...) implements
:class:`BaseSignalLoader`.  Loaders declare their publication lag in days so the
leakage guard can confirm that no signal bleeds information from the future into
the training set.

Design goals:
- Uniform interface for every source so the ablation harness can iterate over them.
- Cheap cache (parquet + .meta.json) to avoid repeated network calls.
- Network failures degrade gracefully (fall back to stale cache with a warning).
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd

from src.config import OUTPUT_DIR

logger = logging.getLogger(__name__)

EXTERNAL_CACHE_DIR = OUTPUT_DIR / "external"
DEFAULT_CACHE_TTL_DAYS = 7


@dataclass
class LoaderMetadata:
    """Schema-versioned metadata stored next to cached parquet."""

    source_name: str
    fetch_timestamp: str
    upstream_url: str
    row_count: int
    date_range_min: str
    date_range_max: str
    schema_hash: str
    signal_cols: list[str] = field(default_factory=list)
    publication_lag_days: int = 0

    def to_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "fetch_timestamp": self.fetch_timestamp,
            "upstream_url": self.upstream_url,
            "row_count": self.row_count,
            "date_range_min": self.date_range_min,
            "date_range_max": self.date_range_max,
            "schema_hash": self.schema_hash,
            "signal_cols": list(self.signal_cols),
            "publication_lag_days": self.publication_lag_days,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LoaderMetadata":
        return cls(**d)


class BaseSignalLoader(ABC):
    """Abstract base for every external signal loader.

    Subclasses must implement :meth:`fetch_raw` (network/file I/O) and
    :meth:`transform` (shape raw data into the monthly (Period, optional keys,
    signal_cols) contract).
    """

    name: str = "base"
    signal_cols: list[str] = []
    join_keys: list[str] = ["Период"]
    publication_lag_days: int = 0
    upstream_url: str = ""
    cache_ttl_days: int = DEFAULT_CACHE_TTL_DAYS

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or EXTERNAL_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / f"{self.name}.parquet"

    @property
    def meta_path(self) -> Path:
        return self.cache_dir / f"{self.name}.meta.json"

    def _schema_hash(self) -> str:
        payload = "|".join(sorted(self.signal_cols) + sorted(self.join_keys))
        return sha256(payload.encode()).hexdigest()[:16]

    @abstractmethod
    def fetch_raw(self) -> pd.DataFrame:
        """Return raw network/file data. No transformation."""

    @abstractmethod
    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Shape raw data to monthly grain with required columns."""

    def validate(self, df: pd.DataFrame) -> None:
        """Enforce contract: Период present as period[M], signal_cols present,
        no duplicate (Период, *extra_join_keys) rows."""

        assert "Период" in df.columns, f"{self.name}: missing Период column"
        assert str(df["Период"].dtype).startswith("period"), (
            f"{self.name}: Период must be period[M], got {df['Период'].dtype}"
        )
        missing = [c for c in self.signal_cols if c not in df.columns]
        assert not missing, f"{self.name}: missing signal columns {missing}"
        dup_mask = df.duplicated(subset=self.join_keys)
        n_dup = int(dup_mask.sum())
        assert n_dup == 0, f"{self.name}: {n_dup} duplicate rows on join keys {self.join_keys}"

    def _cache_is_fresh(self) -> bool:
        if not self.cache_path.exists() or not self.meta_path.exists():
            return False
        try:
            meta = json.loads(self.meta_path.read_text())
        except Exception:  # noqa: BLE001
            return False
        if meta.get("schema_hash") != self._schema_hash():
            return False
        fetched_at = datetime.fromisoformat(meta["fetch_timestamp"].replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - fetched_at).days
        return age_days < self.cache_ttl_days

    def _write_cache(self, df: pd.DataFrame) -> None:
        df.to_parquet(self.cache_path, index=False)
        meta = LoaderMetadata(
            source_name=self.name,
            fetch_timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            upstream_url=self.upstream_url,
            row_count=len(df),
            date_range_min=str(df["Период"].min()),
            date_range_max=str(df["Период"].max()),
            schema_hash=self._schema_hash(),
            signal_cols=list(self.signal_cols),
            publication_lag_days=self.publication_lag_days,
        )
        self.meta_path.write_text(json.dumps(meta.to_dict(), indent=2))

    def _read_cache(self) -> pd.DataFrame:
        df = pd.read_parquet(self.cache_path)
        if "Период" in df.columns:
            df["Период"] = df["Период"].astype("period[M]")
        return df

    def load(self, force_refresh: bool = False) -> pd.DataFrame:
        """Return the transformed monthly DataFrame, cache-aware."""

        if not force_refresh and self._cache_is_fresh():
            df = self._read_cache()
            self.validate(df)
            return df

        try:
            raw = self.fetch_raw()
            df = self.transform(raw)
            self.validate(df)
            self._write_cache(df)
            return df
        except Exception as exc:  # noqa: BLE001
            if self.cache_path.exists():
                logger.warning(
                    "%s fetch failed (%s) — falling back to stale cache", self.name, exc
                )
                df = self._read_cache()
                self.validate(df)
                return df
            raise


# ── Registry ────────────────────────────────────────────────────────────────
LOADER_REGISTRY: dict[str, type[BaseSignalLoader]] = {}


def register_loader(loader_cls: type[BaseSignalLoader]) -> type[BaseSignalLoader]:
    """Class decorator to register a loader by its ``name`` attribute."""

    name = loader_cls.name
    if not name or name == "base":
        raise ValueError(f"Loader {loader_cls.__name__} must set a unique .name")
    if name in LOADER_REGISTRY:
        raise ValueError(f"Duplicate loader name: {name}")
    LOADER_REGISTRY[name] = loader_cls
    return loader_cls


def get_loader(name: str) -> BaseSignalLoader:
    if name not in LOADER_REGISTRY:
        raise KeyError(f"Unknown loader {name!r}; registered: {list(LOADER_REGISTRY)}")
    return LOADER_REGISTRY[name]()


def list_loaders() -> list[str]:
    return sorted(LOADER_REGISTRY)


def import_default_loaders() -> None:
    """Import all loader modules so they register themselves.

    Kept as an explicit call (not auto-imported at module load) so that merely
    importing :mod:`src.external_data` never triggers network I/O.
    """

    # Local imports to avoid cycles and to keep the module cheap to import.
    from src import loaders  # noqa: F401  (side-effect: registers loaders)
    _ = loaders  # silence unused

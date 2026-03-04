# config.py
from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_TZ = "Europe/Brussels"  # UI timezone (ENTSO-E timestamps are UTC underneath)


@dataclass(frozen=True)
class EntsoeAppConfig:
    entsoe_api_key_env: str = "ENTSOE_API"
    entsoe_endpoint_env: str = "ENTSOE_ENDPOINT_URL"

    default_zone_label: str = "ES (Spain)"
    default_months_back: int = 24


CONFIG = EntsoeAppConfig()


def get_api_key() -> str | None:
    v = os.getenv(CONFIG.entsoe_api_key_env)
    return v.strip() if v else None

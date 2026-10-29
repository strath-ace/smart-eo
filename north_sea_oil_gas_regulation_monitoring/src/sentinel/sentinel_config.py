"""Shared Copernicus Data Space configuration for Sentinel collectors."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from sentinelhub import DataCollection, SHConfig

try:
    from .paths import PROJECT_ROOT
except ImportError:
    from paths import PROJECT_ROOT


ENV_PATH = PROJECT_ROOT / ".env"
CDSE_BASE_URL = "https://sh.dataspace.copernicus.eu"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)
SENTINEL5P_CDSE = DataCollection.SENTINEL5P.define_from(
    "SENTINEL5P_CDSE",
    service_url=CDSE_BASE_URL,
)


def load_cdse_config(*, require_credentials: bool = True) -> SHConfig:
    """Load CDSE credentials from the repository-root .env file."""
    load_dotenv(dotenv_path=ENV_PATH)

    client_id = os.getenv("CDSE_CLIENT_ID")
    client_secret = os.getenv("CDSE_CLIENT_SECRET")
    if require_credentials and (not client_id or not client_secret):
        raise RuntimeError(
            "CDSE OAuth credentials were not found. Add CDSE_CLIENT_ID and "
            f"CDSE_CLIENT_SECRET to {ENV_PATH}."
        )

    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_base_url = CDSE_BASE_URL
    config.sh_token_url = CDSE_TOKEN_URL
    return config

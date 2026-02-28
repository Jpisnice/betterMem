from bettermem.api.client import BetterMem
from bettermem.api.config import BetterMemConfig


def test_config_and_client_import() -> None:
    cfg = BetterMemConfig()
    client = BetterMem(config=cfg)
    assert client.config.order in (1, 2)


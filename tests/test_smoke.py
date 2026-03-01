from bettermem.api.client import BetterMem
from bettermem.api.config import BetterMemConfig


def test_config_and_client_import() -> None:
    cfg = BetterMemConfig()
    client = BetterMem(config=cfg)
    assert client.config.max_steps >= 1
    assert 0.0 <= client.config.smoothing_lambda <= 1.0


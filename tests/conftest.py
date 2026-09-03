import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    from make_tiny_models import main

    d = tmp_path_factory.mktemp("tiny")
    main(str(d))
    return d


@pytest.fixture(scope="session")
def tiny_cfg(tiny_dir):
    from state_bridge.config import load_config

    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg["models"]["sender"]["path"] = str(tiny_dir / "sender")
    cfg["models"]["receiver"]["path"] = str(tiny_dir / "receiver")
    cfg["runs_dir"] = str(tiny_dir / "runs")
    return cfg

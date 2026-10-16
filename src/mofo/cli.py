"""Hydra entry point for mofo scripts (minimal port of legacy motionforge/cli.py run_hydra)."""

import sys
from collections.abc import Callable
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "config")


def run(main: Callable[[DictConfig], None], config_name: str, schemas: dict[str, type], commands=()) -> None:
  """Run ``main`` on config/<config_name>.yaml.

  Each ``schemas`` group (name -> frozen dataclass) gets a typed ``schema`` node that its
  YAML options extend, so unknown keys and bad types fail at parse time; read one with
  ``OmegaConf.to_object(cfg.<group>)``. A leading bare word in ``commands`` is shorthand
  for ``cmd=<word>``.
  """
  if len(sys.argv) > 1 and sys.argv[1] in commands:
    sys.argv[1] = f"cmd={sys.argv[1]}"
  for group, schema in schemas.items():
    node = OmegaConf.structured(schema)
    OmegaConf.set_readonly(node, False)  # Frozen dataclass: let YAML and overrides merge.
    ConfigStore.instance().store(group=group, name="schema", node=node)
  hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name=config_name)(main)()

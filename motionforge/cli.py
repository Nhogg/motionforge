"""Shared command-line entry points for reproducible MotionForge experiments.

Hydra supplies typed configuration overrides while experiment modules retain a
plain ``main(config)`` function that can also be called directly from tests.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TypeVar

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

ConfigT = TypeVar("ConfigT")


def run_hydra(
    config_type: type[ConfigT],
    main: Callable[[ConfigT], None],
) -> None:
    """Run ``main`` using a dataclass-backed Hydra configuration."""
    ConfigStore.instance().store(name="config", node=config_type)

    # MotionForge experiments already own their output directories and
    # machine-readable logs. Avoid Hydra's additional timestamped run tree by
    # default while still permitting explicit overrides from the command line.
    default_overrides = {
        "hydra.run.dir": "hydra.run.dir=.",
        "hydra.output_subdir": "hydra.output_subdir=null",
        "hydra/job_logging": "hydra/job_logging=disabled",
    }
    supplied_arguments = tuple(sys.argv[1:])
    inserted_overrides = [
        override
        for prefix, override in default_overrides.items()
        if not any(argument.startswith(prefix) for argument in supplied_arguments)
    ]
    sys.argv[1:1] = inserted_overrides

    @hydra.main(
        version_base="1.3",
        config_path=None,
        config_name="config",
    )
    def hydra_main(config: DictConfig) -> None:
        main(OmegaConf.to_object(config))

    hydra_main()

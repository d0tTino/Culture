from __future__ import annotations

import warnings

from src.sim.kernel.simulation_kernel import SimulationKernel as _AuthoritativeSimulationKernel

warnings.warn(
    "src.sim.simulation_kernel is a deprecated adapter; import "
    "src.sim.kernel.simulation_kernel.SimulationKernel instead.",
    DeprecationWarning,
    stacklevel=2,
)

SimulationKernel = _AuthoritativeSimulationKernel

__all__ = ["SimulationKernel"]

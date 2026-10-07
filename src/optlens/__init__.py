"""optlens — solver-agnostic diagnostics for LP/MILP models."""
from importlib.metadata import PackageNotFoundError as _NotInstalled
from importlib.metadata import version as _version

try:
    __version__ = _version("optlens")
except _NotInstalled:  # run from a source checkout without installing
    __version__ = "0.0.1"
from .backends import (BACKENDS, IIS, race, GurobiBackend, HiGHSBackend, IISNotSupported, LicenseLimit, SCIPBackend,
                       SolverUnusable, SolveResult, StartingPlan, StepTimeUsed, gurobi_problem, gurobi_usable,
                       gurobipy_version, starting_plan)
from .diagnose import (apply_relaxation, check_iis, default_backend, deletion_filter_iis, elastic_model,
                       family_first_iis, fix_conflicts, feas_relax, fix_menu, get_iis, linking_families, lp_relaxation_iis,
                       relaxed_bounds)
from .loaders import from_object, from_pulp, from_pyomo, load_script
from .model import INF, ModelData, UnsupportedModel, from_gurobipy, load

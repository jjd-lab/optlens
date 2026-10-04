"""optlens — solver-agnostic diagnostics for LP/MILP models (Phase 0 minimal engine)."""
from .backends import BACKENDS, IIS, race, GurobiBackend, HiGHSBackend, IISNotSupported, LicenseLimit, SCIPBackend, SolveResult
from .diagnose import (apply_relaxation, check_iis, default_backend, deletion_filter_iis, elastic_model,
                       family_first_iis, fix_conflicts, feas_relax, fix_menu, get_iis, linking_families, lp_relaxation_iis,
                       relaxed_bounds)
from .loaders import from_object, from_pulp, from_pyomo, load_script
from .model import INF, ModelData, UnsupportedModel, from_gurobipy, load

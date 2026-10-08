"""Section 7 items 1 and 2: variant declarations and the joint selection planner.

Not etx.passes.plan.Plan: this Plan is a selection (one variant per phase plus the budget).
Contract: docs/SEC7-interface-fix.md (fix) and docs/SEC7-interface-grok.md (this package).
"""
from .variants import (Budget, Resources, Variant, Structure, REGISTRY, register, variants_for, phase_shapes,
                       default_structure, load_cfg, GEMV_PHASES)
from .planner import Plan, plan, NoLegalVariant, DEFAULT_BUDGETS, BUDGET_LEVELS

__all__ = ["Budget", "Resources", "Variant", "Structure", "REGISTRY", "register", "variants_for", "phase_shapes",
           "default_structure", "load_cfg", "GEMV_PHASES", "Plan", "plan", "NoLegalVariant", "DEFAULT_BUDGETS", "BUDGET_LEVELS"]

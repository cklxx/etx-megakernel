"""Descriptor allocation contracts; values include actual local G13 compiler output."""
import pytest
from examples.experiment_program.native_resources import register_allocation


def test_unified_descriptor_covers_accumulators_once_and_callee_reserve():
    # G04: metadata254=248 VGPR+6 AGPR; G13: metadata96, next-free97.
    assert register_allocation(254,254,31)==(256,0)
    assert register_allocation(96,97,12)==(104,1)


def test_native_register_descriptor_mismatch_stays_closed():
    with pytest.raises(ValueError):register_allocation(98,97,12)
    with pytest.raises(ValueError):register_allocation(96,97,11)

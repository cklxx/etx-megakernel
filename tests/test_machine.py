import pytest

from etx.ir import Scope
from etx.machine import list_archs, load_machine


@pytest.mark.parametrize("arch", list_archs())
def test_every_arch_loads_and_covers_scopes(arch):
    m = load_machine(arch)
    for s in (Scope.DOMAIN, Scope.DEVICE, Scope.SYSTEM):
        v = m.vis(s)
        assert "{ptr}" in v.poll and "{ptr}" in v.arrive
    assert m.total_cus() > 0
    assert m.t_sync_us(Scope.DEVICE) > 0


def test_chiplet_tree_facts():
    mi300 = load_machine("gfx942")
    assert mi300.num_domains == 8 and mi300.cus_per_domain() == 38 and mi300.has_domain_level
    assert mi300.memory_for(Scope.SYSTEM) == "fine_grained"
    assert mi300.effective_scope(Scope.DOMAIN) == Scope.DOMAIN
    mi250 = load_machine("gfx90a")
    assert mi250.devices_per_package == 2 and not mi250.has_domain_level
    assert mi250.effective_scope(Scope.DOMAIN) == Scope.DEVICE
    assert mi250.supports("fp_atomic_over_fabric") is False
    h100 = load_machine("sm_90")
    assert h100.effective_scope(Scope.DOMAIN) == Scope.DEVICE and h100.supports("multicast_reduce")


def test_costs_are_data_not_code():
    m = load_machine("gfx942")
    assert m.t_sync_us(Scope.DOMAIN) < m.t_sync_us(Scope.DEVICE)

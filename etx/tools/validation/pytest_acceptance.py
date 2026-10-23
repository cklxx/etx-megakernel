"""Pytest plugin that rejects deselection and incomplete execution for acceptance."""
import json
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption('--acceptance-inventory', required=True)


def pytest_configure(config):
    config._etx_inventory = {'collected': [], 'selected': [], 'deselected': [], 'executed': []}


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_collection_modifyitems(session, config, items):
    config._etx_inventory['collected'] = [item.nodeid for item in items]
    yield
    config._etx_inventory['selected'] = [item.nodeid for item in items]


def pytest_deselected(items):
    if items:
        items[0].config._etx_inventory['deselected'].extend(item.nodeid for item in items)


def pytest_runtest_logfinish(nodeid, location):
    # Session state is registered by pytest_sessionstart; each plugin invocation
    # runs in its own child interpreter, never in the acceptance controller.
    _session.config._etx_inventory['executed'].append(nodeid)


def pytest_sessionstart(session):
    global _session
    _session = session


def pytest_sessionfinish(session, exitstatus):
    report = session.config._etx_inventory
    collected, selected, executed = (report[k] for k in ('collected', 'selected', 'executed'))
    report['complete'] = bool(collected) and not report['deselected'] and (
        len(collected) == len(set(collected)) and sorted(collected) == sorted(selected) == sorted(executed))
    Path(session.config.getoption('--acceptance-inventory')).write_text(json.dumps(report, indent=2) + '\n')
    if not report['complete'] and session.exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED

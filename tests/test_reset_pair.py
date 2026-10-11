import copy
import pytest
from examples.vllm_llm.reset_pair import check_pair


def identities():
    b = dict(overrides={'ETX_DEV_RESET': '0', 'ETX_VL_TOUCH': '1'}, hipcc_flags='same',
             files={'imports': 'i', 'import_metadata': 'm', 'device_source/dsl_tiles.hip': 'd', 'device_source/imp_tiles.hip': 'a'})
    c = copy.deepcopy(b); c['overrides']['ETX_DEV_RESET'] = '1'
    p = dict(domains=8, workers_per_domain=38, events=[], ev_inst=[], task_in=[], descs=[])
    return b, c, p


def test_reset_pair_only_registered_switch():
    b, c, p = identities(); check_pair(b, c, p, p)
    c['overrides']['ETX_VL_TOUCH'] = '0'
    with pytest.raises(ValueError, match='flags other'): check_pair(b, c, p, p)


def test_reset_pair_refuses_math_and_schedule_changes():
    b, c, p = identities(); c['files']['imports'] = 'new'
    with pytest.raises(ValueError, match='body/import'): check_pair(b, c, p, p)
    b, c, p = identities(); q = dict(p, workers_per_domain=37)
    with pytest.raises(ValueError, match='schedule'): check_pair(b, c, p, q)

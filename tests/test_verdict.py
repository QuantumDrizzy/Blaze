"""ADR-0005 gate: one test, three answers.

Fixtures are the published ones. If an arm misses, the test fails.
The fixture is not moved in the same change.
"""

import pytest

pytest.importorskip("quimb")

from blaze import Kind, verdict
from blaze.examples.substrate_quantum_state import haar_random_state, tfim_groundstate

REL_TOL = 1e-6


def test_one_call_three_answers():
    paramagnet = tfim_groundstate(16, 3.0).reshape((2,) * 16)
    compressed = verdict(paramagnet, rel_tol=REL_TOL)
    assert compressed.kind is Kind.COMPRESSED
    assert compressed.tt is not None
    assert compressed.measured_rel_error is not None
    assert compressed.measured_rel_error <= REL_TOL
    assert compressed.absolute_bound >= 0.0
    assert compressed.ratio > 1.0

    haar = haar_random_state(16, seed=0).reshape((2,) * 16)
    declined = verdict(haar, rel_tol=REL_TOL)
    assert declined.kind is Kind.DECLINED
    assert declined.tt is None
    assert declined.ratio < 1.0
    assert declined.nparams > 0

    critical = tfim_groundstate(8, 1.0).reshape((2,) * 8)
    undecided = verdict(critical, rel_tol=REL_TOL, max_rank=2)
    assert undecided.kind is Kind.UNDECIDED
    assert undecided.tt is None
    assert undecided.cap_cut is not None
    assert undecided.cap_rank == 2

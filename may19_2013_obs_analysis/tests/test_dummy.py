import pytest
from may19_2013_obs_analysis.dummy import dummy_foo


def test_dummy():
    assert dummy_foo(4) == (4 + 4)

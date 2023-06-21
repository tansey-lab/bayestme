import bayestme.common


def test_safe_seed():
    rng = bayestme.common.create_rng(0)
    assert rng.__getstate__()["state"]["state"] != 0

    rng2 = bayestme.common.create_rng(0)
    assert rng.__getstate__()["state"]["state"] == rng2.__getstate__()["state"]["state"]

    rng3 = bayestme.common.create_rng(None)

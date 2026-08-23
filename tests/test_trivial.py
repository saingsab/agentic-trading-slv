"""Phase 0 sanity check: confirms pytest is wired up and can import slv."""
import slv


def test_pytest_runs():
    assert 1 + 1 == 2


def test_slv_package_importable():
    assert slv.__name__ == "slv"

import numpy as np
import pytest

from qfed.data import load_data, partition


@pytest.fixture(scope="module")
def ds():
    return load_data(seed=0)


def test_shapes_and_range(ds):
    assert ds.X_train.shape == (455, 4) and ds.X_test.shape == (114, 4)
    # scaler is fitted on train only, so train lies exactly in [0, pi]
    assert ds.X_train.min() >= 0 and ds.X_train.max() <= np.pi + 1e-5
    assert int(ds.y_train.sum() + ds.y_test.sum()) == 212  # malignant = positive class


def test_split_is_deterministic():
    a, b = load_data(seed=3), load_data(seed=3)
    np.testing.assert_array_equal(a.X_test, b.X_test)


@pytest.mark.parametrize("scheme", ["iid", "dirichlet", "feature_skew"])
def test_partition_disjoint_and_complete(ds, scheme):
    shards = partition(ds.X_train, ds.y_train, 3, scheme=scheme, alpha=0.5, seed=1)
    allidx = np.concatenate(shards)
    assert len(shards) == 3
    assert len(allidx) == len(ds.y_train) == len(np.unique(allidx))
    assert min(len(s) for s in shards) >= 10


def test_iid_keeps_class_balance(ds):
    shards = partition(ds.X_train, ds.y_train, 3, scheme="iid", seed=1)
    overall = ds.y_train.mean()
    for s in shards:
        assert abs(ds.y_train[s].mean() - overall) < 0.03

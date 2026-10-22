import torch

from qfed.fedavg import fedavg
from qfed.models import LogRegClassifier


def test_identical_states_are_unchanged():
    m = LogRegClassifier(4)
    out = fedavg([m.state_dict(), m.state_dict()], [10, 30])
    for k, v in m.state_dict().items():
        torch.testing.assert_close(out[k], v)


def test_weighted_average():
    a, b = LogRegClassifier(4), LogRegClassifier(4)
    with torch.no_grad():
        a.linear.weight.fill_(0.0)
        b.linear.weight.fill_(4.0)
    out = fedavg([a.state_dict(), b.state_dict()], [1, 3])
    torch.testing.assert_close(out["linear.weight"], torch.full((1, 4), 3.0))

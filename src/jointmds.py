import numpy as np
import torch

from src.JointMDS.joint_mds import JointMDS as _JointMDS


class JointMDS(_JointMDS):
    """JointMDS wrapper with the same alignment API as FoSTA and MALI."""

    def __init__(self, random_state=None, **kwargs):
        kwargs.setdefault("max_iter", 50)
        super().__init__(**kwargs)
        self.random_state = random_state
        self.embedding_ = None
        self.T = None

    def fit(self, x_a, x_b, y_a=None, y_b=None):
        del y_a, y_b  # JointMDS is unsupervised.
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        result = _JointMDS.fit_transform(self, x_a, x_b)
        embedding_a, embedding_b, self.T = result[:3]
        self.embedding_ = np.vstack(
            [
                embedding_a.detach().cpu().numpy(),
                embedding_b.detach().cpu().numpy(),
            ]
        )
        return self

    def fit_transform(self, x_a, x_b, y_a=None, y_b=None):
        return self.fit(x_a, x_b, y_a, y_b).embedding_

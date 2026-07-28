from src.pamona import Pamona


class UPamona(Pamona):
    """Unsupervised Pamona wrapper with the label-prior strength fixed to zero."""

    def __init__(self, **kwargs):
        kwargs.pop("gamma", None)
        super().__init__(gamma=0, **kwargs)

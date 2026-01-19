from src.kema import KEMA

class KEMArbf(KEMA):
    def __init__(self, **kwargs):
        super().__init__(kernel='rbf', **kwargs)

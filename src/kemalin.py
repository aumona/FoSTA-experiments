from src.kema import KEMA

class KEMAlin(KEMA):
    def __init__(self, **kwargs):
        super().__init__(kernel='lin', **kwargs)

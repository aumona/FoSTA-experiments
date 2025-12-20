from src.dta import DTA

class MALI(DTA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def fit(self, x_a, x_b, y_a, y_b):
        super().fit(x_a, x_b, labels1=y_a, labels2=y_b)
    
    def fit_transform(self, x_a, x_b, y_a, y_b):
        self.fit(x_a, x_b, y_a, y_b)
        return self.embed()
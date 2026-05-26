class ActionClassifier:
    def predict(self, sequence):
        raise NotImplementedError


class MockActionClassifier(ActionClassifier):
    def __init__(self, default_label="Fight", score=0.85):
        self.default_label = default_label
        self.score = score

    def predict(self, sequence):
        # TODO: replace with a PyTorch video/action-recognition model.
        return {"label": self.default_label, "score": float(self.score)}

class Model():

    def __init__(self) -> None:
        super().__init__()
        self.X, self.EPOCHS = [], 100
        self.mu, self.sigma, self.pi = [], [], []

    def compile(self, optimizer, loss):
        pass

    def fit(self, epochs, batch):
        pass

    def predict(self):
        pass

    def cost(self):
        pass

    def loss(self):
        pass

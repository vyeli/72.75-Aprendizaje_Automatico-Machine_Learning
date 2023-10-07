import numpy as np

class LinealKernel:
    def __init__(self):
        pass

    def __call__(self, x, support_vectors):
        return np.dot(x, support_vectors.T)

class PolinomialKernel(LinealKernel):
    def __init__(self, d=2):
        self.d = d

    def __call__(self, x, support_vectors):
        return (1 + super().__call__(x, support_vectors))**self.d

class RadialKernel:
    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, x, support_vectors):
        return np.exp(-self.sigma * np.sum([(x - support_vector)**2 for support_vector in support_vectors]))


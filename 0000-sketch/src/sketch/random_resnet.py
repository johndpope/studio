import numpy as np

def relu(x):
  return np.maximum(x, 0)

def softmax(x):
  return np.exp(x)/sum(np.exp(x))

class RandomResNet:
  map_fst: np.ndarray
  map_snd: np.ndarray
  add_fst: np.ndarray
  add_snd: np.ndarray

  def __init__(self, dim: int):
    rng = np.random.default_rng()
    self.map_fst = rng.standard_normal((dim, dim))
    self.map_snd = rng.standard_normal((dim, dim))
    self.add_fst = rng.standard_normal(dim)
    self.add_snd = rng.standard_normal(dim)

  def eval(self, value):
    # A two layer residual network.
    residual = np.dot(self.map_fst, value)
    residual = residual+self.add_fst
    residual = relu(residual)
    value = value+residual
    residual = np.dot(self.map_snd, value)
    residual = residual+self.add_snd    
    residual = relu(residual)
    value = value+residual
    return value

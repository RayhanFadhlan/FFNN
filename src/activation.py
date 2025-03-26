import numpy as np
from typing import List, Union, Callable, Any, Optional, TypeVar
from .autodiff import Value

T = TypeVar('T', Value, List[Value])

class Activation:
    """Base class for activation functions"""
    def __init__(self):
        pass
    
    def forward(self, x: List[Value]) -> List[Value]:
        pass
    
    def __call__(self, x: List[Value]) -> List[Value]:
        return self.forward(x)

class Linear(Activation):
    def forward(self, x: List[Value]) -> List[Value]:

        return x

class ReLU(Activation):
    def forward(self, x: List[Value]) -> List[Value]:
        return [xi.relu() for xi in x]

class Sigmoid(Activation):
    def forward(self, x: List[Value]) -> List[Value]:
        return [xi.sigmoid() for xi in x]

class Tanh(Activation):
    def forward(self, x: List[Value]) -> List[Value]:
        return [xi.tanh() for xi in x]

class Softmax(Activation):
    def forward(self, x: List[Value]) -> List[Value]:
        exps = np.array([xi.exp().data for xi in x])
        sum_exps = np.sum(exps)
        softmax_values = exps / sum_exps
        return [Value(s, (xi,), 'softmax') for s, xi in zip(softmax_values, x)]

def get_activation(name: str) -> Activation:
    activations = {
        'linear': Linear,
        'relu': ReLU,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'softmax': Softmax
    }
    return activations[name.lower()]()
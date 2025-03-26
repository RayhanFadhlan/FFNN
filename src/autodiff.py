import numpy as np
from typing import Optional, Dict, Any, Tuple, Union

class Value:
    """Stores value and gradient for automatic differentiation"""
    def __init__(self, data: np.ndarray, _children: Tuple['Value', ...] = (), _op: str = ''):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data, dtype=np.float32)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: 'Value') -> 'Value':
        other_data = other.data if isinstance(other, Value) else other
        out = Value(self.data + other_data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            if isinstance(other, Value):
                other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other: 'Value') -> 'Value':
        other_data = other.data if isinstance(other, Value) else other
        out = Value(self.data * other_data, (self, other), '*')
        
        def _backward():
            self.grad += other_data * out.grad
            if isinstance(other, Value):
                other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other: Union[int, float]) -> 'Value':
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self) -> 'Value':
        return self * -1

    def __sub__(self, other: Union['Value', float]) -> 'Value':
        return self + (-other)

    def __truediv__(self, other: Union['Value', float]) -> 'Value':
        return self * (other ** -1)

    def exp(self) -> 'Value':
        out = Value(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self) -> 'Value':
        out = Value(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self) -> 'Value':
        t = np.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def relu(self) -> 'Value':
        out = Value(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (1 if self.data > 0 else 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self) -> 'Value':
        sig = 1 / (1 + np.exp(-self.data))
        out = Value(sig, (self,), 'sigmoid')

        def _backward():
            self.grad += sig * (1 - sig) * out.grad
        out._backward = _backward
        return out


    def backward(self) -> None:
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data, dtype=np.float32)
        for node in reversed(topo):
            node._backward()


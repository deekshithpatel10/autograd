import numpy as np
from tensor import Tensor

def exp(t: Tensor) -> Tensor:
    out_data = np.exp(t.data)
    out = Tensor(out_data, requires_grad=t.requires_grad, _parents=(t,), _op='exp')

    def _backward():
        if t.requires_grad:
            grad = out.data * out.grad  # d/dx e^x = e^x
            t.grad = t.grad + grad if t.grad is not None else grad

    out._backward = _backward
    return out


def tanh(t: Tensor) -> Tensor:
    out_data = np.tanh(t.data)
    out = Tensor(out_data, requires_grad=t.requires_grad, _parents=(t,), _op='tanh')

    def _backward():
        if t.requires_grad:
            grad = (1 - out.data ** 2) * out.grad  # d/dx tanh = 1 - tanh^2
            t.grad = t.grad + grad if t.grad is not None else grad

    out._backward = _backward
    return out


def relu(t: Tensor) -> Tensor:
    out_data = np.maximum(0, t.data)
    out = Tensor(out_data, requires_grad=t.requires_grad, _parents=(t,), _op='relu')

    def _backward():
        if t.requires_grad:
            grad = (t.data > 0).astype(np.float64) * out.grad  # gradient is 1 for positive inputs, 0 otherwise
            t.grad = t.grad + grad if t.grad is not None else grad

    out._backward = _backward
    return out

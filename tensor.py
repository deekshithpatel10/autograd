import numpy as np 

class Tensor: 
   def __init__(self, data, requires_grad=False, _children=(), _op=''):
      self.data = np.array(data, dtype=np.float64)
      self.requires_grad = requires_grad
      self.grad = 0.0

      # graph related attributes 
      self._backward = lambda : None 
      self._prev = set(_children)
      self._op = _op

   def __repr__(self):
      return f"Tensor(data={self.data}, grad={self.grad})"

   def __add__(self, other): 
      other = other if isinstance(other, Tensor) else Tensor(other)
      out = Tensor(self.data + other.data, True, (self, other), '+')

      def _backward(): 
         self.grad += out.grad
         other.grad += out.grad 
      out._backward = _backward

      return out 
   
   def __mul__(self, other): 
      other = other if isinstance(other, Tensor) else Tensor(other)
      out = Tensor(self.data * other.data, True, (self, other), '*')

      def _backward(): 
         self.grad += other.data * out.grad 
         other.grad += self.data * out.grad 
      out._backward = _backward

      return out 
   
   def backward(self): 
      if not self.requires_grad:
        raise RuntimeError("Cannot call backward on a tensor that does not require gradients.")
      
      # topo sort 
      topo = []
      visited = set()

      def build_topo(v): 
         if v not in visited: 
            visited.add(v)
            for child in v._prev: 
               build_topo(child)
            topo.append(v)

      build_topo(self)

      self.grad = 1.0   # ∂output/∂output 
      for node in reversed(topo): 
         node._backward()

if __name__ == "__main__":
   a = Tensor(2.0)
   b = Tensor(3.0)
   c = a * b + b + 1
   c.backward()

   print("a:", a)  # Should show grad = 3.0
   print("b:", b)  # Should show grad = 3.0 (from mult) + 1.0 (from add)
   print("c:", c)  # Grad = 1.0 (seed)



import numpy as np 

class Tensor: 
   def __init__(self, data, requires_grad=False, _parents=(), _op=''): 
      self.data = np.array(data, dtype=np.float64)
      self.requires_grad = requires_grad 
      self.grad = None 

      self._backward = lambda : None      # gradient function
      self._prev = set(_parents)         # tensors used to create this one
      self._op = _op                      # operation involved

   def __repr__(self):
      return f'Tensor(data={self.data}, grad={self.grad}, requires_grad={self.requires_grad})'

   def __add__(self, other): 
      other = other if isinstance(other, Tensor) else Tensor(other)
      out = Tensor(self.data+other.data, requires_grad=(self.requires_grad or other.requires_grad), _parents=(self, other), _op='+')

      def _backward():
         if self.requires_grad:
            self.grad = self.grad + out.grad if self.grad is not None else out.grad
         if other.requires_grad:
            other.grad = other.grad + out.grad if other.grad is not None else out.grad

      out._backward = _backward
      return out 

   def __mul__(self, other): 
      other = other if isinstance(other, Tensor) else Tensor(other)
      out = Tensor(self.data+other.data, requires_grad=(self.requires_grad or other.requires_grad), _parents=(self, other), _op='*')

      def _backward(): 
         if self.requires_grad:
            grad_self = other.data * out.grad
            self.grad = self.grad + grad_self if self.grad is not None else grad_self
         if other.requires_grad:
            grad_other = self.data * out.grad
            other.grad = other.grad + grad_other if other.grad is not None else grad_other
      
      out._backward = _backward

      return out 

   def __neg__(self): return self * -1

   def __sub__(self, other): return self + (-other)

   def __pow__(self, power):
      assert isinstance(power, (int, float))
      out = Tensor(self.data ** power, requires_grad=self.requires_grad, _parents=(self,), _op=f'**{power}')

      def _backward():
         if self.requires_grad:
            grad = (power * self.data ** (power - 1)) * out.grad
            self.grad = self.grad + grad if self.grad is not None else grad

      out._backward = _backward
      return out


   def backward(self): 
      if self.grad is None: 
         self.grad = np.ones_like(self.data)

      topo = []
      visited = set()
      def build_topo(tensor):
         if tensor not in visited: 
            visited.add(tensor)
            for parent in tensor._prev: 
               build_topo(parent)
            topo.append(tensor)
      
      build_topo(self)
   
      for tensor in reversed(topo): 
         tensor._backward()

   def zero_grad(self): 
      if self.requires_grad: 
         self.grad = 0.0
   
   @property 
   def shape(self): return self.data.shape 

   #right hand methods 
   def __radd__(self, other): return self + other
   def __rmul__(self, other): return self * other 
   def __rsub__(self, other): return (-self) + other 

if __name__=="__main__": 
   a = Tensor(4, requires_grad=True)
   b = Tensor(6, requires_grad=True)
   c =  a**2 + b
   print(c)
   c.backward()
   print(a)
   print(b)



      
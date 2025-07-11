from graphviz import Digraph
import numpy as np 

def draw_computational_graph(output_tensor):
   dot = Digraph(format='png', graph_attr={'rankdir': 'LR'})
   seen = set()

   def tensor_label(t):
      label = ""
      if t.name:
         label += f"{t.name}\n"
      if t.data.ndim <= 2 and t.data.size <= 9:
         label += np.array2string(t.data, precision=2, floatmode='fixed')
      return label or "Tensor"

   def build(t):
      if t not in seen:
         seen.add(t)

         # Tensor node
         dot.node(str(id(t)), label=tensor_label(t), shape='ellipse')

         for parent in t._prev:
            build(parent)
            if t._op:
               op_node_id = f"{id(t)}_{t._op}"
               dot.node(op_node_id, label=t._op, shape='box')
               dot.edge(str(id(parent)), op_node_id)
               dot.edge(op_node_id, str(id(t)))

   build(output_tensor)
   return dot


# autograd
Anyone who has taken an introductory machine learning course knows how important gradients are for training models.
But calculating gradients manually is tedious and error-prone.

PyTorch uses automatic differentiation to compute gradients efficiently using backpropagation.
This repository is a small-scale attempt to mimic PyTorch’s Tensor class and its autograd functionality from scratch using only NumPy.

The core idea is to build a computation graph as operations are performed on tensors.
Each tensor keeps track of how it was created, and when .backward() is called, it walks the graph in reverse to compute gradients.

Supported operations include addition, multiplication, subtraction, and power. The implementation works with both scalars and matrices.

There’s also a small utility that lets you visualize the computation graph using Graphviz. It helps in understanding how data and gradients flow through the graph.

A demo notebook is included with examples showing how this system works in practice.

This is a learning-focused project and not meant for production. But it builds a strong foundation for understanding what happens inside modern ML libraries.
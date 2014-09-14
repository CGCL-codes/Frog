Feynman
=======

Optimization of Asynchronous Graph Processing on GPU with Hybrid Coloring Model

Modern GPUs have been widely used to accelerate the graph pro-
cessing for complicated computational problems regarding graph
theory. Several newly proposed graph computing frameworks for
clusters and multi-core servers adopt the asynchronous comput-
ing model to accelerate the convergence of many iterative graph
algorithms. Unfortunately, the consistent asynchronous computing
requires locking or the atomic operation, which will result in sig-
nificant penalties when implemented on GPUs. In order to avoid
the significant runtime overhead of locking operation on GPU and
guarantee the correctness/consistency of the parallel processing,
many solutions adopt coloring algorithms that can separate the
vertices with potential updating conflicts. Common coloring algo-
rithms, however, may suffer from low parallel degrees because of
a large number of colors generally required for processing a large-
size (a.k.a., large-scale) graph with billions of vertices.

We propose a light-weight asynchronous processing framework
called Feynman with a hybrid coloring model. The fundamental
idea is based on Pareto principle (or 80-20 rule) about coloring
algorithms as we observed through masses of real graph coloring
cases. We find that majority of vertices (about 80%) are colored
with only a few colors, such that they can be read and updated
in a very high parallel degree without violating the sequential
consistency. Accordingly, our solution will separate the processing
of the vertices based on the distribution of colors. In this work, we
mainly answer the two questions: (1) how to partition the vertices
in a super-large graph with maximized parallel processing degree,
and (2) how to reduce the overhead of data transfers on PCI-e
while processing each partition. Experiments based on real-world
data show that our asynchronous GPU graph processing engine
outperforms other state-of-the-art approaches by the speedup over
4.3X-26.2X.

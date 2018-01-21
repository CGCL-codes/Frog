Frog
=======

Asynchronous Graph Processing on GPU with Hybrid Coloring Model

GPUs have been increasingly used to accelerate graph processing for complicated computational problems regarding graph theory. Many parallel graph algorithms adopt the asynchronous computing model to accelerate the iterative convergence. Unfortunately, the consistent asynchronous computing requires locking or atomic operations, leading to signiﬁcant penalties/overheads when implemented on GPUs. To this end, coloring algorithm is adopted to separate the vertices with potential updating conﬂicts, guaranteeing the consistency/correctness of the parallel processing. Common coloring algorithms, however, may suffer from low parallelism because of a large number of colors generally required for processing a large-scale graph with billions of vertices.

We propose a light-weight asynchronous processing framework called Frog with a hybrid coloring model. The fundamental idea is based on Pareto principle (or 80-20 rule) about coloring algorithms as we observed through masses of real graph coloring cases. We ﬁnd that majority of vertices (about 80%) are colored with only a few colors, such that they can be read and updated in a very high degree of parallelism without violating the sequential consistency. Accordingly, our solution will separate the processing of the vertices based on the distribution of colors. In this work, we mainly answer the four questions: (1) how to partition the vertices in a sparse graph with maximized parallelism, (2) how to process large-scale graphs which are out of GPU memory, (3) how to reduce the overhead of data transfers on PCIe while processing each partition, and (4) how to customize the data structure that is particularly suitable for GPU-based graph processing. Experiments based on real-world data show that our asynchronous GPU graph processing engine outperforms other state-of-the-art approaches by 2.23X–55.15X.

Project details can be found at http://grid.hust.edu.cn/xhshi/projects/frog.html .
Publications about Frog:

[1] Xuanhua Shi, Junling Liang, Sheng Di, Bingsheng He, Hai Jin, Lu Lu, Zhixiang Wang, Xuan Luo, Jianlong Zhong, "Optimization of Asynchronous Graph Processing on GPU with Hybrid Coloring Model". in Proceedings of the ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '15), poster, San Francisco, USA, 2015.

[2] Xuanhua Shi, Xuan Luo, Junling Liang, Peng Zhao, Sheng Di, Bingsheng He, Hai Jin, "Frog: Asynchronous Graph Processing on GPU with Hybrid Coloring Model", IEEE Transactions on Knowledge and Data Engineering, 30 (1): 29-42, 2018.

[3] Xuanhua Shi, Zhigao Zheng, Yongluan Zhou, Hai Jin, Ligang He, Bo Liu, Qiang-Sheng Hua, "Graph Processing on GPUs: A Survey", ACM Computing Surveys, 50, 6, Article 81, January 2018, 35 pages.

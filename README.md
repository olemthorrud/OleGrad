Implementation of a simple neural network framework in C++, inspired by [Andrej Karpathy's YouTube video](https://www.youtube.com/watch?v=VMj-3S1tku0). The source code contains two versions of the same core functionality, one clean and one where 
approximately every line of code and general concept is explained carefully. 

In the end, the project was tested on the MNIST dataset, where a simple NN built on this framework got 82% as the final accuracy. In theory, the performance should match that of any standard NN framework (e.g. PyTorch). In practice, however, the absence of optimizations makes meaningful testing impractical, since a single epoch takes over an hour on CPU. CUDA support will have to wait for another day.


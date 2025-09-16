Implementation of a simple neural network framework in C++, inspired by [Andrej Karpathy's YouTube video](https://www.youtube.com/watch?v=VMj-3S1tku0). The source code contains two versions of the same core functionality, one clean and one where 
approximately every line of code and general concept is explained carefully. 

In the end, the project was tested on the MNIST dataset, where a simple NN built on this framework got 82% as the final accuracy. In theory, there is no reason why the performance should not be equal to what you would expect from any NN from an established vendor (e.g. PyTorch), 
in final result, but the lack of optimization techniques in my project that exists in theirs makes testing this pretty much impossible, as a single epoch takes more than an hour on a CPU (and implementing this in CUDA will have to be a project for another day).  


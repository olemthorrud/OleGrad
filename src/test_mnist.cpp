#include <iostream>
#include <stdexcept>
#include <string>
#include "mnist_io.hpp"

void quick_mnist_smoke_test() {
    const std::string base = "/Users/olethorrud/Documents/OleGrad/MNIST_dataset";
    try {
        const std::string img_path = base + "/train-images.idx3-ubyte";
        const std::string lbl_path = base + "/train-labels.idx1-ubyte";

        MnistImages X = load_mnist_images(img_path);
        MnistLabels Y = load_mnist_labels(lbl_path);

        std::cout << "Loaded images: " << X.count
                  << " (" << X.rows << "x" << X.cols << ")\n";
        std::cout << "Loaded labels: " << Y.y.size() << "\n";

        std::size_t i = 1234 % X.count;
        show_image(X, Y, i);
    } catch (const std::exception& e) {
        std::cerr << "[MNIST smoke test] Error: " << e.what() << "\n";
    }
}

#if defined(TEST_MNIST_STANDALONE)
int main() {
    quick_mnist_smoke_test();
    return 0;
}
#endif
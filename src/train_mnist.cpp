#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "olegrad.hpp"
#include "mnist_io.hpp"

static inline void render_progress(
    int ep,
    std::size_t step,
    std::size_t total_steps,
    double loss,
    double imgs_per_s,
    double eta_seconds
) {
    const int bar = 40;
    double frac = (total_steps == 0) ? 1.0
                                     : std::min(1.0, std::max(0.0, double(step) / double(total_steps)));
    int filled = int(std::round(frac * bar));

    std::cout << "\r"
              << "epoch " << ep << " ["
              << std::string(filled, '#')
              << std::string(bar - filled, '.')
              << "] "
              << std::setw(3) << int(std::round(frac * 100)) << "%  "
              << "loss " << std::fixed << std::setprecision(4) << loss
              << " | "   << std::setprecision(0) << imgs_per_s << " img/s"
              << " | ETA " << std::setprecision(0) << eta_seconds << "s"
              << std::flush;
}

static inline Value nll_loss(const std::vector<Value>& logits, uint8_t y) {
    double m = logits.empty() ? 0.0 : logits[0].data();
    for (const auto& v : logits) if (v.data() > m) m = v.data();

    std::vector<Value> shifted;
    shifted.reserve(logits.size());
    for (const auto& v : logits) shifted.push_back(v - m);

    Value sumexp(0.0);
    for (const auto& v : shifted) sumexp = sumexp + v.exp();

    Value logsumexp = sumexp.log();
    return logsumexp - shifted[std::size_t(y)];
}

static inline int argmax_logits(const std::vector<Value>& logits) {
    int best = 0;
    double bestv = logits[0].data();
    for (int i = 1; i < (int)logits.size(); ++i) {
        double v = logits[i].data();
        if (v > bestv) { bestv = v; best = i; }
    }
    return best;
}

static inline std::vector<Value> make_input(const double* img, std::size_t n) {
    std::vector<Value> x;
    x.reserve(n);
    for (std::size_t i = 0; i < n; ++i) x.emplace_back(img[i]);
    return x;
}

static double evaluate_accuracy(
    MLP& net,
    const MnistImages& X,
    const MnistLabels& Y,
    std::size_t max_samples = 0
) {
    std::size_t n = X.count;
    const std::size_t d = X.rows * X.cols;
    if (max_samples && max_samples < n) n = max_samples;

    std::size_t correct = 0;
    for (std::size_t i = 0; i < n; ++i) {
        auto x = make_input(image_ptr(X, i), d);
        auto logits = net(x);
        if (argmax_logits(logits) == int(Y.y[i])) ++correct;
    }
    return 100.0 * double(correct) / double(n);
}

int main() {
    try {
        MnistImages trX = load_mnist_images("train-images.idx3-ubyte");
        MnistLabels trY = load_mnist_labels("train-labels.idx1-ubyte");
        MnistImages teX = load_mnist_images("t10k-images.idx3-ubyte");
        MnistLabels teY = load_mnist_labels("t10k-labels.idx1-ubyte");

        std::cout << "train: " << trX.count << " images, "
                  << trX.rows << "x" << trX.cols << "\n";
        std::cout << "test : " << teX.count << " images, "
                  << teX.rows << "x" << teX.cols << "\n";

        const std::size_t D = trX.rows * trX.cols;
        MLP net(D, {128, 64, 10});
        auto params = net.parameters();

        const int epochs = 3;
        const int batch_size = 128;
        const double lr = 0.02;

        std::vector<std::size_t> idx(trX.count);
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937 rng(std::random_device{}());

        for (int ep = 1; ep <= epochs; ++ep) {
            const std::size_t total_steps = (trX.count + batch_size - 1) / batch_size;
            double ema_batch_seconds = 0.0;
            bool have_ema = false;

            auto t0 = std::chrono::high_resolution_clock::now();
            std::shuffle(idx.begin(), idx.end(), rng);

            double running_loss = 0.0;
            std::size_t steps = 0;

            for (std::size_t start = 0; start < trX.count; start += batch_size) {
                const std::size_t end = std::min<std::size_t>(start + batch_size, trX.count);
                const int bsz = int(end - start);

                auto bt0 = std::chrono::high_resolution_clock::now();

                net.zero_grad();

                double batch_loss_scalar = 0.0;

                for (std::size_t k = start; k < end; ++k) {
                    const std::size_t i = idx[k];
                    auto x = make_input(image_ptr(trX, i), D);
                    auto logits = net(x);
                    Value loss = nll_loss(logits, trY.y[i]);
                    batch_loss_scalar += loss.data();
                    loss.backward();
                }

                const double scale = lr / double(bsz);
                for (auto& p : params) {
                    p.data_ref() -= scale * p.grad();
                }

                running_loss += (batch_loss_scalar / double(bsz));
                ++steps;

                auto bt1 = std::chrono::high_resolution_clock::now();
                double bsecs = std::chrono::duration<double>(bt1 - bt0).count();
                if (bsecs <= 0.0) bsecs = 1e-9;

                double ips = double(bsz) / bsecs;

                if (!have_ema) {
                    ema_batch_seconds = bsecs;
                    have_ema = true;
                } else {
                    ema_batch_seconds = 0.9 * ema_batch_seconds + 0.1 * bsecs;
                }

                double remaining_s = ema_batch_seconds * double(total_steps - steps);

                render_progress(
                    ep,
                    steps,
                    total_steps,
                    (batch_loss_scalar / double(bsz)),
                    ips,
                    remaining_s
                );
            }

            std::cout << "\n";

            auto t1 = std::chrono::high_resolution_clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();

            double tr_acc = evaluate_accuracy(net, trX, trY, 5000);
            double te_acc = evaluate_accuracy(net, teX, teY, 10000);

            std::cout << "epoch " << ep
                      << " avg_loss " << (running_loss / double(steps))
                      << " | train " << tr_acc << "%"
                      << " | test "  << te_acc  << "%"
                      << " | time "  << secs    << "s\n";
        }

        std::cout << "final test accuracy: "
                  << evaluate_accuracy(net, teX, teY, 10000) << "%\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
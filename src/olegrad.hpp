#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

class Value {
    
    struct Node{
        double data;
        double grad;
        std::vector<Value> prev; 
        std::function<void()> backward; 
        std::string op; 
        Node(double d, std::vector<Value> p = {}, std::string o = {}): 
        data(d), grad(0.0), prev(std::move(p)), backward([]{}), op(std::move(o)) {}
    }; 

std::shared_ptr<Node> p_;

public:

Value() : p_(std::make_shared<Node>(0.0)) {}
explicit Value(double d) : p_(std::make_shared<Node>(d)) {}
Value(double d, std::vector<Value> parents, std::string o)
: p_(std::make_shared<Node>(d, std::move(parents), std::move(o))) {}

double data() const { return p_->data; }
double& grad()      { return p_->grad; }
double grad() const { return p_->grad; }
double& data_ref() { return p_->data; }

friend std::ostream& operator<<(std::ostream& os, const Value& v) {
    os << "Value(data=" << v.p_->data << ", grad=" << v.p_->grad << ")";
    return os;
  }

friend Value operator+(const Value& a, const Value& b) {
    Value out(a.data() + b.data(), {a, b}, "+"); 
    Node* pa = a.p_.get();
    Node* pb = b.p_.get();
    Node* po = out.p_.get();
    out.p_->backward = [pa, pb, po]() {
        pa->grad += po->grad;
        pb->grad += po->grad;
    };
    return out;
}

friend Value operator*(const Value& a, const Value& b) {
    Value out(a.data() * b.data(), {a, b}, "*");
    Node* pa = a.p_.get();
    Node* pb = b.p_.get();
    Node* po = out.p_.get();
    out.p_->backward = [pa, pb, po]() {
        pa->grad += pb->data * po->grad;
        pb->grad += pa->data * po->grad;
    };
    return out;
}

friend Value operator+(const Value& a, double b) { return a + Value(b); }
friend Value operator+(double a, const Value& b) { return Value(a) + b; }
friend Value operator*(const Value& a, double b) { return a * Value(b); }
friend Value operator*(double a, const Value& b) { return Value(a) * b; }

void backward() {
    std::vector<Value> topo;
    std::unordered_set<const Node*> visited;
    std::function<void(const Value&)> build = [&](const Value& v) {
        const Node* key = v.p_.get();
        if (!visited.count(key)) {
            visited.insert(key);
            for (const auto& parent : v.p_->prev) build(parent);
                topo.push_back(v);
        }
    };
    build(*this);
    p_->grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        it->p_->backward();
    }
}

Value operator-() const { return (*this) * Value(-1.0); }
friend Value operator-(const Value& a, const Value& b) { return a + (-b); }
friend Value operator-(const Value& a, double b) { return a - Value(b); }
friend Value operator-(double a, const Value& b) { return Value(a) - b; }

Value pow(double exp) const {
    Value out(std::pow(p_->data, exp), { *this }, std::string("**") + std::to_string(exp));
    Node* self = p_.get(); 
    Node* po   = out.p_.get();
    out.p_->backward = [self, po, exp]() {
        self->grad += exp * std::pow(self->data, exp - 1.0) * po->grad;
    };
    return out;
}

friend Value pow(const Value& base, double exp) { return base.pow(exp); }
friend Value operator/(const Value& a, const Value& b) { return a * b.pow(-1.0); }
friend Value operator/(const Value& a, double b) { return a / Value(b); }
friend Value operator/(double a, const Value& b) { return Value(a) / b; }

Value log() const {
  assert(p_->data > 0.0 && "log requires positive input");
  Value out(std::log(p_->data), { *this }, "log");
  Node* self = p_.get(); Node* po = out.p_.get();
  out.p_->backward = [self, po]() {
    self->grad += (1.0 / self->data) * po->grad;
  };
  return out;
}

Value exp() const {
  Value out(std::exp(p_->data), { *this }, "exp");
  Node* self = p_.get(); Node* po = out.p_.get();
  out.p_->backward = [self, po]() {
    self->grad += po->data * po->grad;
  };
  return out;
}

Value relu() const {
    double outd = p_->data < 0.0 ? 0.0 : p_->data;
    Value out(outd, { *this }, "ReLU");
    Node* self = p_.get(); 
    Node* po   = out.p_.get();
    bool pass = (outd > 0.0);
    out.p_->backward = [self, po, pass]() {
        self->grad += (pass ? 1.0 : 0.0) * po->grad;
    };
    return out;
}
};

class Module {
public:
    virtual ~Module() = default;
    virtual std::vector<Value> parameters() { return {}; }
    void zero_grad() {
        for (auto& p : parameters()) p.grad() = 0.0;
    }
};

class Neuron : public Module {
    std::vector<Value> w_;  
    Value b_{0.0};          
    bool nonlin_{true}; 

    static Value dot(const std::vector<Value>& a, const std::vector<Value>& b) {
        assert(a.size() == b.size());
        Value s(0.0);
        for (std::size_t i = 0; i < a.size(); ++i) s = s + a[i] * b[i];
    return s;
    }

    public:

    explicit Neuron(std::size_t nin, bool nonlin = true): w_(nin), b_(0.0), nonlin_(nonlin) {
        static thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto& wi : w_) wi = Value(dist(rng));
        b_ = Value(0.0);
    }

    Value operator()(const std::vector<Value>& x) const {
        Value act = dot(w_, x) + b_;
        return nonlin_ ? act.relu() : act;
    }

    std::vector<Value> parameters() override {
        std::vector<Value> ps = w_;
        ps.push_back(b_);
        return ps;
    }
};

class Layer : public Module {
    std::vector<Neuron> neurons_;

public:
    Layer(std::size_t nin, std::size_t nout, bool nonlin = true) {
        neurons_.reserve(nout);
        for (std::size_t i = 0; i < nout; ++i)
            neurons_.emplace_back(nin, nonlin);
    }

    std::vector<Value> operator()(const std::vector<Value>& x) const {
        std::vector<Value> out;
        out.reserve(neurons_.size());
        for (const auto& n : neurons_) out.push_back(n(x));
        return out;
    }

    std::vector<Value> parameters() override {
        std::vector<Value> ps;
        for (auto& n : neurons_) {
            auto np = n.parameters();
            ps.insert(ps.end(), np.begin(), np.end());
        }
        return ps;
    }
};

class MLP : public Module {
    std::vector<Layer> layers_;

public:
    explicit MLP(std::size_t nin, const std::vector<std::size_t>& nouts) {
        std::vector<std::size_t> sz; sz.reserve(nouts.size() + 1);
        sz.push_back(nin);
        for (auto k : nouts) sz.push_back(k);
        layers_.reserve(nouts.size());
        for (std::size_t i = 0; i < nouts.size(); ++i) {
            bool nonlin = (i + 1 != nouts.size());
            layers_.emplace_back(sz[i], sz[i+1], nonlin);
        }
    }

    std::vector<Value> operator()(std::vector<Value> x) const {
        for (const auto& L : layers_) x = L(x);
        return x;
    }

    std::vector<Value> parameters() override {
        std::vector<Value> ps;
        for (auto& L : layers_) {
            auto lp = L.parameters();
            ps.insert(ps.end(), lp.begin(), lp.end());
        }
    return ps;
    }
};

inline Value mse(const std::vector<Value>& yhat, const std::vector<Value>& y) {
    assert(yhat.size() == y.size());
    Value s(0.0);
    for (std::size_t i = 0; i < y.size(); ++i) {
        Value d = yhat[i] - y[i];
        s = s + d * d;
    }
    return s * Value(1.0 / static_cast<double>(y.size()));
}
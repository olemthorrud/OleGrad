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
    
    // Wrapping the actual data in a node-struct, so that a Value object does not hold the numbers itself.  
    
    // It, in stead, points via the shared-pointer to one of these node-objects, so that multiple Values can
    // share the same node. This is necessary because a hypothetical operation y = m + m, for example, is an operation
    // between three values, but two share a node as their data-container (m and m are different Value instances, but they represent the same
    // data through their shared relationship to the same node).
    // If this were not the case, then they would not be pointing to the same underlying Value data structure, and the gradient 
    // contributions would not accumulate into one place. 

    // Note that a node is just a scalar in a computation - sometimes it is the weight, sometimes it is the value stored
    // in a neuron, sometimes it is a bias. It is NOT the same thing as a neuron - we will make that later. It is a node@
    // in a computation graph.
    
    struct Node{
        // data is this nodes scalar 
        double data;

        // grad is the derivative of the final loss w.r.t this node's scalar (data)
        double grad;

        // prev is a list of which other values were used to compute this one
        // if c = a + b then c's prev contains {a,b}
        // this is how we can walk back through the computation graph
        std::vector<Value> prev; 

        // backward is the relevant backpropagation-rule for this node. It comes in three forms, based on what type of node this is: 
        // 1. A multiplication node (i.e. it is the result of the multiplication of two other nodes (weight * neuron-value)) 
        // 2. A summing node (i.e. it is the result of summing two other nodes (neuron-value + bias))
        // 3. An activation node (i.e. it is the result of applying a non-linear activation function to a single node)

        // The details are a bit complex, but in essence this contains: 
        // "Based on how a tiny nudge to my scalar affects L, how do I update how a tiny nudge to my parent's scalars affect L?"
        // (Of course, a child nodes' parents' gradient w.r.t L depends on the gradient of the child)  
        std::function<void()> backward; 

        // Just a label variable that expresses which operation created this node
        std::string op; 

        //  Constructor
            // parameters

                // double d - the data that we want this node to hold
                // std::vector<Value> p - a list of parents made optional by the "= {}" (not every node has parents)
                // std::string o - an operation label made optional by the "= {}"
        
            // constructing operations
                // data(d) just stores the argument d in the Node objects data variable
                // grad(0.0) just stores the initial value 0.0 in the Node objects grad variable
                // prev(std::move(p)) moves (not copies!) the passed-in vector of parent values (containing shared pointers to the relevant Node objects)
                    // into the Node objects prev vector
                // backward([]{}) defaults the backward operation to a no-operation lambda (that's what the syntax []{} means)
                    // it takes in nothing, does nothing, and returns nothing
                // op(std::move(o)) stores the operation label, taking ownership of the passed-in label rather than copying it
        Node(double d, std::vector<Value> p = {}, std::string o = {}): 
        data(d), grad(0.0), prev(std::move(p)), backward([]{}), op(std::move(o)) {}
    }; 

// p_ is the shared pointer to a specific Node object.
// An operaton in the computation graph is an operation between two **different** values, but it could potentially
// be an operation on two instances of the **same** node. y = m + m is an operation on two different instances of the object Value, 
// where each of these instances point to the same Node. Therefore, we want one Node instance to correspond to several Value instances.
std::shared_ptr<Node> p_;

public:

// The default Value constructor. All it does is allocate a fresh node on the heap with no parents, and make a shared pointer to it. 
// Copying a Value instance then creates another Value instance that corresponds to the same Node as the original one. 
// Represents a constant 0 in the computation grap (no parents, no grad, data = 0.0) - leaf node
Value() : p_(std::make_shared<Node>(0.0)) {}

// To construct a Value with a specific data d. Otherwise, the same as the above constructor.
// These are also leaf nodes (weights, inputs, biases) that do not need parents 
// The 'explicit' keyword is there to prevent silent double -> Value conversions. 
explicit Value(double d) : p_(std::make_shared<Node>(d)) {}

// This is to construct an internal node in the graph - i.e "I just performed some operation (e.g. y = a*b), and the result was y - create that node"
// It has parents so backprop knows where to send the gradients
Value(double d, std::vector<Value> parents, std::string o)
: p_(std::make_shared<Node>(d, std::move(parents), std::move(o))) {}

// Just get-functions to return the data and gradients of a Value
double data() const { return p_->data; } //returns the data as a double (the type) copy (read-only getter) - pass-by-value
double& grad()      { return p_->grad; } //returns a **reference** to the gradient, so that we can change it - pass-by-reference
double grad() const { return p_->grad; } //returns a copy of the gradient, not a reference (is run in stead of the above when the Value object is const)
double& data_ref() { return p_->data; } // to allow optimizer to update the data during training

// A simple overload to be able to print a Value-object
friend std::ostream& operator<<(std::ostream& os, const Value& v) {
    os << "Value(data=" << v.p_->data << ", grad=" << v.p_->grad << ")";
    return os;
  }


// Operations
    // Multiplications: 
        // When a node y is the result of a muliplication of nodes a and b (y = a * b), then:
            // The derivative of y w.r.t a.data is b.data.
            // The derivative of y w.r.t b.data is a.data.
        
    // Addition: 
        // When a node y is the result of an addition of nodes a and b (y = a + b), then:
            // The derivative of y w.r.t a.data is 1.0.
            // The derivative of y w.r.t b.data is 1.0.
    // In the base case, from which we can propagate, y = L (our loss function)


friend Value operator+(const Value& a, const Value& b) {
    // Consider the operation (out = a + b)

    // First, compute the new scalar for the node (the sum of the data from its parent-nodes)
    // And create the new result-node (called out)
    Value out(a.data() + b.data(), {a, b}, "+"); 

    // Capture raw pointers for the relevant nodes to avoid shared_ptr cycles
    Node* pa = a.p_.get();
    Node* pb = b.p_.get();
    Node* po = out.p_.get();

    // Definition of the backprop-rule for this new node: 
    // For addition, this just means multiplying the gradient of out w.r.t L by 1 (i.e do nothing) (see above)
    // and add it to both parents' gradients. 
    out.p_->backward = [pa, pb, po]() {
        pa->grad += po->grad;
        pb->grad += po->grad;
    };
    return out;
}

friend Value operator*(const Value& a, const Value& b) {
    // Consider the operation (out = a + b)

    // First, compute the new scalar for the node (the product of the data from its parent-nodes)
    // And create the new result-node (called out)
    Value out(a.data() * b.data(), {a, b}, "*");
    
    // Capture raw pointers for the relevant nodes to avoid shared_ptr cycles
    Node* pa = a.p_.get();
    Node* pb = b.p_.get();
    Node* po = out.p_.get();

    // Definition of the backprop-rule for this new node: 
    // For multiplication, this means: 
        // for a: multiplying the gradient of out w.r.t L by b.data and add it to both parents' gradients. (see above)
        // for b: multiplying the gradient of out w.r.t L by a.data and add it to both parents' gradients. (see above)
    out.p_->backward = [pa, pb, po]() {
        pa->grad += pb->data * po->grad;
        pb->grad += pa->data * po->grad;
    };
    return out;
}

// simple convenience overloads (scalars)
friend Value operator+(const Value& a, double b) { return a + Value(b); }
friend Value operator+(double a, const Value& b) { return Value(a) + b; }
friend Value operator*(const Value& a, double b) { return a * Value(b); }
friend Value operator*(double a, const Value& b) { return Value(a) * b; }


// The actual function performing the backpropagation by propagating backwards and adjusting the gradients of all
// nodes that lead to it. I.e. it is to be called on L, our final node (loss function). 
void backward() {
    // Build topological order (DAG) of the graph that ends at whatever node this function is called upon. 
    std::vector<Value> topo;

    // Excactly what it sounds like. Note that we are collecting not Values here, but their underlying Node-pointers. 
    // In this way, two different Value instances are correctly considered to correspond to the same underlying data structure. 
    std::unordered_set<const Node*> visited;

    // We build a "topo-tree" using DFS, where the parents come before the node, using a simple lambda-function.
    // The [&] simply means that all variables in the scope of backwards() are availabl to the lambda as references. 
    std::function<void(const Value&)> build = [&](const Value& v) {
        const Node* key = v.p_.get();
        if (!visited.count(key)) {
            visited.insert(key);
            for (const auto& parent : v.p_->prev) build(parent);
                topo.push_back(v);
        }
    };

    // Actually make topo around the node on which this function was called (L)
    build(*this);

    // The derivative of L w.r.t L is 1.0
    p_->grad = 1.0;

  // 3) run local chain-rule lambdas in reverse topological order
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        it->p_->backward();
    }
}

// Functionality to flip the sign of a Value's data (piggybacking on the already existing multiplication functionality)
Value operator-() const { return (*this) * Value(-1.0); }

// Functionality to subract one value from another (piggybacking on the already existing summing functionality)
friend Value operator-(const Value& a, const Value& b) { return a + (-b); }

// Convenience functions that allow you to: 
    // 1. Write v - 3.0 (subtract a double from a Value's data)
    // 2. Write 3.0 - v (subtract a Value's data from a double)
friend Value operator-(const Value& a, double b) { return a - Value(b); }
friend Value operator-(double a, const Value& b) { return Value(a) - b; }


// Exponential functionality
Value pow(double exp) const {
    // Define a new Value out, containing a pointer to a Node containig the data of p_ raised to the power of exp as its data
    Value out(std::pow(p_->data, exp), { *this }, std::string("**") + std::to_string(exp));
    
    // Capture the pointers (raw)
    Node* self = p_.get(); 
    Node* po   = out.p_.get();
    
    // Define the backpropagation rule for this case (high school maths)
    out.p_->backward = [self, po, exp]() {
        self->grad += exp * std::pow(self->data, exp - 1.0) * po->grad;
    };
    return out;
}

// Just a helper that wraps the above function
// Allows writing pow(v, 3.0) instead of v.pow(3.0)
friend Value pow(const Value& base, double exp) { return base.pow(exp); }

// The division operation reuses the exponential functionality
// Three cases, one for Value/Value, one for double/Value and one for Value/double
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

// Implementation of ReLu activation function. 
Value relu() const {
    // If p_->data less than zero, p_->data is set equal to zero
    // For computation of the forward pass
    double outd = p_->data < 0.0 ? 0.0 : p_->data;
    
    // Define the out node containing the result of this ReLu
        // The data it holds is the clamped value outd
        // Has only one parent - the node on which we perform ReLu
    Value out(outd, { *this }, "ReLU");
    
    // Capture the relevant pointers (raw)
    Node* self = p_.get(); 
    Node* po   = out.p_.get();

    // Did the ReLu simply pass through the value, or was it clamped to zero? 
    bool pass = (outd > 0.0);

    // Define the relevant backpropagation rule. For ReLu, this is: 
        // If data flowed through un-clamped, the gradient flows through unchanged during the backwards pass
        // If data was clamped to zero, no gradient flows through during the backwards pass
    out.p_->backward = [self, po, pass]() {
        self->grad += (pass ? 1.0 : 0.0) * po->grad;
    };
    return out;
}
};


// Base class for all "trainable things" - Neuron, Layer, etc. 

class Module {
public:
    // Use a default destructor, virtual so that the "lower-level" destructor also runs
    // in the case that you delete a derived object through a Module pointer
    virtual ~Module() = default;

    // Virtual function that can be overridden by derived objects - here, it just returns an empty vector
    virtual std::vector<Value> parameters() { return {}; }
    
    // Concrete function, not virtual - i.e., derived objects cannot override it. All instances get the same behaviour. 
    // Zeroes out the gradients after a pass, making the network ready for the next one
    void zero_grad() {
        for (auto& p : parameters()) p.grad() = 0.0;
    }
};

class Neuron : public Module {
    // The weights, one per input to this Neuron. A weight is an instance of Value. 
    std::vector<Value> w_;  
    
    // The bias, a single value, initialized to 0.0
    Value b_{0.0};          
    
    // Just a flag - if true, apply ReLu, if false then do not
    bool nonlin_{true}; 


    // Functionality for calculating \sum_{i = 0}^{I}(w_i * x_i) (i.e. updating the data of a neuron, the value it holds)
    static Value dot(const std::vector<Value>& a, const std::vector<Value>& b) {
        assert(a.size() == b.size());
        Value s(0.0);
        for (std::size_t i = 0; i < a.size(); ++i) s = s + a[i] * b[i];
    return s;
    }

    public:


    // Neuron constructor. Takes in nin (number of inputs to this neuron), and initializes w_
    // Also contains fast random number generating functionality to make sure we do not start with uniform weights. 
    explicit Neuron(std::size_t nin, bool nonlin = true): w_(nin), b_(0.0), nonlin_(nonlin) {
        static thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto& wi : w_) wi = Value(dist(rng));
        b_ = Value(0.0);
    }

    // Defines () so that the neuron is callable like a function
    // Takes in the activations of the previous layer, and computes the dot product using dot(), 
    // returning ReLu treated result if nonlin_ = true, else raw result. 
    Value operator()(const std::vector<Value>& x) const {
        Value act = dot(w_, x) + b_;
        return nonlin_ ? act.relu() : act;
    }

    // Overrides Module's parameters() and returns a list of this neurons trainable parameters
    // (its weights and biases)
    // This is what will be updated based on the information in the gradients, before a new training pass starts. 
    std::vector<Value> parameters() override {
        std::vector<Value> ps = w_;
        ps.push_back(b_);
        return ps;
    }
};

class Layer : public Module {
    // A layer of course has many neurons
    std::vector<Neuron> neurons_;

public:
    // Constructor
    // Takes in the amount of weights for each neuron (i.e. the amount of neurons in the previous layer) (nin)
    // Takes in the amount neurons in the layer 
    // Takes in a nonlin flag
    Layer(std::size_t nin, std::size_t nout, bool nonlin = true) {
        neurons_.reserve(nout);
        for (std::size_t i = 0; i < nout; ++i)
            neurons_.emplace_back(nin, nonlin);
    }

    // operator() makes a Layer object callable like a function. 
    // You pass it a vector of Values (the activations from the previous layer)
    // and it outputs a vector of Values (the activations of this layer).
    std::vector<Value> operator()(const std::vector<Value>& x) const {
        std::vector<Value> out;
        out.reserve(neurons_.size());
        for (const auto& n : neurons_) out.push_back(n(x));
        return out;
    }

   // Generate one large vector containing the trainable parameters of the entire layer
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
    // Constructor
    explicit MLP(std::size_t nin, const std::vector<std::size_t>& nouts) {
        std::vector<std::size_t> sz; sz.reserve(nouts.size() + 1);
        sz.push_back(nin);
        for (auto k : nouts) sz.push_back(k);
        layers_.reserve(nouts.size());
        for (std::size_t i = 0; i < nouts.size(); ++i) {
            bool nonlin = (i + 1 != nouts.size()); // last layer linear
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


// Simple loss function (MSE)
inline Value mse(const std::vector<Value>& yhat, const std::vector<Value>& y) {
    assert(yhat.size() == y.size());
    Value s(0.0);
    for (std::size_t i = 0; i < y.size(); ++i) {
        Value d = yhat[i] - y[i];
        s = s + d * d;
    }
    return s * Value(1.0 / static_cast<double>(y.size()));
}

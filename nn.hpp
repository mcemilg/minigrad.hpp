#ifndef NN_H
#define NN_H
#include <iostream>
#include <vector>
#include <memory>
#include <random>

#include "minigrad.hpp"

using namespace std;

class Module
{
protected:
    vector<shared_ptr<Scalar>> parameters;

public:
    virtual vector<shared_ptr<Scalar>> params() { return parameters; };
    void zero_grad()
    {
        for (auto p : params())
            p->setGrad(0.0);
    };
};

class Neuron : public Module
{
    vector<shared_ptr<Scalar>> w;
    shared_ptr<Scalar> b = Scalar::create(0.0);
    bool nonlin;

public:
    Neuron(int, bool);
    shared_ptr<Scalar> operator()(vector<shared_ptr<Scalar>>);
    vector<shared_ptr<Scalar>> params();
};

class Layer : public Module
{
    vector<Neuron> nodes;

public:
    Layer(int, int, bool);
    vector<shared_ptr<Scalar>> operator()(vector<shared_ptr<Scalar>>);
    vector<shared_ptr<Scalar>> params();
};

class MLP : public Module
{
    vector<Layer> layers;

public:
    MLP(int, vector<int>);
    vector<shared_ptr<Scalar>> operator()(vector<shared_ptr<Scalar>>);
    vector<shared_ptr<Scalar>> params();
};

// NEURON
Neuron::Neuron(int nin, bool nonlin)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i <= nin; i++)
    {
        auto weight = Scalar::create(dis(gen));
        this->w.push_back(weight);
        this->parameters.push_back(weight);
    }
    this->parameters.push_back(this->b);
    this->nonlin = nonlin;
}

shared_ptr<Scalar> Neuron::operator()(vector<shared_ptr<Scalar>> inp)
{
    auto res = Scalar::create(0.0);
    for (int i = 0; i < inp.size(); i++)
        res = res + this->w[i] * inp[i];

    res = res + this->b;

    if (this->nonlin)
        res = res->relu();
    return res;
}

vector<shared_ptr<Scalar>> Neuron::params()
{
    return this->parameters;
}

// LAYER
Layer::Layer(int nin, int nout, bool nonlin)
{
    for (int i = 0; i < nout; i++)
        this->nodes.push_back(Neuron(nin, nonlin));
}

vector<shared_ptr<Scalar>> Layer::operator()(vector<shared_ptr<Scalar>> inp)
{
    vector<shared_ptr<Scalar>> out;
    for (auto n : this->nodes)
        out.push_back(n(inp));
    return out;
}

vector<shared_ptr<Scalar>> Layer::params()
{
    vector<shared_ptr<Scalar>> all_params;
    for (auto n : this->nodes)
        for (auto p : n.params())
            all_params.push_back(p);
    return all_params;
}

// MLP
MLP::MLP(int nin, vector<int> nouts)
{
    vector<int> sz;
    sz.push_back(nin);
    for (auto o : nouts)
        sz.push_back(o);

    bool nonlin = true;
    for (int i = 0; i < nouts.size(); i++)
    {
        if (i == nouts.size() - 1)
            nonlin = false;
        this->layers.push_back(Layer(sz[i], sz[i + 1], nonlin));
    }
}

vector<shared_ptr<Scalar>> MLP::operator()(vector<shared_ptr<Scalar>> x)
{
    for (auto l : this->layers)
        x = l(x);
    return x;
}

vector<shared_ptr<Scalar>> MLP::params()
{
    vector<shared_ptr<Scalar>> all_params;
    for (auto l : this->layers)
        for (auto p : l.params())
            all_params.push_back(p);
    return all_params;
}

#endif

#ifndef MINIGRAD_H
#define MINIGRAD_H

#include <set>
#include <vector>
#include <cstdio>
#include <memory>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <functional>

using namespace std;

///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///                               Scalar
///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// The Scalar is a node to build the computation graph. It stores the data,
/// grad value, backward operation and previous children. It is the core class to
/// handle the forward and the backward operations.
/// It supports backward of basic math operations and relu function.
/// Because the nature of the implementation of it requires to references, the
/// class is forced to use with smart pointers.
///

class Scalar : public enable_shared_from_this<Scalar>
{
protected:
    struct private_key;

private:
    double data;
    vector<shared_ptr<Scalar>> prev;
    double grad = 0.0;

    function<void()> _backward = []() {};

public:
    // For class to be able to used with only shared pointer.
    // Check here for more details:
    // https://stackoverflow.com/questions/8147027
    Scalar(const private_key &, double d) { data = d; };
    explicit Scalar(const private_key &) {}
    template <typename... T>
    static shared_ptr<Scalar> create(T &&...args)
    {
        return make_shared<Scalar>(private_key{0}, forward<T>(args)...);
    }

    void backward();

    friend shared_ptr<Scalar> operator+(shared_ptr<Scalar>, shared_ptr<Scalar>);
    friend shared_ptr<Scalar> operator+(shared_ptr<Scalar>, double);
    friend shared_ptr<Scalar> operator+(double, shared_ptr<Scalar>);
    friend shared_ptr<Scalar> operator*(shared_ptr<Scalar>, shared_ptr<Scalar>);
    friend shared_ptr<Scalar> operator*(shared_ptr<Scalar>, double);
    friend shared_ptr<Scalar> operator*(double, shared_ptr<Scalar>);
    friend shared_ptr<Scalar> operator-(shared_ptr<Scalar>);
    friend shared_ptr<Scalar> operator-(shared_ptr<Scalar>, shared_ptr<Scalar>);
    friend shared_ptr<Scalar> operator-(double, shared_ptr<Scalar>);
    friend shared_ptr<Scalar> operator/(shared_ptr<Scalar>, shared_ptr<Scalar>);
    friend shared_ptr<Scalar> operator/(shared_ptr<Scalar>, double);
    friend shared_ptr<Scalar> operator/(double, shared_ptr<Scalar>);
    friend shared_ptr<Scalar> pow(shared_ptr<Scalar>, double);
    shared_ptr<Scalar> relu();

    double getData() const { return data; };
    double getGrad() { return grad; };
    void setGrad(double g) { grad = g; };
    void setData(double d) { data = d; };
    vector<shared_ptr<Scalar>> getPrev() { return prev; };
    operator std::string() const { return to_string(data); };

protected:
    struct private_key
    {
        explicit private_key(int) {}
    };
};

// OPERATORS
shared_ptr<Scalar> operator+(shared_ptr<Scalar> a, shared_ptr<Scalar> b)
{
    auto o = Scalar::create(a->data + b->data);
    o->prev.push_back(a);
    o->prev.push_back(b);

    function<void()> op_backward = [o, a, b]()
    {
        a->grad += o->grad;
        b->grad += o->grad;
    };
    o->_backward = op_backward;

    return o;
}

shared_ptr<Scalar> operator+(shared_ptr<Scalar> a, double v)
{
    auto b = Scalar::create(v);
    return a + b;
}

shared_ptr<Scalar> operator+(double v, shared_ptr<Scalar> b)
{
    auto a = Scalar::create(v);
    return b + a;
}

shared_ptr<Scalar> operator*(shared_ptr<Scalar> a, shared_ptr<Scalar> b)
{
    auto o = Scalar::create(a->data * b->data);
    o->prev.push_back(a);
    o->prev.push_back(b);

    function<void()> op_backward = [a, b, o]()
    {
        a->grad += b->data * o->grad;
        b->grad += a->data * o->grad;
    };
    o->_backward = op_backward;

    return o;
}

shared_ptr<Scalar> operator*(shared_ptr<Scalar> a, double v)
{
    auto b = Scalar::create(v);
    return a * b;
}

shared_ptr<Scalar> operator*(double v, shared_ptr<Scalar> b)
{
    auto a = Scalar::create(v);
    return b * a;
}

shared_ptr<Scalar> operator-(shared_ptr<Scalar> a)
{
    auto b = Scalar::create(-1.0);
    return a * b;
}

shared_ptr<Scalar> operator-(shared_ptr<Scalar> a, shared_ptr<Scalar> b)
{
    return a + (-b);
}

shared_ptr<Scalar> operator-(double v, shared_ptr<Scalar> a)
{
    auto b = Scalar::create(v);
    return b - a;
}

shared_ptr<Scalar> pow(shared_ptr<Scalar> a, double v)
{
    auto o = Scalar::create(std::pow(a->data, v));
    o->prev.push_back(a);

    function<void()> op_backward = [a, o, v]()
    {
        a->grad += v * std::pow(a->data, v - 1.0) * o->grad;
    };
    o->_backward = op_backward;

    return o;
}

shared_ptr<Scalar> operator/(shared_ptr<Scalar> a, shared_ptr<Scalar> b)
{
    return a * pow(b, -1.0);
}

shared_ptr<Scalar> operator/(shared_ptr<Scalar> a, double v)
{
    auto b = Scalar::create(v);
    return a / b;
}

shared_ptr<Scalar> operator/(double v, shared_ptr<Scalar> b)
{
    auto a = Scalar::create(v);
    return a / b;
}

shared_ptr<Scalar> Scalar::relu()
{
    double v = (data <= 0.0) ? 0.0 : data;
    auto o = Scalar::create(v);
    o->prev.push_back(shared_from_this());

    function<void()> op_backward = [a = shared_from_this(), o]()
    {
        a->grad += o->grad * (o->data > 0);
    };
    o->_backward = op_backward;

    return o;
}

// BACKWARD
void topo_sort(shared_ptr<Scalar> v, vector<shared_ptr<Scalar>> &nodes, set<shared_ptr<Scalar>> &visited)
{
    if (visited.find(v) == visited.end())
    {
        visited.insert(v);
        for (auto prev : v->getPrev())
        {
            topo_sort(prev, nodes, visited);
        }
        nodes.push_back(v);
    }
}

void Scalar::backward()
{
    vector<shared_ptr<Scalar>> nodes;
    set<shared_ptr<Scalar>> visited;

    // reversed topoligical sort of nodes in computation graph
    topo_sort(shared_from_this(), nodes, visited);
    std::reverse(nodes.begin(), nodes.end());

    // grad of the first node is always 1.0
    this->grad = 1.0;
    for (auto n : nodes)
        n->_backward();
}

#endif

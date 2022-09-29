# minigrad.hpp
A minimal C++ autograd engine implementation inspired by [micrograd](https://github.com/karpathy/micrograd) using C++ Standard Library. It supports only scalar values and it has limited number of operations. Also includes a simple Multi Layer Perceptron network implementation with the scalar values. 

## Autograd Example

Here is an example of how to use `Scalar` engine:

```c++
#include <iostream>
#include "minigrad.hpp"

using namespace std;

int main()
{
    auto a = Scalar::create(-4.0);
    auto b = Scalar::create(2.0);
    auto c = a + b;
    auto d = a * b + b * b * b;
    c = c + c + 1;
    c = c + 1 + c + (-a);
    d = d + d * 2 + (b + a)->relu();
    d = d + 3 * d + (b - a)->relu();
    auto e = c - d;
    auto f = e * e;
    auto g = f / 2.0;
    g = g + 10.0 / f;
    cout << "g value : " << string(*g) << " grad value " << g->getGrad() << endl; // value 24.704082
    g->backward();
    cout << "a value : " << string(*a) << " grad value " << a->getGrad() << endl; // grad 138.834
    cout << "b value : " << string(*b) << " grad value " << b->getGrad() << endl; // grad 645.577

    return 0;
}
```

It can be build and tried as follows using the example `Makefile`:

```bash
$ make
$ ./minigrad
```

## MLP Example

An example of how to train an MLP network using SGD with `minigrad.hpp` shown in `nn_example.cpp`. The data set is generated with sckit-learn make moons method and a model requires nonlinearity to fit it.

The MLP demo can be run as follows:

```bash
$ make nn
$ ./nn
Step 0 Loss 0.950308 Accuracy 42
Step 1 Loss 2.413774 Accuracy 72
Step 2 Loss 0.829944 Accuracy 81
Step 3 Loss 0.538805 Accuracy 80
Step 4 Loss 0.342237 Accuracy 82
Step 5 Loss 0.293294 Accuracy 84
Step 6 Loss 0.270085 Accuracy 88
Step 7 Loss 0.253929 Accuracy 89
Step 8 Loss 0.241127 Accuracy 92
Step 9 Loss 0.231659 Accuracy 93
Step 10 Loss 0.229211 Accuracy 94
Step 11 Loss 0.220955 Accuracy 93
Step 12 Loss 0.216317 Accuracy 94
Step 13 Loss 0.200522 Accuracy 95
Step 14 Loss 0.194710 Accuracy 96
Step 15 Loss 0.189528 Accuracy 95
Step 16 Loss 0.212723 Accuracy 95
Step 17 Loss 0.336296 Accuracy 90
Step 18 Loss 0.163745 Accuracy 95
Step 19 Loss 0.140295 Accuracy 96
Step 20 Loss 0.111517 Accuracy 97
.
.
.
```
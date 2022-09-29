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
    cout << "g value : " << string(*g) << " grad value " << g->getGrad() << endl;
    g->backward();
    cout << "a value : " << string(*a) << " grad value " << a->getGrad() << endl;
    cout << "b value : " << string(*b) << " grad value " << b->getGrad() << endl;

    return 0;
}

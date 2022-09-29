#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "nn.hpp"

using namespace std;

vector<vector<shared_ptr<Scalar>>> readX(string fname)
{
    vector<vector<shared_ptr<Scalar>>> X;
    string line;
    ifstream f(fname);
    if (!f)
    {
        std::cerr << "Unable to open file " << fname << endl;
        exit(-1);
    }
    double x1, x2;
    while (f >> x1 >> x2)
    {
        vector<shared_ptr<Scalar>> row{Scalar::create(x1), Scalar::create(x2)};
        X.push_back(row);
    }
    f.close();
    return X;
}

vector<shared_ptr<Scalar>> ready(string fname)
{
    vector<shared_ptr<Scalar>> y;
    string line;
    ifstream f(fname);
    if (!f)
    {
        std::cerr << "Unable to open file " << fname << endl;
        exit(-1);
    }
    double x;
    while (f >> x)
        y.push_back(Scalar::create(x));
    f.close();
    return y;
}

shared_ptr<Scalar> loss(MLP &model, vector<shared_ptr<Scalar>> scores, vector<shared_ptr<Scalar>> y)
{
    // svm max margin loss
    shared_ptr<Scalar> data_loss = Scalar::create(0.0);
    vector<shared_ptr<Scalar>> losses;
    for (int i = 0; i < scores.size(); i++)
        data_loss = data_loss + (1 + -y[i] * scores[i])->relu();
    data_loss = data_loss / scores.size();

    // L2 regularization
    double alpha = 0.0001;
    shared_ptr<Scalar> reg_loss = Scalar::create(0.0);

    auto total_loss = data_loss + reg_loss;
    return total_loss + reg_loss;
}

double accuracy(vector<shared_ptr<Scalar>> scores, vector<shared_ptr<Scalar>> y)
{
    double acc = 0.0;
    double si, yi;
    for (int i = 0; i < scores.size(); i++)
    {
        si = scores[i]->getData();
        yi = y[i]->getData();
        if (((si >= 0) && (yi >= 0)) || ((si <= 0) && (yi <= 0)))
            acc = acc + 1.0;
    }
    return acc / scores.size();
}

int main()
{
    auto X = readX("dataset/X.csv");
    auto y = ready("dataset/y.csv");

    vector<int> outs = {16, 16, 1};
    auto model = MLP(2, outs);

    vector<shared_ptr<Scalar>> scores;
    shared_ptr<Scalar> total_loss;
    double acc;
    double lr;

    // optimization
    for (int i = 0; i < 100; i++)
    {
        // forward model
        for (auto xi : X)
            scores.push_back(model(xi)[0]);

        // calc loss & acc
        total_loss = loss(model, scores, y);
        acc = accuracy(scores, y);

        // grad
        model.zero_grad();
        total_loss->backward();

        // sgd
        lr = 1.0 - 0.9 * i / 100;
        for (auto p : model.params())
            p->setData(p->getData() - (lr * p->getGrad()));

        scores.clear();

        cout << "Step " << i << " Loss " << string(*total_loss) << " Accuracy " << acc * 100 << endl;
    }

    return 0;
}


minigrad: minigrad.hpp example.cpp
	g++ example.cpp -o minigrad

nn: minigrad.hpp nn.hpp nn_example.cpp
	g++ nn_example.cpp -o nn


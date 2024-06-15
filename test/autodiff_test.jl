using Test
include("../src/autodiff/graph.jl")
include("../src/autodiff/operations.jl")

# sin(x^2) at x = 1.4
x = Variable(1.40f0, name="x")
two = Constant(2.0f0)
y = sin(x^two)

graph = topological_sort(y)
res = forward!(graph)
backward!(graph)

@test res ≈ 0.925 atol = 0.001
@test x.gradient ≈ -1.062 atol = 0.001

# sigmoid(tanh(x+y)) at x = 0.5, y = 0.3
x = Variable(0.50f0, name="x")
y = Variable(0.30f0, name="y")

z = σ(tanh(x + y))
graph = topological_sort(z)
res = forward!(graph)
backward!(graph)

# https://www.wolframalpha.com/input?i=+sigmoid%28tanh%28x%2By%29%29+at+%280.5%2C++0.3%29
@test res ≈ 0.660167 atol = 0.001
# https://www.wolframalpha.com/input?i=d%2Fdx+sigmoid%28tanh%28x%2By%29%29+at+%280.5%2C++0.3%29
@test x.gradient ≈ 0.125422 atol = 0.001
# https://www.wolframalpha.com/input?i=d%2Fdy+sigmoid%28tanh%28x%2By%29%29+at+%280.5%2C++0.3%29
@test y.gradient ≈ 0.125422 atol = 0.001
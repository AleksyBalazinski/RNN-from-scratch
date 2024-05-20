include("autodiff/graph.jl")
include("autodiff/operations.jl")

x = Variable(5.0, name="x")
two = Constant(2.0)
squared = x^two
sine = sin(squared)

order = topological_sort(sine)

y = forward!(order)
backward!(order)
println(y)
println(x.gradient)
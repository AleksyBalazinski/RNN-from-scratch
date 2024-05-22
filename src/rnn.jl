include("autodiff/graph.jl")
include("autodiff/operations.jl")

using Random
Random.seed!(0)

function make_rnn(seq_len, in_size, out_size, m, xs)
    l = out_size # default
    h0 = Variable(zeros(l, m))
    b = Variable(rand(l, 1); name="b")
    W = Variable(rand(l, l); name="W")
    U = Variable(rand(l, in_size); name="U")
    V = Variable(rand(out_size, l); name="V")
    c = Variable(rand(out_size, 1); name="c")

    as = []
    hs = []
    for t in 1:seq_len
        h = t == 1 ? h0 : hs[t-1]
        push!(as, W * h .+ U * xs[t] .+ b)
        push!(hs, tanh.(as[t]))
    end
    o = V * last(hs) .+ c
    return o
end

seq_len = 3
in_size = 3
out_size = 4
m = 2
xs = [Variable(rand(in_size, m); name="x") for _ in 1:seq_len]
y = Variable(rand(out_size, m), name="y")

ŷ = make_rnn(seq_len, in_size, out_size, m, xs)
E = cross_entropy(ŷ, y)
E.name = "loss"

graph = topological_sort(E)
loss = forward!(graph)
backward!(graph)
println("loss = $loss")

for (i, n) in enumerate(graph)
    print(i, ". ")
    println(n)
end
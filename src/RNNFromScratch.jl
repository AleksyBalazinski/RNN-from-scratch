include("autodiff/graph.jl")
include("autodiff/operations.jl")

using MLDatasets, Flux, Random
Random.seed!(0)

train_data = MLDatasets.MNIST(split=:train)
test_data = MLDatasets.MNIST(split=:test)

function loader(data::MNIST; batchsize::Int=1)
    x1dim = reshape(data.features, 28 * 28, :)
    yhot = Flux.onehotbatch(data.targets, 0:9)
    Flux.DataLoader((x1dim, yhot); batchsize, shuffle=true)
end

using Statistics: mean  # standard library
function accuracy(batch)
    (x_test, y_test) = first(loader(test_data; batchsize=batch))

    reset_state()
    set_value!(xs[1], x_test[1:196, :])
    set_value!(xs[2], x_test[197:392, :])
    set_value!(xs[3], x_test[393:588, :])
    set_value!(xs[4], x_test[589:end, :])
    set_value!(y, y_test)
    forward!(graph)

    acc = round(100 * mean(Flux.onecold(ŷ.output) .== Flux.onecold(y_test)); digits=2)
    acc
end

model_params = []
function add_params(ps...)
    for p in ps
        push!(model_params, p)
    end
end

function add_hidden(hs...)
    for h in hs
        push!(hidden_state, h)
    end
end

hidden_state = []
function reset_state()
    for h in hidden_state
        h.output = zeros(size(h.output))
        h.gradient = nothing
    end
end

function adjust_params(learning_rate)
    for param in model_params
        grad = param.gradient
        if size(param.output, 2) == 1
            grad = reshape(param.gradient[:, 1], :, 1)
        end
        param.output -= learning_rate * grad
    end
end

function dense(in_size::Int, out_size::Int, x::GraphNode)
    W = Variable(Flux.glorot_uniform(out_size, in_size), name="W (dense)")
    b = Variable(Flux.glorot_uniform(out_size, 1), name="b (dense)")
    add_params(W, b)

    return W * x .+ b
end

function rnn(seq_len::Int, in_size::Int, out_size::Int, m::Int, xs::Vector{Variable{Matrix{Float64}}})
    l = out_size # default
    h0 = Variable(zeros(l, m), name="h0")
    b = Variable(Flux.glorot_uniform(l, 1); name="b (rnn)")
    W = Variable(Flux.glorot_uniform(l, l); name="W (rnn)")
    U = Variable(Flux.glorot_uniform(l, in_size); name="U (rnn)")
    V = Variable(Flux.glorot_uniform(out_size, l); name="V (rnn)")
    c = Variable(Flux.glorot_uniform(out_size, 1); name="c (rnn)")
    add_params(b, W, U, V, c, h0)

    as = Vector{GraphNode}()
    hs = Vector{GraphNode}()
    for t in 1:seq_len
        h = t == 1 ? h0 : hs[t-1]
        push!(as, W * h .+ U * xs[t] .+ b)
        push!(hs, tanh.(as[t]))
    end

    h0.name = "h0"
    for (i, h) in enumerate(hs)
        h.name = "h" * string(i)
    end

    add_params(hs...)
    add_hidden(h0, hs...)
    o = V * last(hs) .+ c

    return o
end

settings = (;
    eta=15e-3,
    epochs=5,
    batchsize=100,
)

rnn_in_size = 28 * 28 ÷ 4
rnn_out_size = 64
seq_len = 4
xs = [Variable(rand(rnn_in_size, settings.batchsize); name="x" * string(i)) for i in 1:seq_len]
o_rnn = rnn(seq_len, rnn_in_size, rnn_out_size, settings.batchsize, xs)
o_rnn.name = "o_rnn"

dense_in_size = rnn_out_size
dense_out_size = 10
ŷ = dense(dense_in_size, dense_out_size, o_rnn)
ŷ.name = "ŷ"

y = Variable(rand(dense_out_size, settings.batchsize), name="y")
E = cross_entropy(ŷ, y)
E.name = "loss"

graph = topological_sort(E)
loss = forward!(graph)
backward!(graph)

for (i, n) in enumerate(graph)
    print(i, ". ")
    println(n)
end

for epoch in 1:settings.epochs
    local limit = 100 # limit the number of batches
    local i = 1
    local loss = Inf
    @time for (x_mnist, y_mnist) in loader(train_data, batchsize=settings.batchsize)
        # x_mnist <- (28 * 28 = 784, batchsize = 100)
        # y_mnist <- (10, batchsize = 100)

        # input to the net: sequence [ x[1:196, :], x[197:392,:], x[393:588,:], x[589:end,:] ]
        reset_state()

        set_value!(xs[1], x_mnist[1:196, :])
        set_value!(xs[2], x_mnist[197:392, :])
        set_value!(xs[3], x_mnist[393:588, :])
        set_value!(xs[4], x_mnist[589:end, :])
        set_value!(y, y_mnist)

        loss = forward!(graph)
        backward!(graph)
        adjust_params(settings.eta)

        if i == limit
            break
        end
        i += 1
    end
    acc = accuracy(settings.batchsize)
    @info epoch loss acc
end


include("layers/layers.jl")
Random.seed!(1)

using MLDatasets, Flux

train_data = MLDatasets.MNIST(split=:train)
test_data = MLDatasets.MNIST(split=:test)

function loader(data::MNIST; batchsize::Int=1)
    x1dim = reshape(data.features, 28 * 28, :)
    yhot = Flux.onehotbatch(data.targets, 0:9)
    Flux.DataLoader((x1dim, yhot); batchsize, shuffle=true)
end

using Statistics: mean
function loss_and_accuracy(model::Function, data::MNIST)
    batchsize = length(data)
    (x_test, y_test) = only(loader(data; batchsize))
    xs = [Variable(x_test[(i*196+1):((i+1)*196), :]) for i in 0:3]
    y = Variable(y_test)

    ŷ = model(xs)
    E = cross_entropy(ŷ, y)
    graph = topological_sort(E)

    reset_hidden_state()
    loss = forward!(graph)
    acc = round(100 * mean(Flux.onecold(ŷ.output) .== Flux.onecold(y_test)); digits=2)

    return (loss, acc)
end

settings = (;
    eta=15e-3,
    epochs=5,
    batchsize=100,
)

rnn_in_size = 28 * 28 ÷ 4
rnn_out_size = 64
seq_len = 4
dense_in_size = rnn_out_size
dense_out_size = 10

net = chain(
    rnn(rnn_in_size, rnn_out_size, tanh),
    dense(dense_in_size, dense_out_size, bias=true)
)

xs = [Variable(zeros(rnn_in_size, settings.batchsize); name="x" * string(i)) for i in 1:seq_len]
ŷ = net(xs)
ŷ.name = "ŷ"

y = Variable(zeros(dense_out_size, settings.batchsize))
E = cross_entropy(ŷ, y)
E.name = "loss"

graph = topological_sort(E)
loss = forward!(graph)
backward!(graph)

for (i, n) in enumerate(graph)
    print(i, ". ")
    println(n)
end

@show loss_and_accuracy(net, test_data)

using StatProfilerHTML

for epoch in 1:settings.epochs
    local loss = Inf
    function a()
        @time for (x_mnist, y_mnist) in loader(train_data, batchsize=settings.batchsize)
            # x_mnist <- (28 * 28 = 784, batchsize = 100)
            # y_mnist <- (10, batchsize = 100)

            reset_hidden_state()

            set_value!(xs[1], x_mnist[1:196, :])
            set_value!(xs[2], x_mnist[197:392, :])
            set_value!(xs[3], x_mnist[393:588, :])
            set_value!(xs[4], x_mnist[589:end, :])
            set_value!(y, y_mnist)

            forward!(graph)
            backward!(graph)
            adjust_params(settings.eta)
        end
    end
    #@profilehtml a()
    a()

    loss, acc = loss_and_accuracy(net, train_data)
    test_loss, test_acc = loss_and_accuracy(net, test_data)
    @info epoch acc test_acc
end
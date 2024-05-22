include("../autodiff/graph.jl")
include("../autodiff/operations.jl")

using Random
Random.seed!(0)

is_trained = false
cur_param = 1
params = []

function get_cur_param()
    global cur_param += 1
    return params[cur_param-1]
end

function chain(fs::Function...)
    return x -> foldl((acc, f) -> f(acc), fs, init=x)
end

function dense(in_size::Int, out_size::Int, activation::Function; bias=false)
    if is_trained
        W = get_cur_param()
    else
        W = Variable(randn(out_size, in_size), name="W")
        push!(params, W)
    end

    if bias
        if is_trained
            b = get_cur_param()
        else
            b = Variable(randn(out_size, 1), name="b")
            push!(params, b)
        end

        return x -> activation.(W * x .+ b)
    end
    return x -> activation.(W * x)
end

function dense(in_size::Int, out_size::Int; bias=false)
    if is_trained
        W = get_cur_param()
    else
        W = Variable(randn(out_size, in_size), name="W")
        push!(params, W)
    end

    if bias
        if is_trained
            b = get_cur_param()
        else
            b = Variable(randn(out_size, 1), name="b")
            push!(params, b)
        end

        return x -> W * x .+ b
    end
    return x -> W * x
end

function train(x::Variable, y::Variable, net::Function, loss::Function; epochs::Int=10, learning_rate::AbstractFloat=0.01)
    ŷ = net(x)
    E = loss(ŷ, y)
    graph = topological_sort(E)

    for _ in 1:epochs
        loss = forward!(graph)
        backward!(graph)
        println("loss = $loss")
        for param in params
            grad = param.gradient
            if size(param.output, 2) == 1
                grad = reshape(param.gradient[:, 1], :, 1)
            end
            param.output -= learning_rate * grad
        end
    end

    global is_trained = true

    return graph
end
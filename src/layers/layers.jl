include("../autodiff/graph.jl")
include("../autodiff/operations.jl")

using Random
Random.seed!(0)

mutable struct ModelState
    is_trained::Bool
    cur_param::Int64
    params::Vector{Variable}
end

mutable struct HiddenState
    hs::Vector{GraphNode}
end

hidden_state = HiddenState([])

model_state = ModelState(false, 1, [])

function clear_params()
    model_state.is_trained = false
    model_state.cur_param = 1
    empty!(model_state.params)
end

function get_cur_param()
    model_state.cur_param += 1
    return model_state.params[cur_param-1]
end

function add_hidden(hs...)
    for h in hs
        push!(hidden_state.hs, h)
    end
end

function reset_hidden_state()
    for h in hidden_state.hs
        h.output = zeros(size(h.output))
        h.gradient = nothing
    end
end

function chain(fs::Function...)
    return x -> foldl((acc, f) -> f(acc), fs, init=x)
end

using Flux: glorot_uniform

# TODO remove; this is completely useless
function declare_or_get(size::Tuple{Int,Int}, name::String)
    if model_state.is_trained
        var = get_cur_param()
    else
        var = Variable(glorot_uniform(size...); name)
        push!(model_state.params, var)
    end
    return var
end

function dense(in_size::Int, out_size::Int, activation::Function; bias=false)
    W = declare_or_get((out_size, in_size), "W (dense)")

    if bias
        b = declare_or_get((out_size, 1), "b (dense)")
        return x -> activation.(W * x .+ b)
    end
    return x -> activation.(W * x)
end

function dense(in_size::Int, out_size::Int; bias=false)
    W = declare_or_get((out_size, in_size), "W (dense)")

    if bias
        b = declare_or_get((out_size, 1), "b (dense)")
        return x -> W * x .+ b
    end
    return x -> W * x
end

function create_hidden(xs::Vector, b::Variable, W::Variable, U::Variable, out_size::Int)
    empty!(hidden_state.hs)
    l = out_size # default
    _, m = size(first(xs).output)
    seq_len = length(xs)
    h0 = Variable(zeros(l, m), name="h0")

    as = Vector{GraphNode}()
    hs = Vector{GraphNode}()
    for t in 1:seq_len
        h = t == 1 ? h0 : hs[t-1]
        push!(as, W * h .+ U * xs[t] .+ b)
        push!(hs, tanh.(as[t]))
    end

    for (i, h) in enumerate(hs)
        h.name = "h" * string(i)
    end

    add_hidden(h0)
    return hs
end

function rnn(in_size::Int, out_size::Int)
    l = out_size # defualt
    b = declare_or_get((l, 1), "b (rnn)")
    W = declare_or_get((l, l), "W (rnn)")
    U = declare_or_get((l, in_size), "U (rnn)")
    V = declare_or_get((out_size, l), "V (rnn)")
    c = declare_or_get((out_size, 1), "c (rnn)")
    println("rnn params")

    return xs -> V * last(create_hidden(xs, b, W, U, out_size)) .+ c
end

function train(xs::Vector{Variable}, y::Variable, net::Function, loss::Function; epochs::Int=10, learning_rate::AbstractFloat=0.01)
    ŷ = net(xs)
    E = loss(ŷ, y)
    graph = topological_sort(E)

    for _ in 1:epochs
        loss = forward!(graph)
        backward!(graph)
        println("loss = $loss")
        for param in model_state.params
            grad = param.gradient
            if size(param.output, 2) == 1
                grad = reshape(param.gradient[:, 1], :, 1)
            end
            param.output -= learning_rate * grad
        end
    end

    model_state.is_trained = true

    return graph
end

function adjust_params(learning_rate::AbstractFloat)
    for param in model_state.params
        grad = param.gradient
        if size(param.output, 2) == 1
            grad = reshape(param.gradient[:, 1], :, 1)
        end
        param.output -= learning_rate * grad
    end
end
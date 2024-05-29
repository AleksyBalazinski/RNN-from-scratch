include("../autodiff/graph.jl")
include("../autodiff/operations.jl")

using Random
Random.seed!(0)

mutable struct ModelState
    params::Vector{Variable}
    hs::Vector{GraphNode}
end

model_state = ModelState([], [])

function clear_params()
    model_state.is_trained = false
    model_state.cur_param = 1
    empty!(model_state.params)
end

function add_hidden(hs...)
    for h in hs
        push!(model_state.hs, h)
    end
end

function reset_hidden_state()
    for h in model_state.hs
        fill!(h.output, zero(eltype(h.output)))
        h.gradient = nothing
    end
end

function chain(fs::Function...)
    return x -> foldl((acc, f) -> f(acc), fs, init=x)
end

using Flux: glorot_uniform

function declare_var(size::Tuple{Int,Int}, name::String)
    var = Variable(glorot_uniform(size...); name)
    push!(model_state.params, var)
    return var
end

function dense(in_size::Int, out_size::Int, activation::Function; bias=false)
    W = declare_var((out_size, in_size), "W (dense)")

    if bias
        b = declare_var((out_size, 1), "b (dense)")
        return x -> activation.(W * x .+ b)
    end
    return x -> activation.(W * x)
end

function dense(in_size::Int, out_size::Int; bias=false)
    W = declare_var((out_size, in_size), "W (dense)")

    if bias
        b = declare_var((out_size, 1), "b (dense)")
        return x -> W * x .+ b
    end
    return x -> W * x
end

function create_hidden(xs::Vector, b::Variable, W::Variable, U::Variable, out_size::Int)
    empty!(model_state.hs)
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
    b = declare_var((l, 1), "b (rnn)")
    W = declare_var((l, l), "W (rnn)")
    U = declare_var((l, in_size), "U (rnn)")
    V = declare_var((out_size, l), "V (rnn)")
    c = declare_var((out_size, 1), "c (rnn)")

    return xs -> V * last(create_hidden(xs, b, W, U, out_size)) .+ c
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
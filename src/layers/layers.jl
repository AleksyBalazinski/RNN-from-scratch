include("../autodiff/operations.jl")

# nfan (as in Flux)
nfan() = 1, 1
nfan(n) = 1, n
nfan(n_out, n_in) = n_in, n_out
nfan(dims::Tuple) = nfan(dims...)
nfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end])

# Xavier weight initialization
using Random
function glorot_uniform(dims::Integer...; gain::Float32=1.0f0)
    scale = gain * sqrt(24.0f0 / sum(nfan(dims...)))
    rng = Random.default_rng()
    (rand(rng, Float32, dims...) .- 0.5f0) .* scale
end

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
        fill!(h.output, 0)
        h.has_grad = false
    end
end

"""
Chain different layers together
"""
function chain(fs::Function...)
    return x -> foldl((acc, f) -> f(acc), fs, init=x)
end

function declare_var(size::Tuple{Int,Int}, name::String)
    var = Variable(glorot_uniform(size...); name)
    push!(model_state.params, var)
    return var
end

"""
Dense layer
"""
function dense(in_size::Int, out_size::Int, activation::Function; bias=false)
    W = declare_var((out_size, in_size), "W (dense)")

    if bias
        b = declare_var((out_size, 1), "b (dense)")
        return x -> activation.(W * x .+ b)
    end
    return x -> activation.(W * x)
end

"""
Dense layer with no activation function
"""
function dense(in_size::Int, out_size::Int; bias=false)
    W = declare_var((out_size, in_size), "W (dense)")

    if bias
        b = declare_var((out_size, 1), "b (dense)")
        return x -> W * x .+ b
    end
    return x -> W * x
end

function create_hidden(xs::Vector, b::Variable, W::Variable, U::Variable, out_size::Int, σ::Function)
    empty!(model_state.hs)
    l = out_size # default
    _, m = size(first(xs).output)
    seq_len = length(xs)
    h0 = Variable(zeros(Float32, l, m), name="h0")

    as = Vector{GraphNode}()
    hs = Vector{GraphNode}()
    for t in 1:seq_len
        h = t == 1 ? h0 : hs[t-1]
        push!(as, W * h .+ U * xs[t] .+ b)
        push!(hs, σ.(as[t]))
    end

    for (i, h) in enumerate(hs)
        h.name = "h" * string(i)
    end

    add_hidden(h0)
    return hs
end

"""
Vanilla RNN layer
"""
function rnn(in_size::Int, out_size::Int, σ::Function=tanh)
    l = out_size # defualt
    b = declare_var((l, 1), "b (rnn)")
    W = declare_var((l, l), "W (rnn)")
    U = declare_var((l, in_size), "U (rnn)")
    V = declare_var((out_size, l), "V (rnn)")
    c = declare_var((out_size, 1), "c (rnn)")

    return xs -> V * last(create_hidden(xs, b, W, U, out_size, σ)) .+ c
end

"""
One step of gradient descent
"""
function adjust_params(learning_rate::Float32)
    for param in model_state.params
        grad = param.gradient
        if size(param.output, 2) == 1
            grad = reshape(param.gradient[:, 1], :, 1)
        end
        param.output -= learning_rate * grad
    end
end
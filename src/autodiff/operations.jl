include("graph.jl")
import Base: ^
^(x::GraphNode, n::Constant) = ScalarOperator(^, x, n)
forward(::ScalarOperator{typeof(^)}, x, n) = return x^n
backward(::ScalarOperator{typeof(^)}, x, n, g) = tuple(g * n * x^(n - 1), 0)

import Base: sin
sin(x::GraphNode) = ScalarOperator(sin, x)
forward(::ScalarOperator{typeof(sin)}, x) = return sin(x)
backward(::ScalarOperator{typeof(sin)}, x, g) = tuple(g * cos(x))

import Base: *
import LinearAlgebra: mul!
# x * y (aka matrix multiplication)
*(x::GraphNode, y::GraphNode) = BroadcastedOperator(mul!, x, y)
forward(::BroadcastedOperator{typeof(mul!)}, x, y) = return x * y
backward(::BroadcastedOperator{typeof(mul!)}, x, y, g) = tuple(g * y', x' * g)

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = return x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g, -g)

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = return x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(g, g)

σ(x) = 1 / (1 + exp(-x))
Base.Broadcast.broadcasted(σ, x::GraphNode) = BroadcastedOperator(σ, x)
forward(::BroadcastedOperator{typeof(σ)}, x) = return σ.(x)
function backward(op::BroadcastedOperator{typeof(σ)}, x, g)
    y = op.output
    res = y .* (1 .- y) .* g
    return tuple(res)
end

Base.Broadcast.broadcasted(tanh, x::GraphNode) = BroadcastedOperator(tanh, x)
forward(::BroadcastedOperator{typeof(tanh)}, x) = return tanh.(x)
function backward(op::BroadcastedOperator{typeof(tanh)}, x, g)
    y = op.output
    res = (1 .- y .^ 2) .* g
    return tuple(res)
end

Base.Broadcast.broadcasted(^, x::GraphNode, n::Constant) = BroadcastedOperator(^, x, n)
forward(::BroadcastedOperator{typeof(^)}, x, n) = return x .^ n
backward(::BroadcastedOperator{typeof(^)}, x, n, g) = tuple(n .* x .^ (n - 1) .* g, 0)

mean_squared_loss(ŷ, y) = 0.5 .* (y .- ŷ) * (y .- ŷ)'
mean_squared_loss(x::GraphNode, y::GraphNode) = ScalarOperator(mean_squared_loss, x, y)
forward(::ScalarOperator{typeof(mean_squared_loss)}, x, y) = return mean_squared_loss(x, y)
backward(::ScalarOperator{typeof(mean_squared_loss)}, x, y, g) = return tuple(x .- y, x .- y)

function cross_entropy(ŷ, y)
    _, m = size(ŷ)
    e = exp.(ŷ)
    p = e ./ sum(e, dims=1)
    ϵ = eps(eltype(ŷ))
    res = -1 / m * sum(y .* log.(p .+ ϵ))
    return res
end
cross_entropy(x::GraphNode, y::GraphNode) = ScalarOperator(cross_entropy, x, y)
forward(::ScalarOperator{typeof(cross_entropy)}, x, y) = return cross_entropy(x, y)
function backward(::ScalarOperator{typeof(cross_entropy)}, ŷ, y, g)
    _, m = size(ŷ)
    e = exp.(ŷ)
    p = e ./ sum(e, dims=1)

    dŷ = 1 / m * (p .- y)
    dy = zero(y)
    return tuple(dŷ, dy)
end
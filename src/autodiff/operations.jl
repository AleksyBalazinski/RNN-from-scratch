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
*(x::GraphNode, y::GraphNode) = BroadcastedOperator(mul!, Array{Float64,2}(undef, size(x.output, 1), size(y.output, 2)), x, y)
forward(o::BroadcastedOperator{typeof(mul!)}, x, y) = mul!(o.output, x, y)
function backward(o::BroadcastedOperator{typeof(mul!)}, x, y, g)
    mul!(o.temp[1], g, y')
    mul!(o.temp[2], x', g)
    return tuple(o.temp[1], o.temp[2])
end

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, Array{Float64,2}(undef, size(x.output)), x, y)
forward(o::BroadcastedOperator{typeof(-)}, x, y) = broadcast!(-, o.output, x, y)
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g, -g)

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, Array{Float64,2}(undef, size(x.output)), x, y)
forward(o::BroadcastedOperator{typeof(+)}, x, y) = broadcast!(+, o.output, x, y)
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(g, g)

σ(x) = 1 / (1 + exp(-x))
Base.Broadcast.broadcasted(σ, x::GraphNode) = BroadcastedOperator(σ, Array{Float64,2}(undef, size(x.output)), x)
forward(o::BroadcastedOperator{typeof(σ)}, x) = o.output[:] = σ.(x)
function backward(op::BroadcastedOperator{typeof(σ)}, x, g)
    y = op.output
    res = y .* (1 .- y) .* g
    return tuple(res)
end

Base.Broadcast.broadcasted(tanh, x::GraphNode) = BroadcastedOperator(tanh, Array{Float64,2}(undef, size(x.output, 1), size(x.output, 2)), x)
function forward(o::BroadcastedOperator{typeof(tanh)}, x) 
    @inbounds @simd for i in eachindex(x)
        o.output[i] = tanh(x[i])
    end
end
function backward(op::BroadcastedOperator{typeof(tanh)}, x, g)
    y = op.output
    res = (1 .- y .^ 2) .* g
    return tuple(res)
end

Base.Broadcast.broadcasted(^, x::GraphNode, n::Constant) = BroadcastedOperator(^, Array{Float64,2}(undef, size(x.output, 1), size(x.output, 2)), x, n)
forward(o::BroadcastedOperator{typeof(^)}, x, n) = o.output .= x .^ n
backward(::BroadcastedOperator{typeof(^)}, x, n, g) = tuple(n .* x .^ (n - 1) .* g, 0)

mean_squared_loss(ŷ, y) = 0.5 .* (y .- ŷ) * (y .- ŷ)'
mean_squared_loss(x::GraphNode, y::GraphNode) = ScalarOperator(mean_squared_loss, x, y)
forward(o::ScalarOperator{typeof(mean_squared_loss)}, x, y) = o.output = mean_squared_loss(x, y)
backward(::ScalarOperator{typeof(mean_squared_loss)}, x, y, g) = return tuple(x .- y, x .- y)

function cross_entropy(ŷ, y)
    _, m = size(ŷ)
    e = exp.(ŷ)
    p = e ./ sum(e, dims=1)
    ϵ = eps(eltype(ŷ))
    res = -1 / m * sum(y .* log.(p .+ ϵ))
    return res
end
cross_entropy(x::GraphNode, y::GraphNode) = ScalarOperator(cross_entropy, 0.0, x, y)
forward(o::ScalarOperator{typeof(cross_entropy)}, x, y) = o.output = cross_entropy(x, y)
function backward(::ScalarOperator{typeof(cross_entropy)}, ŷ, y, g)
    _, m = size(ŷ)
    e = exp.(ŷ)
    p = e ./ sum(e, dims=1)

    dŷ = 1 / m * (p .- y)
    dy = nothing
    return tuple(dŷ, dy)
end
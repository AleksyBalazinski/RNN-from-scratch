include("graph.jl")

import Base: ^
^(x::GraphNode, n::Constant) = ScalarOperator(^, 0.0f0, x, n)
forward(o::ScalarOperator{typeof(^)}, x::Float32, n::Float32) = o.output = x^n
backward(::ScalarOperator{typeof(^)}, x::Float32, n::Float32, g::Float32) = tuple(g * n * x^(n - 1), 0.0f0)

import Base: sin
sin(x::GraphNode) = ScalarOperator(sin, 0.0f0, x)
forward(o::ScalarOperator{typeof(sin)}, x::Float32) = o.output = sin(x)
backward(::ScalarOperator{typeof(sin)}, x::Float32, g::Float32) = tuple(g * cos(x))

import Base: *
import LinearAlgebra: mul!
import LinearAlgebra.BLAS: gemm!
# x * y (aka matrix multiplication)
*(x::GraphNode, y::GraphNode) = BroadcastedOperator(mul!, Matrix{Float32}(undef, size(x.output, 1), size(y.output, 2)), x, y)
function forward(o::BroadcastedOperator{typeof(mul!)}, x::Matrix{Float32}, y::Matrix{Float32})
    mul!(o.output, x, y)
end

function backward(o::BroadcastedOperator{typeof(mul!)}, x::Matrix{Float32}, y::Matrix{Float32}, g::Matrix{Float32})
    gemm!('N', 'T', 1.0f0, g, y, 0.0f0, o.temp[1])
    gemm!('T', 'N', 1.0f0, x, g, 0.0f0, o.temp[2])
    return tuple(o.temp[1], o.temp[2])
end

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, Matrix{Float32}(undef, size(x.output)), x, y)
forward(o::BroadcastedOperator{typeof(-)}, x::Matrix{Float32}, y::Matrix{Float32}) = broadcast!(-, o.output, x, y)
backward(::BroadcastedOperator{typeof(-)}, x::Matrix{Float32}, y::Matrix{Float32}, g::Matrix{Float32}) = tuple(g, -g)

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, Matrix{Float32}(undef, size(x.output)), x, y)
forward(o::BroadcastedOperator{typeof(+)}, x::Matrix{Float32}, y::Matrix{Float32}) = broadcast!(+, o.output, x, y)
backward(::BroadcastedOperator{typeof(+)}, x::Matrix{Float32}, y::Matrix{Float32}, g::Matrix{Float32}) = tuple(g, g)

import Base: +
+(x::GraphNode, y::GraphNode) = ScalarOperator(+, 0.0f0, x, y)
forward(o::ScalarOperator{typeof(+)}, x, y) = o.output = x + y
backward(::ScalarOperator{typeof(+)}, x, y, g) = tuple(g, g)

σ(x) = 1 / (1 + exp(-x))
Base.Broadcast.broadcasted(σ, x::GraphNode) = BroadcastedOperator(σ, Matrix{Float32}(undef, size(x.output)), x)
forward(o::BroadcastedOperator{typeof(σ)}, x::Matrix{Float32}) = o.output .= σ.(x)
function backward(op::BroadcastedOperator{typeof(σ)}, ::Matrix{Float32}, g::Matrix{Float32})
    y .= op.output
    res = y .* (1.0f0 .- y) .* g
    return tuple(res)
end

σ(x::GraphNode) = ScalarOperator(σ, 0.0f0, x)
forward(o::ScalarOperator{typeof(σ)}, x::Float32) = o.output = σ(x)
function backward(op::ScalarOperator{typeof(σ)}, ::Float32, g::Float32)
    y = op.output
    res = y * (1.0f0 - y) * g
    return tuple(res)
end

Base.Broadcast.broadcasted(tanh, x::GraphNode) = BroadcastedOperator(tanh, Matrix{Float32}(undef, size(x.output, 1), size(x.output, 2)), x)
function forward(o::BroadcastedOperator{typeof(tanh)}, x::Matrix{Float32})
    o.output .= tanh.(x)
end
function backward(op::BroadcastedOperator{typeof(tanh)}, ::Matrix{Float32}, g::Matrix{Float32})
    y = op.output
    res = (1.0f0 .- y .^ 2) .* g
    return tuple(res)
end

import Base: tanh
tanh(x::GraphNode) = ScalarOperator(tanh, 0.0f0, x)
forward(o::ScalarOperator{typeof(tanh)}, x::Float32) = o.output = tanh(x)
function backward(op::ScalarOperator{typeof(tanh)}, ::Float32, g::Float32)
    y = op.output
    res = (1.0f0 - y^2) * g
    return tuple(res)
end

Base.Broadcast.broadcasted(^, x::GraphNode, n::Constant) = BroadcastedOperator(^, Matrix{Float32}(undef, size(x.output, 1), size(x.output, 2)), x, n)
forward(o::BroadcastedOperator{typeof(^)}, x, n) = o.output .= x .^ n
backward(::BroadcastedOperator{typeof(^)}, x, n, g) = tuple(n .* x .^ (n - 1) .* g, 0)

mean_squared_loss(ŷ, y) = 0.5f0 .* (y .- ŷ) * (y .- ŷ)'
mean_squared_loss(x::GraphNode, y::GraphNode) = ScalarOperator(mean_squared_loss, x, y)
forward(o::ScalarOperator{typeof(mean_squared_loss)}, x, y) = o.output = mean_squared_loss(x, y)
backward(::ScalarOperator{typeof(mean_squared_loss)}, x, y, g) = return tuple(x .- y, x .- y)

function cross_entropy(ŷ, y)
    _, m = size(ŷ)
    e = exp.(ŷ)
    p = e ./ sum(e, dims=1)
    ϵ = eps(eltype(ŷ))
    res = -1.0f0 / m * sum(y .* log.(p .+ ϵ))
    return res
end
cross_entropy(x::GraphNode, y::GraphNode) = ScalarOperator(cross_entropy, 0.0f0, x, y)
forward(o::ScalarOperator{typeof(cross_entropy)}, x, y) = o.output = cross_entropy(x, y)
function backward(::ScalarOperator{typeof(cross_entropy)}, ŷ::Matrix{Float32}, y::Matrix{Float32}, g::Float32)
    _, m = size(ŷ)
    e = exp.(ŷ)
    p = e ./ sum(e, dims=1)

    dŷ = 1.0f0 / m * (p .- y)
    dy = nothing
    return tuple(dŷ, dy)
end
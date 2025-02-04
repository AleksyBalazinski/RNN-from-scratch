abstract type GraphNode end
abstract type Operator <: GraphNode end
const MaybeValue = Union{Float32,Matrix{Float32},Nothing}

struct Constant{T} <: GraphNode
    output::T
end

mutable struct Variable{T} <: GraphNode
    output::T
    gradient::Union{T,Nothing}
    has_grad::Bool
    name::String
    function Variable(output::T; name="?") where {T}
        new{T}(output, nothing, false, name)
    end
end

set_value!(variable::Variable, value) = variable.output .= value

mutable struct ScalarOperator{F} <: Operator
    inputs::Tuple
    output::Float32
    gradient::Float32
    has_grad::Bool
    name::String
    function ScalarOperator(fun, output, inputs...; name="?")
        new{typeof(fun)}(inputs, output, 0.0f0, false, name)
    end
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs::Tuple
    output::Matrix{Float32}
    gradient::Matrix{Float32}
    has_grad::Bool
    name::String
    temp_mul::Tuple{Matrix{Float32},Matrix{Float32}}
    temp_broadcast::Matrix{Float32}

    function BroadcastedOperator(fun, output::Matrix{Float32}, inputs...; name::String="?")
        temp_mul = (Matrix{Float32}(undef, 0, 0), Matrix{Float32}(undef, 0, 0))
        temp_broadcast = Matrix{Float32}(undef, 0, 0)

        if fun == mul!
            temp_mul = (Matrix{Float32}(undef, size(output, 1), size(inputs[2].output, 1)),
                Matrix{Float32}(undef, size(inputs[1].output, 2), size(output, 2)))
        else
            temp_broadcast = Matrix{Float32}(undef, size(output))
        end

        new{typeof(fun)}(inputs, output, Matrix{Float32}(undef, 0, 0), false, name, temp_mul, temp_broadcast)
    end
end

import Base: show, summary
show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")");
show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")");
show(io::IO, x::Constant) = print(io, "const ", x.output)
show(io::IO, x::Variable) = begin
    print(io, "var ", x.name)
    print(io, "\n ┣━ ^ ")
    summary(io, x.output)
    print(io, "\n ┗━ ∇ ")
    summary(io, x.gradient)
end

function visit(node::GraphNode, visited::Set{GraphNode}, order::Vector{GraphNode})
    if node ∈ visited
    else
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end

function visit(node::Operator, visited::Set{GraphNode}, order::Vector{GraphNode})
    if node ∈ visited
    else
        push!(visited, node)
        for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function topological_sort(head::GraphNode)
    visited = Set{GraphNode}()
    order = Vector{GraphNode}()
    visit(head, visited, order)
    return order
end

reset!(node::Constant) = nothing
reset!(node::Variable) = node.has_grad = false
reset!(node::Operator) = node.has_grad = false

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
function compute!(node::Operator)
    forward(node, [input.output for input in node.inputs]...)
end

function forward!(order::Vector{GraphNode})
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end

update!(node::Constant, gradient::MaybeValue) = nothing
function update!(node::GraphNode, gradient::MaybeValue)
    if isnothing(gradient)
        node.has_grad = false
        return
    end
    if !node.has_grad
        node.gradient = gradient
        node.has_grad = true
    else
        node.gradient .+= gradient
    end
end

function backward!(order::Vector{GraphNode}; seed=1.0f0)
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::Operator)
    inputs = node.inputs
    gradients = backward(node, [input.output for input in inputs]..., node.gradient)
    for (input, gradient) in zip(inputs, gradients)
        update!(input, gradient)
    end
    return nothing
end
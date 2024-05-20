# copy-paste of the class notebook

include("autodiff/graph.jl")
include("autodiff/operations.jl")

using Random

# Set the seed
Random.seed!(0)

using LinearAlgebra
Wh = Variable(randn(10, 2), name="wh")
Wo = Variable(randn(1, 10), name="wo")
x = Variable([1.98, 4.434], name="x")
y = Variable([0.064], name="y")
losses = Float64[]

function dense(w, b, x, activation)
    return activation.(w * x .+ b)
end
function dense(w, x, activation)
    return activation.(w * x)
end
function dense(w, x)
    return w * x
end

function net(x, wh, wo, y)
    x̂ = dense(wh, x, σ)
    x̂.name = "x̂"
    ŷ = dense(wo, x̂)
    ŷ.name = "ŷ"
    E = mean_squared_loss(y, ŷ)
    E.name = "loss"

    return topological_sort(E)
end
graph = net(x, Wh, Wo, y)
forward!(graph)
backward!(graph)

for (i, n) in enumerate(graph)
    print(i, ". ")
    println(n)
end

# Manual (adjusted indices)
eye(n) = diagm(ones(n))
Eŷ = graph[7].output - y.output #ŷ
ŷȳ = graph[7].output |> length |> eye #ŷ
ȳWo = graph[6].output |> transpose #x̂
x̄Wh = graph[4].output |> transpose #x
ȳx̂ = graph[2].output |> transpose #Wo
x̂x̄ = graph[6].output .* (1.0 .- graph[6].output) |> diagm #x̂
Eȳ = ŷȳ * Eŷ
Ex̂ = ȳx̂ * Eȳ
Ex̄ = x̂x̄ * Ex̂
EWo = Eȳ * ȳWo
EWh = Ex̄ * x̄Wh
nothing

println(EWh)


currentloss = forward!(graph)
backward!(graph)
Wh.output -= 0.01Wh.gradient
Wo.output -= 0.01Wo.gradient
println("Current loss: ", currentloss)
push!(losses, first(currentloss))

println(Wh.gradient)
println(y.gradient)


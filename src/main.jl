include("autodiff/graph.jl")
include("autodiff/operations.jl")

using CSV
using DataFrames

train_cnt = 60
df = CSV.read("./data/housing_normalized.csv", DataFrame; delim=",")
x = Variable(Matrix(df[1:train_cnt, 1:13])')
y = Variable(reshape(Vector(df[1:train_cnt, 14]), :, 1)')

function dense(w, b, x, activation)
    return activation.(w * x .+ b)
end
function dense(w, x, activation)
    return activation.(w * x)
end
function dense(w, x)
    return w * x
end

function net(x, w₁, b₁, w₂, b₂, w₃, y)
    a₁ = dense(w₁, b₁, x, σ)
    a₂ = dense(w₂, b₂, a₁, σ)
    ŷ = dense(w₃, a₂)
    E = mean_squared_loss(y, ŷ)
    E.name = "loss"

    return topological_sort(E)
end

using Random
# Set the seed
Random.seed!(0)
features_cnt = 13
w1 = Variable(randn(3, features_cnt), name="w1")
b1 = Variable(randn(3, 1), name="b1")
w2 = Variable(randn(2, 3), name="w2")
b2 = Variable(randn(2, 1), name="b2")
w3 = Variable(randn(1, 2), name="w3")

graph = net(x, w1, b1, w2, b2, w3, y)

# training loop
for i in 1:10
    local loss = forward!(graph)
    backward!(graph)
    println("loss = $loss")
    local learning_rate = 0.01
    w1.output -= learning_rate * w1.gradient
    b1.output -= learning_rate * reshape(b1.gradient[:, 1], :, 1)
    w2.output -= learning_rate * w2.gradient
    b2.output -= learning_rate * reshape(b2.gradient[:, 1], :, 1)
    w3.output -= learning_rate * w3.gradient
end

# testing
test_cnt = 20
test_range = (train_cnt+1):(train_cnt+test_cnt)
x_test = Variable(Matrix(df[test_range, 1:13])')
y_test = Variable(reshape(Vector(df[test_range, 14]), :, 1)')

graph_test = net(x_test, w1, b1, w2, b2, w3, y_test)
forward!(graph_test)
ŷ = graph_test[length(graph_test)-1]
avg_error = (1 / test_cnt) * sum(abs.(y_test.output - ŷ.output))
println("avg_error = $avg_error")

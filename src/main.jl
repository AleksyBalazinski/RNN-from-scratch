include("autodiff/graph.jl")
include("autodiff/operations.jl")
include("layers/layers.jl")

using CSV
using DataFrames

function min_max_scaling(df::DataFrame, cols::Vector{Symbol})
    scaled_df = copy(df)
    for col in cols
        if eltype(df[!, col]) <: Number
            min_val = minimum(df[!, col])
            max_val = maximum(df[!, col])
            scaled_df[!, col] = (df[!, col] .- min_val) ./ (max_val - min_val)
        end
    end
    return scaled_df
end

train_cnt = 80
df = CSV.read("./data/housing_normalized.csv", DataFrame; delim=",")
df = min_max_scaling(df, Symbol.(names(df)[1:13]))
x = Variable(Matrix(df[1:train_cnt, 1:13])', name="x")
y = Variable(reshape(Vector(df[1:train_cnt, 14]), :, 1)', name="y")

features_cnt = 13
net = chain(dense(13, 3, σ, bias=true), dense(3, 2, bias=true), dense(2, 1))

graph = train(x, y, net, mean_squared_loss; epochs=20, learning_rate=0.0001)
for (i, n) in enumerate(graph)
    print(i, ". ")
    println(n)
end

# testing
test_cnt = 20
test_range = (train_cnt+1):(train_cnt+test_cnt)
x_test = Variable(Matrix(df[test_range, 1:13])')
y_test = Variable(reshape(Vector(df[test_range, 14]), :, 1)')

ŷ = net(x_test) # net() is stateful, every call uses the same parameters
graph_test = topological_sort(ŷ)
forward!(graph_test)
println(ŷ.output)
avg_error = (1 / test_cnt) * sum(abs.(y_test.output - ŷ.output))
println("avg_error = $avg_error")

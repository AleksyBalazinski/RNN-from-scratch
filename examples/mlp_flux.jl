using Flux

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
X = Matrix(df[1:train_cnt, 1:13])'
y = reshape(Vector(df[1:train_cnt, 14]), :, 1)'

model = Chain(
    Dense(13 => 3, σ),
    Dense(3 => 2),
    Dense(2 => 1; bias=false)
)

loss(x, y) = Flux.mse(model(x), y)
optimizer = Descent(0.0001)

epochs = 20
for epoch in 1:epochs
    gs = Flux.gradient(Flux.params(model)) do
        l = loss(X, y)
    end
    Flux.update!(optimizer, Flux.params(model), gs)
    println("Epoch: $epoch, Loss: $(loss(X, y))")
end

# testing
test_cnt = 20
test_range = (train_cnt+1):(train_cnt+test_cnt)
x_test = Matrix(df[test_range, 1:13])'
y_test = reshape(Vector(df[test_range, 14]), :, 1)'

ŷ = model(x_test)
avg_error = (1 / test_cnt) * sum(abs.(ŷ - y_test))
@show ŷ
@show y
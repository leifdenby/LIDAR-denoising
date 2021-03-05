using DrWatson
@quickactivate

include(srcdir("train.jl"))

# generate some fake data to test with
nx, ny, nz = 40, 30, 20
x = rand(Float32, (nz, ny, nx))

trained_model = train_model_on_data(x)

using DrWatson
@quickactivate

include(srcdir("train.jl"))
include(srcdir("model.jl"))

# generate some fake data to test with
nx, ny, nz = 40, 30, 20
x_grid = Float32.(collect(1:nx))
y_grid = Float32.(collect(1:ny))
z_grid = Float32.(collect(1:nz))
data = GriddedData3D(randn(Float32, (nz, ny, nx)), x_grid, y_grid, z_grid)


Nf = 5  # filter size in model convolutions
Nc = 6  # number of "channels" in model convolutions
model = build_model("linear_1x1")
trained_model = train_model_on_data(model, data)

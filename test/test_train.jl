using DrWatson
@quickactivate

include(srcdir("train.jl"))

# generate some fake data to test with
nx, ny, nz = 40, 30, 20
x_grid = Float32.(collect(1:nx))
y_grid = Float32.(collect(1:ny))
z_grid = Float32.(collect(1:nz))
data = GriddedData3D(randn(Float32, (nz, ny, nx)), x_grid, y_grid, z_grid)

trained_model = train_model_on_data(data)

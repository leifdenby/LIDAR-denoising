using DrWatson
using Test
@quickactivate

include(srcdir("model.jl"))
include(srcdir("plot.jl"))
include(srcdir("ncfile.jl"))

nx, ny, nz = 100, 100, 30
x_grid = Float32.(collect(1:nx))
y_grid = Float32.(collect(1:ny))
z_grid = Float32.(collect(1:nz))
data = GriddedData3D(randn(Float32, (nz, ny, nx)), x_grid, y_grid, z_grid)

σ_noise = 0.2
Nf = 5  # filter size in model convolutions
Nc = 6  # number of "channels" in model convolutions
model = build_model(Nf, Nc)

plot_example(data, model, σ_noise, "plot.png")

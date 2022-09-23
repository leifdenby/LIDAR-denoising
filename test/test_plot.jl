using Test
using LIDARdenoising


nx, ny, nz = 100, 100, 30
x_grid = Float32.(collect(1:nx))
y_grid = Float32.(collect(1:ny))
z_grid = Float32.(collect(1:nz))
data = LIDARdenoising.GriddedData3D(randn(Float32, (nz, ny, nx)), x_grid, y_grid, z_grid)

σ_noise = 0.2
Nf = 5  # filter size in model convolutions
Nc = 6  # number of "channels" in model convolutions
model = LIDARdenoising.build_model()

LIDARdenoising.plot_example(data, model, σ_noise, "plot.png")

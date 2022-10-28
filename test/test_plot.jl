using Test
using LIDARdenoising
using LIDARdenoising.Models


@testset "plot example" begin
    nx, ny, nz = 100, 100, 100 # XXX: need to create a SSDN network which supports non-square input
    x_grid = Float32.(collect(1:nx))
    y_grid = Float32.(collect(1:ny))
    z_grid = Float32.(collect(1:nz))
    data = LIDARdenoising.GriddedData3D(randn(Float32, (nz, ny, nx)), x_grid, y_grid, z_grid)

    σ_noise = 0.2
    n_features_in, n_features_out = 1, 1
    n_layers = 2
    model = SSDN(n_features_in, n_layers, n_features_out)

    LIDARdenoising.plot_example(data, model, σ_noise, "plot.png")
    @test isfile("plot.png")
end
using LIDARdenoising
using Test
using Flux


@testset "training /w $(data_kind) data" for data_kind in ["random", "example"]
    if data_kind == "random"
        # generate some fake data to test with
        nx, ny, nz = 40, 30, 20
        x_grid = Float32.(collect(1:nx))
        y_grid = Float32.(collect(1:ny))
        z_grid = Float32.(collect(1:nz))
        data = LIDARdenoising.GriddedData3D(randn(Float32, (nz, ny, nx)), x_grid, y_grid, z_grid)
    elseif data_kind == "example"
        data = LIDARdenoising.load_data(joinpath(@__DIR__, "../sample-data/qv.x0.0.nc"))
    else
        throw("Data kind $(data_kind) not implemented")
    end

    Nf = 5  # filter size in model convolutions
    Nc = 6  # number of "channels" in model convolutions
    model = LIDARdenoising.build_model("linear_1x1")

    initial_params = deepcopy(Flux.params(model))
    trained_model = LIDARdenoising.train_model_on_data(model, data, n_epochs=2) |> cpu
    @test initial_params != Flux.params(trained_model)
end

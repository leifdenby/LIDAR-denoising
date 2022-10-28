using LIDARdenoising
using Test
using Flux


@testset "training" begin
    noisy_data = randn(Float32, (3, 3, 1, 2)); # HWCS = HeightWidthChannelSample
    clean_data = randn(Float32, (3, 3, 1, 2)); # HWCS

    # linear (won't work well)
    denoiser = LinearDenoiser()
    # train noisy-clean pairs without known noise
    train!(denoiser, noisy_data, clean_data)
    # do inference with trained model
    denoiser(noisy_data)

    # n2c
    denoiser = Noise2CleanDenoiser(n_layers=2)
    # train noisy-clean pairs without known noise
    train!(denoiser, noisy_data, clean_data)
    # make noisy samples by adding Gaussian noise to clean samples and train on noisy-clean pairs
    # train!(denoiser, clean_data, :gaussian, σ_noise)
    # do inference with trained model
    denoiser(noisy_data)

    @test_skip @testset "all" begin
        # n2n
        denoiser = Noise2NoiseDenoiser()
        # train by adding Gaussian noise with width σ_noise to each sample seen
        # train!(denoiser, clean_data, :gaussian, σ_noise)
        # train on pair-wise noisy samples without known noise level
        train!(denoiser, noisy_data1, noisy_data2)  # TODO

        # ssdn
        denoiser = LaineSelfSupervisedDenoiser()  # make blindspot=true/false option
        # train directly on noisy data without known noise level
        train!(denoiser, noisy_data)
        # make noisy samples by adding Gaussian noise to clean samples and train on noisy-clean pairs
        # train!(denoiser, clean_data, :gaussian, σ_noise)
    end
end


"""
@testset "training /w $(data_kind) data" for data_kind in ["random", "example"]
    if data_kind == "random"
        # generate some fake data to test with
        nx, ny, nz = 40, 40, 40
        x_grid = Float32.(collect(1:nx))
        y_grid = Float32.(collect(1:ny))
        z_grid = Float32.(collect(1:nz))
        data = LIDARdenoising.GriddedData3D(randn(Float32, (nz, ny, nx)), x_grid, y_grid, z_grid)
    elseif data_kind == "example"
        data = LIDARdenoising.load_data(joinpath(@__DIR__, "../sample-data/qv.x0.0.nc"))
    else
        throw("Data kind $(data_kind) not implemented")
    end

    n_features_in, n_features_out = 1, 1
    n_layers = 2
    models = Dict(
        :linear => LIDARdenoising.Models.Linear(;conv_size=1),
    #        :SSDN => LIDARdenoising.Models.SSDN(n_features_in, n_layers, n_features_out)
    )

    @testset "train $(model_name) model" for (model_name, model) in models
        initial_params = deepcopy(Flux.params(model))
        trained_model = LIDARdenoising.train_model_on_data(model, data, n_epochs=2) |> cpu
        @test initial_params != Flux.params(trained_model)
    end
end
"""

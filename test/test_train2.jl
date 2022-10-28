using LIDARdenoising.Models
using Flux


noisy_data = randn(Float32, (3, 3, 1, 2)); # HWCS = HeightWidthChannelSample
clean_data = randn(Float32, (3, 3, 1, 2)); # HWCS

denoiser = Noise2NoiseDenoiser()
# train on pair-wise noisy samples without known noise level
train!(denoiser, noisy_data1, noisy_data2)  # TODO

@testset "all" begin
    # n2n

    # ssdn
    denoiser = LaineSelfSupervisedDenoiser()  # make blindspot=true/false option
    # train directly on noisy data without known noise level
    train!(denoiser, noisy_data)
    # make noisy samples by adding Gaussian noise to clean samples and train on noisy-clean pairs
    # train!(denoiser, clean_data, :gaussian, σ_noise)
end

    """
    if obs_dim != :last
        dims = collect(1:ndims(data))
        # make the correct obs dim the last one, because that's what Flux.DataLoader assumes
        dims[obs_dim], dims[end] = dims[end], dims[obs_dim]
        data = permutedims(data, dims)
    end

    if ndims(data) == 3
        data = unsqueeze(data; dims=3)
    elseif ndims(data) != 4
        throw("need data to be 3D or 4D (ie with batch dim already")
    end
    """

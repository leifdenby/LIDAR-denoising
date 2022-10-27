using MLUtils
using Flux
using Test

abstract type AbstractDenoiser end
struct LinearDenoiser <: AbstractDenoiser
    model
end
Flux.@functor LinearDenoiser
LinearDenoiser() = LinearDenoiser(Conv((1, 1), 1 => 1, pad=0))

struct Noise2CleanDenoiser <: AbstractDenoiser
    model
end

function Noise2CleanDenoiser(n_layers)
    model = Chain(
        [Conv((2, 2), 1=>1) for i in 1:n_layers]...,
        [ConvTranspose((2, 2), 1=>1) for i in 1:n_layers]...
    )
    return Noise2CleanDenoiser(model)
end


struct Noise2NoiseDenoiser <: AbstractDenoiser end
struct LaineSelfSupervisedDenoiser <: AbstractDenoiser end

"""
general purpose data-loaders for denoising applications
for denoisers using more than one input/output `data` can be a tuple
"""
function create_dataloader(data; test_valid_fraction=0.9) where {N,T}
    data_train, data_test = splitobs(shuffleobs(data); at=test_valid_fraction)
    
    dl_train = Flux.DataLoader(data_train)
    dl_test = Flux.DataLoader(data_test)
    return dl_train, dl_test
end


function add_noise(data::T, noise_kind, σ_noise) where T
    if noise_kind == :gaussian
        return data + (σ_noise .* randn(T, size(data)))
    else
        throw(noise_kind)
    end
end

function loss(denoiser::LinearDenoiser, x_noisy, y_true)
    ŷ = denoiser.model(x_noisy)
    return Flux.Losses.mse(ŷ, y_true)
end

function device(denoiser)
    if Flux.params(denoiser)[1] isa CUDA.CuArray
        return gpu
    else
        return cpu
    end
end

"""
General-purpose training routine for denoiser
"""
function train!(denoiser::AbstractDenoiser, dl_train::MLUtils.DataLoader, dl_test::MLUtils.DataLoader)
    opt = Flux.Optimise.Descent()

    to_device = device(denoiser)

    Flux.train!(
        # general loss function with any batch contents calling denoiser specific loss
        (batch...) -> loss(denoiser, to_device(batch)...),
        Flux.params(denoiser.model),
        dl_train,
        opt
    )
end


"""
Train a denoiser that only requires a single noisy sample
"""
function train!(denoiser::AbstractDenoiser, noisy_data::AbstractArray{T,N}) where {T,N}
    dl_train, dl_valid = create_dataloader(noisy_data)
    train!(denoiser, dl_train, dl_test)
end

"""
Train a denoiser that requires clean and noisy inputs
"""
function train!(denoiser::AbstractDenoiser, noisy_data::AbstractArray{T,N}, clean_data::AbstractArray{T,N}) where {T,N}
    dl_train, dl_test = create_dataloader((noisy_data, clean_data))
    train!(denoiser, dl_train, dl_test)
end

"""
Train a denoiser on clean data with added noise of `noise_kind`
"""
function train!(denoiser::AbstractDenoiser, clean_data::AbstractArray{T,N}, noise_kind::Symbol, σ_noise) where {T,N}
    noisy_data = add_noise.(clean_data, noise_kind, T(σ_noise))
    dl_train, dl_test = create_dataloader((noisy_data, clean_data))
    train!(denoiser, dl_train, dl_test)
end


noisy_data = randn(Float32, (3, 3, 1, 2)); # HWCS = HeightWidthChannelSample
clean_data = randn(Float32, (3, 3, 1, 2)); # HWCS
σ_noise = 0.1
denoiser = LinearDenoiser() |> gpu

# train noisy-clean pairs without known noise
train!(denoiser, noisy_data, clean_data)
# make noisy samples by adding Gaussian noise to clean samples and train on noisy-clean pairs
train!(denoiser, clean_data, :gaussian, σ_noise)

@testset "all" begin
    # linear (won't work well)
    denoiser = LinearDenoiser()
    # train noisy-clean pairs without known noise
    train!(denoiser, noisy_data, clean_data)
    # make noisy samples by adding Gaussian noise to clean samples and train on noisy-clean pairs
    train!(denoiser, clean_data, :gaussian, σ_noise)

    # n2c
    denoiser = Noise2CleanDenoiser()
    # train noisy-clean pairs without known noise
    train!(denoiser, noisy_data, clean_data)
    # make noisy samples by adding Gaussian noise to clean samples and train on noisy-clean pairs
    train!(denoiser, clean_data, :gaussian, σ_noise)

    # n2n
    denoiser = Noise2NoiseDenoiser()
    # train by adding Gaussian noise with width σ_noise to each sample seen
    train!(denoiser, clean_data, :gaussian, σ_noise)
    # train on pair-wise noisy samples without known noise level
    # train!(denoiser, noisy_data1, noisy_data2)  # TODO

    # ssdn
    denoiser = LaineSelfSupervisedDenoiser()  # make blindspot=true/false option
    # train directly on noisy data without known noise level
    train!(denoiser, noisy_data)
    # make noisy samples by adding Gaussian noise to clean samples and train on noisy-clean pairs
    train!(denoiser, clean_data, :gaussian, σ_noise)
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

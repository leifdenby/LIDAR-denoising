"""
SupervisedDenoiser takes (noisy, clean) pairs during training
"""
abstract type SupervisedDenoiser <: AbstractDenoiser end

function loss(denoiser::SupervisedDenoiser, x_noisy, y_true)
    ŷ = denoiser.model(x_noisy)
    return Flux.Losses.mse(ŷ, y_true)
end

struct LinearDenoiser <: SupervisedDenoiser
    model
end
Flux.@functor LinearDenoiser
# constructor
LinearDenoiser() = LinearDenoiser(Conv((1, 1), 1 => 1, pad=0))
# inference call
(denoiser::LinearDenoiser)(noisy_data) = denoiser.model(noisy_data |> device(denoiser))


struct Noise2CleanDenoiser <: SupervisedDenoiser
    model
end
Flux.@functor Noise2CleanDenoiser

function Noise2CleanDenoiser(;n_layers::Int)
    model = Chain(
        [Conv((2, 2), 1=>1) for i in 1:n_layers]...,
        [ConvTranspose((2, 2), 1=>1) for i in 1:n_layers]...
    )
    return Noise2CleanDenoiser(model)
end
# inference call on trained model
(denoiser::Noise2CleanDenoiser)(noisy_data) = denoiser.model(noisy_data |> device(denoiser))
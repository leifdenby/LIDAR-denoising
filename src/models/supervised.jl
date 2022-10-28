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
LinearDenoiser(;activation=relu) = LinearDenoiser(Conv((1, 1), 1 => 1, activation, pad=0))
# inference call
function (denoiser::LinearDenoiser)(noisy_data::AbstractArray{T,4}) where T
     denoiser.model(noisy_data |> device(denoiser))
end


struct Noise2CleanDenoiser <: SupervisedDenoiser
    model
end
Flux.@functor Noise2CleanDenoiser

function Noise2CleanDenoiser(;n_layers::Int, activation=relu)
    model = Chain(
        [Conv((2, 2), 1=>1, activation) for i in 1:n_layers]...,
        [ConvTranspose((2, 2), 1=>1) for i in 1:n_layers]...
    )
    return Noise2CleanDenoiser(model)
end
# inference call on trained model
function (denoiser::Noise2CleanDenoiser)(noisy_data::AbstractArray{T,4}) where T
    denoiser.model(noisy_data |> device(denoiser))
end


"""
DnCNN: Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising
Zhang et al 2017
https://arxiv.org/abs/1608.03981

Based off https://github.com/SaoYan/DnCNN-PyTorch/blob/6b0804951484eadb7f1ea24e8e5c9ede9bea485b/models.py
"""
function DnCNN(;kernel_size=3, n_channels=1, n_layers=4, n_features=64, activation=relu)
    layers = Any[]
    c_filter = (kernel_size, kernel_size)

    push!(layers, Conv(c_filter, n_channels => n_features, relu, pad=SamePad(), bias=false))
    for _ in 1:(n_layers-2)
        push!(layers, Conv(c_filter, n_features => n_features, identity, pad=SamePad(), bias=false))
        push!(layers, BatchNorm(n_features))
        push!(layers, relu)
    end
    push!(layers, Conv(c_filter, n_features => n_channels, identity, pad=SamePad(), bias=false))

    LinearDenoiser(Chain(layers))
end
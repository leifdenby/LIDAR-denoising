"""
SupervisedDenoiser takes (noisy, clean) pairs during training
"""
struct SupervisedDenoiser <: AbstractDenoiser
    model
    learn_residual
end
Flux.@functor SupervisedDenoiser

# inference call on trained model
function (denoiser::SupervisedDenoiser)(noisy_data::AbstractArray{T,4}) where T
    z = denoiser.model(noisy_data |> device(denoiser)) |> cpu
    if denoiser.learn_residual
        return noisy_data + z
    else
        return z
    end
end

function loss(denoiser::SupervisedDenoiser, x_noisy, y_true)
    if denoiser.learn_residual
        z = denoiser.model(x_noisy)
        return Flux.Losses.mse(x_noisy + z, y_true)
    else
        ŷ = denoiser.model(x_noisy)
        return Flux.Losses.mse(ŷ, y_true)
    end
end


function Noise2CleanDenoiser(;n_layers::Int, activation=relu)
    model = Chain(
        [Conv((2, 2), 1=>1, activation) for i in 1:n_layers]...,
        [ConvTranspose((2, 2), 1=>1) for i in 1:n_layers]...
    )
    return SupervisedDenoiser(model, true)
end


"""
DnCNN: Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising
Zhang et al 2017
https://arxiv.org/abs/1608.03981

Comprised of successive convolutions with padding to keep a constant
height-width size and additional channels (n_features) in hidden layers.

Based off https://github.com/SaoYan/DnCNN-PyTorch/blob/6b0804951484eadb7f1ea24e8e5c9ede9bea485b/models.py
"""
function DnCNN(;kernel_size=3, n_channels=1, n_layers=5, n_features=64, activation=relu, train_residual=True)
    layers = Any[]
    c_filter = (kernel_size, kernel_size)

    push!(layers, Conv(c_filter, n_channels => n_features, relu, pad=SamePad(), bias=false))
    for _ in 1:(n_layers-2)
        push!(layers, Conv(c_filter, n_features => n_features, identity, pad=SamePad(), bias=false))
        push!(layers, BatchNorm(n_features))
        push!(layers, relu)
    end
    push!(layers, Conv(c_filter, n_features => n_channels, identity, pad=SamePad(), bias=false))

    SupervisedDenoiser(Chain(layers), train_residual)
end
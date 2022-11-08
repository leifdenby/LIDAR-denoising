
struct SelfSupervisedDenoiser <: AbstractDenoiser
    model
end
Flux.@functor SelfSupervisedDenoiser

"""
SSDN tries to predict the noisy data from itself using a blindspot UNet
"""
function loss(denoiser::SelfSupervisedDenoiser, x_noisy)
    Flux.Losses.mse(denoiser.model(x_noisy), x_noisy)
end

function (denoiser::SelfSupervisedDenoiser)(noisy_data::AbstractArray{T,4}) where {T}
    return denoiser.model(noisy_data |> device(denoiser)) |> cpu
end

function SelfSupervisedDenoiser(; n_features::Pair{Int,Int}= 1=>1, n_levels=3)
    model = BlindspotUNet(n_levels, n_features)
    return SelfSupervisedDenoiser(model)
end
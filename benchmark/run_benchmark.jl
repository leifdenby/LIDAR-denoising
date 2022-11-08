using LIDARdenoising
using LIDARdenoising.Models
using Plots
using MLUtils
using Flux
using Statistics

data = LIDARdenoising.load_data(joinpath(@__DIR__, "../sample-data/qv.x0.0.nc"))[:,:]
data = LIDARdenoising.load_data(joinpath(@__DIR__, "/mnt/speedy/projects/eurec4a-lidar/LES_data/rico.qv.nc"))

function random_crop(data::AbstractArray{T,2}, N::Int) where {T}
    nz, nx = size(data)
    @assert N <= nx && N <= nz
    i, k = rand(1:(nx-N)), rand(1:(nz-N))
    return data[k:k+N, i:i+N]
end

function random_crop(data::AbstractArray{T,3}, N::Int) where {T}
    nz, nx, ny = size(data)
    @assert N <= nx && N <= nz && N <= ny
    k0 = 3
    i, j = rand(1:(nx-N)), rand(1:(nz-N))
    return data[k0:k0+N-1, i:i+N-1, j]
end


N_samples = 10
N_size = 32
clean_samples = stack([random_crop(data, N_size) for i in 1:N_samples], dims=3);
σ_noise = 0.2 * Statistics.std(clean_samples) 
noisy_samples = LIDARdenoising.add_noise.(clean_samples, σ=σ_noise)
noisy_samples2 = LIDARdenoising.add_noise.(clean_samples, σ=σ_noise)

i_sample = rand(1:size(noisy_samples, 3))
plot(
    heatmap(clean_samples[:,:,i_sample]),
    heatmap(noisy_samples[:,:,i_sample]),
    heatmap(noisy_samples2[:,:,i_sample]),
    layout=(3,1),
    size=(400, 1000)
)
    

noisy_samples_normed = LIDARdenoising.normalize(noisy_samples)
noisy_samples_normed2 = LIDARdenoising.normalize(noisy_samples2, noisy_samples_normed.mean, noisy_samples_normed.std)
clean_samples_normed = LIDARdenoising.normalize(clean_samples, noisy_samples_normed.mean, noisy_samples_normed.std)

#denoiser = DnCNN(n_layers=10, train_residual=false) |> gpu

#denoiser = LinearDenoiser(Conv((3, 3), 1 => 1, identity, pad=SamePad()))
#denoiser = Noise2CleanDenoiser(n_layers=4) |> gpu
# denoiser = Noise2Noise() |> gpu
denoiser = LIDARdenoising.Models.SelfSupervisedDenoiser(n_levels=1) |> gpu
train_losses, valid_losses = train!(denoiser, noisy_samples_normed; n_epochs=5, learning_rate=1.0e-5)
#losses = train!(denoiser, clean_samples_normed, noisy_samples_normed; n_epochs=5, learning_rate=1.0e-3)

p_training = plot(train_losses, label="train")
plot!(p_training, valid_losses, label="validation")

function plot_sample(denoiser)
    i_sample = rand(1:size(noisy_samples, 3))
    clean_sample = clean_samples_normed[:,:,i_sample]
    noisy_sample = noisy_samples_normed[:,:,i_sample]
    pred_sample = denoiser(noisy_sample) |> cpu

    loss_orig = Flux.Losses.mse(clean_sample, noisy_sample)
    loss_pred = Flux.Losses.mse(clean_sample, pred_sample)
    plot(
        heatmap(clean_sample, aspectratio=1, title="clean"),
        heatmap(noisy_sample, aspectratio=1, title="noisy $(loss_orig)"),
        heatmap(noisy_sample - clean_sample, aspectratio=1, title="added noise"),
        heatmap(pred_sample, aspectratio=1, title="prediction $(loss_pred)")
    )
end

plot_sample(denoiser)

denoiser2 = LinearDenoiser(Conv((3, 3), 1 => 1, identity, pad=SamePad()))
#denoiser2 = deepcopy(denoiser)
denoiser2.model.weight[:] .= 1.0
denoiser2.model.weight[:] ./= sum(denoiser2.model.weight)
#denoiser2.model.weight[:] .= 0.0
#denoiser2.model.weight[2,2] = 1.0
denoiser2.model.bias[:] .= 0.0
denoiser2.model.weight
fieldnames(typeof(denoiser2.model))

mod = denoiser2.model


plot_sample(denoiser2)


function baseline_error(clean_samples_normed, noisy_samples_normed; model=nothing)
    dl = LIDARdenoising.Models.create_dataloader((noisy_samples_normed, clean_samples_normed))
    total_error = 0f0
    for (y_noisy, y_clean) in dl
        @show size(y_noisy) size(y_clean)
        if model === nothing
            total_error += Flux.Losses.mse(y_noisy, y_clean)
        else
            total_error += Flux.Losses.mse(model(y_noisy), y_clean)
        end
    end
    total_error / length(dl)
end

b_err = baseline_error(clean_samples_normed, noisy_samples_normed)
b_err = baseline_error(clean_samples_normed, noisy_samples_normed)
hline!(p_training, [b_err])
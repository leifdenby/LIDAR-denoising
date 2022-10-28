using LIDARdenoising
using LIDARdenoising.Models
using Plots
using MLUtils
using Flux

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
    i, j = rand(1:(nx-N)), rand(1:(nz-N))
    return data[1:N+1, i:i+N, j]
end


N_samples = 1000
N_size = 26
clean_samples = stack([random_crop(data, N_size) for i in 1:N_samples], dims=3);
σ_noise = 0.5
noisy_samples = LIDARdenoising.add_noise.(clean_samples, σ=σ_noise)

noisy_samples_normed = LIDARdenoising.normalize(noisy_samples)
clean_samples_normed = LIDARdenoising.normalize(clean_samples, noisy_samples_normed.mean, noisy_samples_normed.std)

denoiser = Noise2CleanDenoiser(n_layers=4) |> gpu
losses = train!(denoiser, clean_samples_normed, noisy_samples_normed; n_epochs=50, learning_rate=0.1)
plot(losses)
denoiser(noisy_samples[:,:,i_sample]) |> cpu

function plot_sample(denoiser)
    i_sample = rand(1:size(noisy_samples, 3))
    plot(
        heatmap(clean_samples_normed[:,:,i_sample], aspectratio=1),
        heatmap(noisy_samples_normed[:,:,i_sample], aspectratio=1),
        heatmap(denoiser(noisy_samples_normed[:,:,i_sample]) |> cpu, aspectratio=1)
    )
end

plot_sample(denoiser)

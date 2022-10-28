using LIDARdenoising
using LIDARdenoising.Models
using Plots
using MLUtils
using Flux


function (denoiser::LIDARdenoising.Models.Noise2CleanDenoiser)(data::AbstractMatrix{T}) where {T}
    data_batch_ch = data |> unsqueeze(;dims=3) |> unsqueeze(;dims=4)
    denoiser(data_batch_ch)[:,:]
end


data = LIDARdenoising.load_data(joinpath(@__DIR__, "../sample-data/qv.x0.0.nc"))[:,:]

function random_crop(data::AbstractArray{T,2}, N::Int) where {T}
    nx, ny = size(data)
    @assert N <= nx && N <= ny
    @show nx ny
    i, j = rand(1:(nx-N)), rand(1:(ny-N))
    return data[i:i+N, j:j+N]
end

clean_samples = stack([random_crop(data, 26) for i in 1:10], dims=3)
σ_noise = 0.5
noisy_samples = LIDARdenoising.add_noise.(clean_samples, σ=σ_noise)

denoiser = Noise2CleanDenoiser(n_layers=2)
for n in 1:10
    @info n
    train!(denoiser, clean_samples, noisy_samples)
end
denoiser(noisy_samples[:,:,i_sample]) |> cpu

heatmap(random_crop(data, 26), aspectratio=1)

i_sample = rand(1:size(noisy_samples, 3))
plot(
    heatmap(clean_samples[:,:,i_sample], aspectratio=1),
    heatmap(noisy_samples[:,:,i_sample], aspectratio=1),
    heatmap(denoiser(noisy_samples[:,:,i_sample]) |> cpu, aspectratio=1)
)
import PyPlot
import Statistics
import Logging: AbstractLogger, with_logger, @info
using Flux: gpu, cpu

include("normalization.jl")
include("noise.jl")
include("ncfile.jl")

function plot_example(data::GriddedData3D{T}, model, σ_noise, target=nothing; label="example plot") where {T}
    # y-index to plot
    i = 1
    y = data[:, i, :]
    data_mean, data_std = Statistics.mean(data), Statistics.std(data)
    y_normed = normalize(y, data_mean, data_std)
    y_normed_noisy = add_noise.(y_normed; σ = σ_noise)

    y_noisy = denormalize(NormalizedArray(y_normed_noisy, data_mean, data_std))

    add_batch_and_channel_dim(x::Array{T,2}) = reshape(x, Val(4))
    remove_batch_and_channel_dim(x_batch::Array{T,4}) = reshape(x_batch, Val(2))

    gpu_batch = gpu(add_batch_and_channel_dim(y_normed_noisy))
    ŷ_batch = model(gpu_batch)
    ŷ_normed = remove_batch_and_channel_dim(cpu(ŷ_batch))
    ŷ = denormalize(NormalizedArray(ŷ_normed, data_mean, data_std))

    x_grid = data.x
    z_grid = data.z

    fig, axes = PyPlot.subplots(3, 1, sharex=true, sharey=true, figsize=(10, 8))
    vmax = max(maximum(y), maximum(y_noisy), maximum(ŷ))
    vmin = min(minimum(y), minimum(y_noisy), minimum(ŷ))

    for (ax, data_, title) in zip(
        axes,
        [y, y_noisy, ŷ],
        ["y (from LES)", "x (added noise)", "ŷ (NN prediction)"]
    )
        p = ax.pcolormesh(x_grid, z_grid, data_, vmax=vmax, vmin=vmin)
        cb = PyPlot.colorbar(p, ax = ax)
        cb.set_label("water vap. [g/kg]")
        ax.set_xlabel("horz. dist. [m]")
        ax.set_ylabel("altitude [m]")
        ax.set_title(title)
    end

    PyPlot.tight_layout()
    PyPlot.suptitle("example", y=1.02)

    if target isa String
        PyPlot.savefig(target)
    elseif target isa AbstractLogger
        with_logger(target) do
            @info "$label plot" fig
        end
    end
end

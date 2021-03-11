import Plots
import Statistics
import Logging: AbstractLogger, with_logger, @info

include("normalization.jl")
include("noise.jl")
include("ncfile.jl")

function plot_example(data::GriddedData3D{T}, model, σ_noise, target=nothing) where T
    # y-index to plot
    i = 1
    y = data[:,i,:]
    data_mean, data_std = Statistics.mean(data), Statistics.std(data)
    y_normed = normalize(y, data_mean, data_std)
    y_normed_noisy = add_noise.(y_normed; σ=σ_noise)

    y_noisy = denormalize(NormalizedArray(y_normed_noisy, data_mean, data_std))

    add_batch_dim(x::Array{T,2}) = reshape(x, Val(3))
    remove_batch_dim(x_batch::Array{T,3}) = reshape(x_batch, Val(2))

    ŷ_normed = remove_batch_dim(model(add_batch_dim(y_normed_noisy)))
    ŷ = denormalize(NormalizedArray(ŷ_normed, data_mean, data_std))

    x_grid = data.x
    z_grid = data.z

    x_grid_conv = x_grid[6:end-7]
    z_grid_conv = z_grid[6:end-7]

    p1 = Plots.heatmap(x_grid, z_grid, y, title="y (from LES)", ylabel="altitude [m]", colorbar_title="water vap. [g/kg]")
    p2 = Plots.heatmap(x_grid, z_grid, y_noisy, title="x (added noise)", ylabel="altitude [m]", colorbar_title="water vap. [g/kg]")
    p3 = Plots.heatmap(x_grid_conv, z_grid_conv, ŷ, title="ŷ (NN prediction)", ylabel="altitude [m]", xlabel="horz. dist. [m]", colorbar_title="water vap. [g/kg]")
    figure = Plots.plot(p1, p2, p3, layout=(3,1); size=(1200, 800))

    if target isa String
        Plots.savefig(target)
    elseif target isa AbstractLogger
        with_logger(target) do
            @info figure
        end
    end
end

function plot_example(data::GriddedData3D{T}, model, σ_noise, target=nothing; label="example plot") where {T}
    # y-index to plot
    i = 1
    y = data[:, i, :]
    data_mean, data_std = Statistics.mean(data), Statistics.std(data)
    y_normed = normalize(y, data_mean, data_std)
    y_normed_noisy = add_noise.(y_normed; σ = σ_noise)

    y_noisy = denormalize(NormalizedArray(y_normed_noisy, data_mean, data_std))

    add_batch_and_channel_dim(x::Array{T,2}) = x |> unsqueeze(; dims=3) |> unsqueeze(; dims=4)
    remove_batch_and_channel_dim(x_batch::Array{T,4}) = x_batch[:,:,1,1]

    device_batch = add_batch_and_channel_dim(y_normed_noisy) |> _device
    ŷ_batch = _device(model)(device_batch) |> cpu
    ŷ_normed = remove_batch_and_channel_dim(ŷ_batch)
    ŷ = denormalize(NormalizedArray(ŷ_normed, data_mean, data_std))

    x_grid = data.x
    z_grid = data.z

    vmax = max(maximum(y), maximum(y_noisy), maximum(ŷ))
    vmin = min(minimum(y), minimum(y_noisy), minimum(ŷ))

    subplots = []
    for (data_, title) in zip(
        [y, y_noisy, ŷ],
        ["y (from LES)", "x (added noise)", "ŷ (NN prediction)"]
    )
        p_ = heatmap(
            x_grid, z_grid, data_, colorbar_title="water vap. [g/kg]",
            xlabel="horz. dist. [m]", ylabel="altitude [m]", title=title,
            vmin=vmin, vmax=vmax
        )
        push!(subplots, p_)
    end
    p = plot(subplots..., layout=(3, 3), size=(1200, 300))

    if target isa String
        savefig(p, target)
    elseif target isa AbstractLogger
        with_logger(target) do
            @info "$label plot" fig
        end
    end
end

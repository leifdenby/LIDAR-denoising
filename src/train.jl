
function train_model_on_data(model, data::AbstractArray{T,3}; n_epochs=2, batchsize=32, σ_noise=0.5, lr=0.5, logger=ConsoleLogger(), train_residual=true) where T <: AbstractFloat
    data_normed = normalize(data)
    model = model |> _device
    # if we are training on the CPU we make a copy of the model here so we don't update the original
    # I'm not sure how to update the on CPU model when training on the GPU, so I'm not sure how to
    # do an inplace update of the model when training on the GPU
    if _device == cpu
        model = deepcopy(model)
    end

    dl_train = DataLoaderLES(data_normed; batchsize=batchsize, σ_noise=σ_noise)

    if data isa GriddedData3D
        # plot with initial model
        plot_example(data, model, σ_noise, logger; label="starting model")
    end

    function lossfn(x, y)
        x_device = _device(x)
        if train_residual
            ϵ̂ = model(x_device)
            ŷ = x_device - ϵ̂
        else
            ŷ = model(x_device)
        end
        return Flux.Losses.mse(ŷ, _device(y))
    end

    function loss_all(dataloader)
        loss_dl = 0f0
        for valid_batch in dataloader
            x_valid, y_valid = valid_batch
            loss_dl += lossfn(x_valid, y_valid)
        end
        @info "loss" loss_dl/length(dataloader)

        if train_residual
            ϵ̂_mean_sum = 0f0
            ϵ̂_std_sum = 0f0
            for valid_batch in dataloader
                x_valid, y_valid = valid_batch
                ϵ̂_batch = model(_device(x_valid))
                ϵ̂_mean_sum += mean(ϵ̂_batch)
                ϵ̂_std_sum += std(ϵ̂_batch)
            end

            @info "mean(ϵ̂)" ϵ̂_mean_sum/length(dataloader)
            @info "std(ϵ̂)" ϵ̂_std_sum/length(dataloader)
        end
    end

    opt = Flux.Optimise.Descent(lr)

    # create a data-loader and callback to show loss on validation set
    dl_valid = DataLoaderLES(data_normed; batchsize=batchsize, σ_noise=σ_noise)

    # Train model on training dataset
    with_logger(logger) do
        for n in 1:n_epochs
            @info "epoch" n
            Flux.train!(lossfn, Flux.params(model), dl_train, opt)
            loss_all(dl_valid)
        end
    end

    if data isa GriddedData3D
        # plot with trained model
        plot_example(data, model, σ_noise, logger; label="trained model")
    end

    return model
end

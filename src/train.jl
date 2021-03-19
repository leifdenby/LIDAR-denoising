using Flux: gpu, cpu, @epochs, train!, params, throttle
import Flux
using Logging: @info, ConsoleLogger, with_logger
using CUDA


include("dataloader.jl")
include("model.jl")
include("normalization.jl")
include("plot.jl")


if has_cuda() # Check if CUDA is available
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

function train_model_on_data(model, data::AbstractArray{T,3}; n_epochs=2, batchsize=32, σ_noise=0.5, lr=0.5, logger=ConsoleLogger(), train_residual=true) where T <: AbstractFloat
    data_normed = normalize(data)
    model = model |> gpu

    dl_train = DataLoaderLES(data_normed; batchsize=batchsize, σ_noise=σ_noise)

    if data isa GriddedData3D
        # plot with initial model
        plot_example(data, model, σ_noise, logger; label="starting model")
    end

    function lossfn(x, y)
        if train_residual
            ŷ = x - model(gpu(x))
        else
            ŷ = model(gpu(x))
        end
        return Flux.Losses.mse(ŷ, gpu(y))
    end

    function loss_all(dataloader)
        loss_dl = 0f0
        for valid_batch in dataloader
            x_valid, y_valid = valid_batch
            loss_dl += lossfn(x_valid, y_valid)
        end
        @info "loss" loss_dl/length(dataloader)
    end

    opt = Flux.Optimise.Descent(lr)

    # create a data-loader and callback to show loss on validation set
    dl_valid = DataLoaderLES(data_normed; batchsize=batchsize, σ_noise=σ_noise)
    evalcb = () -> loss_all(dl_valid)

    # Train model on training dataset
    with_logger(logger) do
        @epochs n_epochs train!(lossfn, params(model), dl_train, opt; cb=throttle(evalcb, 10))
    end

    if data isa GriddedData3D
        # plot with trained model
        plot_example(data, model, σ_noise, logger; label="trained model")
    end

    return model
end

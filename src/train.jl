import Flux
using Logging: @info NullLogger


include("dataloader.jl")
include("model.jl")
include("normalization.jl")

function train_model_on_data(data::AbstractArray{T,3}; n_epochs=2, batchsize=32, σ_noise=0.5, lr=0.5, logger=NullLogger()) where T <: AbstractFloat
    data_normed = normalize(data)

    dl = DataLoaderLES(data_normed; batchsize=batchsize, nbatches=4, σ_noise=σ_noise)

    Nf = 5  # filter size in model convolutions
    Nc = 6  # number of "channels" in model convolutions
    model = build_model(Nf, Nc)

    opt = Flux.Optimise.Descent(lr)
    # TODO: calculate cropping in loss function from model
    lossfn(x, y) = Flux.Losses.mse(model(x), y[6:(end - 7), 6:(end - 7), :])

    # use a single batch to evaluate the model
    with(logger) do
        function evalcb()
            for valid_batch in DataLoaderLES(data; nbatches=1, batchsize=batchsize, σ_noise=σ_noise)
                x_valid, y_valid = valid_batch
                loss = lossfn(x_valid, y_valid)
                @info "loss" loss
            end
        end
        evalcb()

        Flux.@epochs n_epochs Flux.train!(lossfn, Flux.params(model), dl, opt; cb=Flux.throttle(evalcb, 10))
    end

    return model
end

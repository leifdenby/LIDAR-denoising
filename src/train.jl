import Flux


include("dataloader.jl")
include("model.jl")
include("normalization.jl")

function train_model_on_data(data::AbstractArray{T,3}; n_epochs=2, batchsize=32, σ_noise=0.5) where T <: AbstractFloat
    learning_rate = 0.5

    data_normed = normalize(data)

    dl = DataLoaderLES(data_normed; batchsize=batchsize, nbatches=4, σ_noise=σ_noise)

    Nf = 5  # filter size in model convolutions
    Nc = 6  # number of "channels" in model convolutions
    model = build_model(Nf, Nc)

    opt = Flux.Optimise.Descent(learning_rate)
    # TODO: calculate cropping in loss function from model
    lossfn(x, y) = Flux.Losses.mse(model(x), y[6:(end - 7), 6:(end - 7), :])

    # use a single batch to evaluate the model
    function evalcb()
        for valid_batch in DataLoaderLES(data; nbatches=1, batchsize=batchsize, σ_noise=σ_noise)
            x_valid, y_valid = valid_batch
            @show(lossfn(x_valid, y_valid))
        end
    end

    Flux.@epochs n_epochs Flux.train!(lossfn, Flux.params(model), dl, opt; cb=Flux.throttle(evalcb, 10))

    return model
end

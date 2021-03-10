import Flux: gpu, cpu
using Logging: @info, ConsoleLogger, with_logger
using CUDA


include("dataloader.jl")
include("model.jl")
include("normalization.jl")


if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

function train_model_on_data(data::AbstractArray{T,3}; n_epochs=2, batchsize=32, σ_noise=0.5, lr=0.5, logger=ConsoleLogger()) where T <: AbstractFloat
    data_normed = normalize(data)

    dl_train = DataLoaderLES(data_normed; batchsize=batchsize, nbatches=4, σ_noise=σ_noise)

    Nf = 5  # filter size in model convolutions
    Nc = 6  # number of "channels" in model convolutions
    model = build_model(Nf, Nc) |> gpu

	function lossfn(x, y)
	    y_hat = model(gpu(x))
	    return Flux.Losses.mse(y_hat, gpu(y)[6:(end - 7), 6:(end - 7), :])
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
    # TODO: calculate cropping in loss function from model

	# create a data-loader and callback to show loss on validation set
	dl_valid = DataLoaderLES(data_normed; nbatches=50, batchsize=batchsize, σ_noise=σ_noise)
	evalcb = () -> loss_all(dl_valid)

    # Train model on training dataset
    with_logger(logger) do
        Flux.@epochs n_epochs Flux.train!(lossfn, Flux.params(model), dl_train, opt; cb=Flux.throttle(evalcb, 10))
    end

    return model
end

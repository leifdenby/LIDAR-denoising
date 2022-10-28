"""
abstract type that all denoisers should base from
"""
abstract type AbstractDenoiser end

"""
inference on trained model with single example
"""
function (denoiser::AbstractDenoiser)(data::AbstractMatrix{T}) where {T}
    data_batch_ch = data |> unsqueeze(;dims=3) |> unsqueeze(;dims=4)
    denoiser(data_batch_ch)[:,:]
end

"""
Ensure that 2D input samples (shape HWS where S in the sample counter)
are given a channel dimension and that 3D inputs are unchanged
"""
function ensure_has_channel_dim(data::AbstractArray{T,N}) where {N,T}
    if N == 3
        # assume shape is HWS and channel dimension
        unsqueeze(data, dims=3)
    elseif N == 4
        # this is how shape should be HWCS
    else
        throw(N)
    end
end

function ensure_has_channel_dim(data_tuple::NTuple{N, AbstractArray{Nd,T}}) where {N,Nd,T}
    ensure_has_channel_dim.(data_tuple)
end

"""
general purpose data-loaders for denoising applications
for denoisers using more than one input/output `data` can be a tuple
"""
function create_dataloader(data; test_valid_fraction=0.9, batchsize=128) where {N,T}
    data_train, data_test = splitobs(shuffleobs(ensure_has_channel_dim(data)); at=test_valid_fraction)
    
    dl_train = Flux.DataLoader(data_train, batchsize=batchsize)
    dl_test = Flux.DataLoader(data_test, batchsize=batchsize)
    return dl_train, dl_test
end

"""
return the device (gpu/cpu) that a denoiser is stored on
"""
function device(denoiser::AbstractDenoiser)
    if Flux.params(denoiser)[1] isa CuArray
        return gpu
    else
        return cpu
    end
end

function early_stopping(losses; N_steps=8)
    N_total = length(losses)
    if N_total > N_steps+1
        maximum(losses[N_total-N_steps-1:N_total-1]) < losses[end]
    else
        false
    end
end

        

"""
General-purpose training routine for denoiser
"""
function train!(denoiser::AbstractDenoiser, dl_train::MLUtils.DataLoader, dl_test::MLUtils.DataLoader; n_epochs=10, learning_rate=1.0e-2)
    @show length(dl_train) length(dl_test)

    opt = Flux.Optimise.Descent(learning_rate)

    to_device = device(denoiser)
    # general loss function with any batch contents calling denoiser specific loss
    denoiser_loss(batch...) = loss(denoiser, to_device(batch)...)
    
    losses = []
    
    function evalcb()
        loss_dl = 0f0
        for valid_batch in dl_test
            loss_dl += denoiser_loss(valid_batch...)
        end
        avg_loss = loss_dl/length(dl_test)
        @info "loss" avg_loss
        push!(losses, avg_loss)
    end
    throttled_cb = Flux.throttle(evalcb, 1)

    for n in 1:n_epochs
        Flux.train!(
            denoiser_loss,
            Flux.params(denoiser.model),
            dl_train,
            opt,
            cb = throttled_cb
        )
        if early_stopping(losses)
            @info "stopped"
            break
        end
    end
    return losses
end

"""
Train a denoiser that only requires a single noisy sample
"""
function train!(denoiser::AbstractDenoiser, noisy_data::AbstractArray{T,N}; kwargs...) where {T,N}
    dl_train, dl_test = create_dataloader(noisy_data)
    train!(denoiser, dl_train, dl_test; kwargs...)
end

"""
Train a denoiser that requires clean and noisy inputs
"""
function train!(denoiser::AbstractDenoiser, noisy_data::AbstractArray{T,N}, clean_data::AbstractArray{T,N}; kwargs...) where {T,N}
    dl_train, dl_test = create_dataloader((noisy_data, clean_data))
    train!(denoiser, dl_train, dl_test; kwargs...)
end
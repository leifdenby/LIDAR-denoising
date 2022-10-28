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
function create_dataloader(data; test_valid_fraction=0.9) where {N,T}
    data_train, data_test = splitobs(shuffleobs(ensure_has_channel_dim(data)); at=test_valid_fraction)
    
    dl_train = Flux.DataLoader(data_train)
    dl_test = Flux.DataLoader(data_test)
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

"""
General-purpose training routine for denoiser
"""
function train!(denoiser::AbstractDenoiser, dl_train::MLUtils.DataLoader, dl_test::MLUtils.DataLoader)
    opt = Flux.Optimise.Descent(1.0e-4)

    to_device = device(denoiser)

    Flux.train!(
        # general loss function with any batch contents calling denoiser specific loss
        (batch...) -> loss(denoiser, to_device(batch)...),
        Flux.params(denoiser.model),
        dl_train,
        opt
    )
end

"""
Train a denoiser that only requires a single noisy sample
"""
function train!(denoiser::AbstractDenoiser, noisy_data::AbstractArray{T,N}) where {T,N}
    dl_train, dl_test = create_dataloader(noisy_data)
    train!(denoiser, dl_train, dl_test)
end

"""
Train a denoiser that requires clean and noisy inputs
"""
function train!(denoiser::AbstractDenoiser, noisy_data::AbstractArray{T,N}, clean_data::AbstractArray{T,N}) where {T,N}
    dl_train, dl_test = create_dataloader((noisy_data, clean_data))
    train!(denoiser, dl_train, dl_test)
end
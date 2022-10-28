"""
abstract type that all denoisers should base from
"""
abstract type AbstractDenoiser end

"""
general purpose data-loaders for denoising applications
for denoisers using more than one input/output `data` can be a tuple
"""
function create_dataloader(data; test_valid_fraction=0.9) where {N,T}
    data_train, data_test = splitobs(shuffleobs(data); at=test_valid_fraction)
    
    dl_train = Flux.DataLoader(data_train)
    dl_test = Flux.DataLoader(data_test)
    return dl_train, dl_test
end

"""
return the device (gpu/cpu) that a denoiser is stored on
"""
function device(denoiser)
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
    opt = Flux.Optimise.Descent()

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
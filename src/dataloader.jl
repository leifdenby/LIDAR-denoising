
function create_dataloader(data::AbstractArray{T,N}; σ=0.1, obs_dim=:last, test_valid_fraction=0.9) where {N,T}
    @show typeof(data) T
    if obs_dim != :last
        dims = collect(1:ndims(data))
        # make the correct obs dim the last one, because that's what Flux.DataLoader assumes
        dims[obs_dim], dims[end] = dims[end], dims[obs_dim]
        data = permutedims(data, dims)
    end

    if ndims(data) == 3
        data = unsqueeze(data; dims=3)
    elseif ndims(data) != 4
        throw("need data to be 3D or 4D (ie with batch dim already")
    end
    
    data_train, data_test = splitobs(shuffleobs(data); at=test_valid_fraction)
    
    dl_train = Flux.DataLoader(data_train)
    dl_test = Flux.DataLoader(data_test)
    return dl_train, dl_test
end
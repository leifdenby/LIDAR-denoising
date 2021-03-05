include("noise.jl")


struct DataLoaderLES{T}
    data::AbstractArray{T,3}
    batchsize::Int
    nbatches::Int
    σ_noise::T
end

function DataLoaderLES(data::AbstractArray{T,3}; batchsize = 100, nbatches = 1, σ_noise = 0.5) where T
    batchsize > 0 || throw(ArgumentError("Need positive batchsize"))
    nbatches > 0 || throw(ArgumentError("Need positive nbatches"))

    DataLoaderLES(data, batchsize, nbatches, T(σ_noise))
end

function _getSlice(data::AbstractArray{D,3}, i::Int) where {D}
    # TODO: add random offset here and pick a random axis between y and x-axis
    data[:, i, :]
end

# required functions to support iteration
Base.length(dl::DataLoaderLES) = dl.nbatches


function Base.iterate(d::DataLoaderLES, i = 0)
    if i >= d.nbatches
        return nothing
    end

    indecies = rand(1:size(d.data)[2], d.batchsize)
    y_batch = cat([_getSlice(d.data, idx) for idx in indecies]...; dims = 3)
    # In predicition the convolutions mean we can't predict the edge
    x_batch = add_noise.(y_batch; σ = d.σ_noise)

    batch = (x_batch, y_batch)
    return (batch, i + 1)
end

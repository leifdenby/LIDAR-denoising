include("noise.jl")


struct DataLoaderLES{D}
    data::AbstractArray{D,3}
    batchsize::Int
    nbatches::Int
end

function DataLoaderLES(data; batchsize = 100, nbatches = 1)
    batchsize > 0 || throw(ArgumentError("Need positive batchsize"))
    nbatches > 0 || throw(ArgumentError("Need positive nbatches"))

    DataLoaderLES(data, batchsize, nbatches)
end

function _getSlice(data::AbstractArray{D,3}, i::Int) where {D}
    # TODO: add random offset here and pick a random axis between y and x-axis
    data[:, i, :]
end

# required functions to support iteration
Base.length(dl::DataLoaderLES) = dl.nbatches


function Base.iterate(d::DataLoaderLES, i = 0; σ_noise = 0.5)
    if i >= d.nbatches
        return nothing
    end

    indecies = rand(1:size(d.data)[2], d.batchsize)
    y_batch = cat([_getSlice(d.data, idx) for idx in indecies]...; dims = 3)
    # In predicition the convolutions mean we can't predict the edge
    x_batch = add_noise.(y_batch; σ = σ_noise)

    batch = (x_batch, y_batch)
    return (batch, i + 1)
end

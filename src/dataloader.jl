using Flux: gpu, unsqueeze
using Random: shuffle
include("noise.jl")


mutable struct DataLoaderLES{T}
    data::AbstractArray{T,3}
    batchsize::Int
    σ_noise::T
    data_order::Vector{Int}
end

function DataLoaderLES(data::AbstractArray{T,3}; batchsize = 100, σ_noise = 0.5) where T
    batchsize > 0 || throw(ArgumentError("Need positive batchsize"))

    data_order = shuffle(1:size(data)[2])
    DataLoaderLES(data, batchsize, T(σ_noise), data_order)
end

function _getSlice(data::AbstractArray{D,3}, i::Int) where {D}
    # TODO: add random offset here and pick a random axis between y and x-axis
    data[:, i, :]
end

# required functions to support iteration
Base.length(dl::DataLoaderLES) = size(dl.data)[2]


function Base.iterate(dl::DataLoaderLES, i = 0)
    if i == 0
        dl.data_order = shuffle(1:length(dl))
    end

    if i >= length(dl)
        return nothing
    end

    indecies = rand(1:size(dl.data)[2], dl.batchsize)
    srcdata_batch = cat([_getSlice(dl.data, dl.data_order[idx]) for idx in indecies]...; dims = 3)
    # add channel dimension
    y_batch = unsqueeze(srcdata_batch, 3)
    # In predicition the convolutions mean we can't predict the edge
    x_batch = add_noise.(y_batch; σ = dl.σ_noise)

    batch = (x_batch, y_batch)
    return (batch, i + 1)
end

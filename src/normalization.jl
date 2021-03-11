import Statistics

"""
Create a "normalized array" which behaves like a normal array, but carries
around a value for the mean and standard div it was normalized with so that the
original values can be reconstructed.
"""
struct NormalizedArray{T,N} <: AbstractArray{T,N}
    values::AbstractArray{T,N}
    mean::T
    std::T
end

Base.size(x::NormalizedArray) = size(x.values)
Base.getindex(x::NormalizedArray, key...) = Base.getindex(x.values, key...)

function normalize(x::AbstractArray{T,N}, x_mean::T, x_std::T) where {T,N}
    NormalizedArray((x .- x_mean) ./ x_std, x_mean, x_std)
end

function normalize(x::AbstractArray{T,N}) where {T,N}
    normalize(x, Statistics.mean(x), Statistics.std(x))
end

function denormalize(x_normed::NormalizedArray{T,N}, x_mean::T, x_std::T) where {T,N}
    x_normed.values .* x_std .+ x_mean
end

function denormalize(x_normed::NormalizedArray{T,N}) where {T,N}
    denormalize(x_normed, x_normed.mean, x_normed.std)
end

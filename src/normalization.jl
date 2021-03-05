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

function normalize(x::AbstractArray{T,N}) where {T,N}
    x_mean = Statistics.mean(x)
    x_std = Statistics.std(x)
    NormalizedArray((x .- x_mean) ./ x_std, x_mean, x_std)
end

function denormalize(x_normed::NormalizedArray{T,N}) where {T,N}
    x_normed.values .* x_normed.std .+ x_normed.mean
end

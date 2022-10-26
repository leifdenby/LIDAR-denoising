"""
Extend 90-degree rotations of matrices to apply to axis subsets of a N-dimensional vector
Based on discussion on https://discourse.julialang.org/t/rotr90-of-a-cuda-cuarray/88308
"""
function Base.rotr90(v::AbstractArray{T, N}; axes::NTuple{2, Int}=(1,2)) where {T, N}
    if N == 2
        return rotr90(v)
    end
    # if rotating on the first two axes, the below would be the same as
    # permutedims(v[end:-1:begin, :, :, :], (2,1,3,4))
    perm = collect(1:N)
    axes = collect(axes)  # create list from the axes tuple
    perm[reverse(axes)] = perm[axes]  # create new axes permutations
    permutedims(
        selectdim(v, axes[1], lastindex(v, axes[1]):-1:firstindex(v, axes[1])),
        perm
    )
end
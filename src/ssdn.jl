export HalfPlane, halfplane_offset

using Flux: pad_zeros, MaxPool, Conv, SamePad


"""
    HalfPlane(op, dim)

Turn a Flux operator (e.g. MaxPool or Conv) into a half-plane operation along
dimension `dim`
"""
struct HalfPlane
    op
    dim::Integer
end

"""For a HalfPlaneOp work out what offset is needed to make it half-plane"""
function halfplane_offset(stencil_width::Integer)
    n_offset = stencil_width ÷ 2
    if stencil_width % 2 == 1
        n_offset += 1
    end
    return n_offset
end

# calculate half-plane offsets for different operators
function halfplane_offset(op::Conv, dim::Integer)
    stencil_width = size(op.weight, 1)
    return halfplane_offset(stencil_width)
end

function halfplane_offset(op::MaxPool, dim::Integer)
    stencil_width = size(op.k, 1)
    return halfplane_offset(stencil_width)
end

halfplane_offset(c::HalfPlane) = halfplane_offset(c.op, c.dim)


"""calculate halfplane offset"""
function halfplane_offset(c::HalfPlane, x::AbstractArray{T}) where T
    n_offset = halfplane_offset(c)

    # pad with zeros by the same amount at the beginning
    x_padded = pad_zeros(x, (n_offset, 0), dims=[c.dim])

    # and crop of last n_offset elements (same as cropping to original size) in
    # c.dim direction
    selectdim(x_padded, c.dim, 1:size(x, c.dim))
end

"""
    (c::HalfPlane)(x)
Apply half-plane operation to x
"""
function (c::HalfPlane)(x::AbstractArray{T}) where T
    return c.op(halfplane_offset(c, x))
end

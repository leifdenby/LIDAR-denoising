export HalfPlaneOp, DummyHalfPlaneOp, HalfPlaneConv2D, HalfPlaneMaxPool2D
export offset, calc_padding

using Flux: pad_zeros, MaxPool, Conv, SamePad


"""
    HalfPlane(op, dim)

Turn a Flux operator (e.g. MaxPool or Conv) into a half-plane operation along
dimension `dim`
"""
struct HalfPlane
    op
    dim
end


function offset(c::HalfPlane)
    """For a HalfPlaneOp work out what offset is needed to make it half-plane"""
    stencil_width = c.filter[c.dim]
    n_offset = stencil_width ÷ 2
    if stencil_width % 2 == 1
        n_offset += 1
    end
    return n_offset
end


function halfplane_offset(c::HalfPlaneOp, x::AbstractArray{T}) where T
    """apply offset required for HalfPlaneOp `c` to `x`"""
    n_offset = offset(c)

    # pad with zeros by the same amount at the beginning
    x_padded = pad_zeros(x, (n_offset, 0), dims=[c.dim])

    # and crop of last n_offset elements (same as cropping to original size) in
    # c.dim direction
    selectdim(x_padded, c.dim, 1:size(x, c.dim))
end






abstract type HalfPlaneOp end

function offset(c::HalfPlaneOp, x::AbstractArray{T}) where T
    """apply offset required for HalfPlaneOp `c` to `x`"""
    n_offset = offset(c)

    # pad with zeros by the same amount at the beginning
    x_padded = pad_zeros(x, (n_offset, 0), dims=[c.dim])

    # and crop of last n_offset elements (same as cropping to original size) in
    # c.dim direction
    selectdim(x_padded, c.dim, 1:size(x, c.dim))
end

function offset(c::HalfPlaneOp)
    """For a HalfPlaneOp work out what offset is needed to make it half-plane"""
    stencil_width = c.filter[c.dim]
    n_offset = stencil_width ÷ 2
    if stencil_width % 2 == 1
        n_offset += 1
    end
    return n_offset
end

struct DummyHalfPlaneOp <: HalfPlaneOp
    filter::Tuple{Integer, Integer}
    dim::Integer
end


# Half-plane convolutions

struct HalfPlaneConv2D <: HalfPlaneOp
    filter::Tuple{<:Integer,<:Integer}
    dim
    conv
end

function HalfPlaneConv2D(
    filter::Tuple{<:Integer,<:Integer},
    ch::Pair{<:Integer,<:Integer},
    activation::Function;
    dim=1,
    pad=0,
)
    conv = Conv(filter, ch, activation, pad=pad)
    HalfPlaneConv2D(size(conv.weight)[1:2], dim, conv)
end


function (c::HalfPlaneConv2D)(x::AbstractArray{T}) where T
    """
    apply half-plane convolution. If the size of the convolution
    in the half-plan direction is an odd number we ensure that
    the centre of the convolution filter is also excluded
    """
    x_offset = offset(c, x)
    return c.conv(x_offset)
end

# half-plane max-pooling


struct HalfPlaneMaxPool2D <: HalfPlaneOp
    filter::Tuple{Integer,Integer}
    dim::Integer
end


function (c::HalfPlaneMaxPool2D)(x::AbstractArray{T}) where T
    # in Laine et al 2019 they apply a crop before padding with zeros so we'll do the same here
    d_xlen = size(x, c.dim)

    x_crop = copy(x)

    if c.dim == 1
        x_crop = @view x[1:d_xlen-1,:,:,:]
    elseif c.dim == 2
        x_crop = @view x[:,1:d_xlen-1,:,:]
    else
        throw("Not implemented")
    end
    n_offset = offset(c)

    # TODO: Laine et al 2019 use 2x2 max-pooling which results in a offset of
    # 1, I need to work out what to do to generalise this
    # TODO: Laine et al 2019 use "same"-padding, but in Flux if the shape has
    # an odd size in a dimension then zeros are added to the beginning in that
    # dimension which is also where we're adding our own zero-padding. Need to
    # double-check what happens in tensor flow
    if n_offset != 1
        throw(MethodError(n_offset))
    end

    x_offset = offset(c, x_crop)

    pad = SamePad()
    mp = MaxPool(c.filter, pad=)(x_offset)

    x_offset = offset(c, x)
    return c.conv(x_offset)

    return x_mp
end

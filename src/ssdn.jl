export HalfPlaneOp, DummyHalfPlaneOp, HalfPlaneConv2D, HalfPlaneMaxPool2D
export offset, make_ssdn

using Flux: pad_zeros, MaxPool, Conv, SamePad
using Flux: leakyrelu, ConvTranspose, Chain, SkipConnection


abstract type HalfPlaneOp end

function offset(c::HalfPlaneOp, x::AbstractArray{T}) where T
    """apply offset required for HalfPlaneOp `c` to `x`"""
    n_offset = offset(c)
    pad_zeros(x, (n_offset, 0), dims=[c.dim])
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
    n_offset = offset(c)
    x_conv = c.conv(x_offset)

    pad = c.conv.pad[c.dim]
    i_start = 1 + pad
    i_end = size(x_conv, c.dim) - n_offset

    if c.dim == 1
        return x_conv[i_start:i_end,:,:,:]
    elseif c.dim == 2
        return x_conv[:,i_start:i_end,:,:]
    else
        throw("Not implemented")
    end
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
    x_mp = MaxPool(c.filter, pad=0)(x_offset)

    return x_mp
end



function make_ssdn(nc_in, n_levels, n_out_features)
    lower_level = undef
    # convolution stencil size, same as Laine 2019
    w = 3
    # number of hidden layers after first convolution
    nc_hd = 4
    # activation function
    act = leakyrelu

    for n in 1:n_levels
        layers = Vector{Any}([
            # offset convolution with padding
            HalfPlaneConv2D((w,w), nc_hd => nc_hd, act),
            # max-pool coarsening
            HalfPlaneMaxPool2D((2,2), 1)
        ])
        
        if n > 1
            append!(layers, [
                # skip-connetion with concatenation across lower level
                SkipConnection(lower_level, (mx, x) -> cat(mx, x, dims=3))
                # conv-transpose up in lower level doubled number channels
                # so with concat that is (2+1)*nc_hd. Conv with padding
                # to return to nc_hd channels
                HalfPlaneConv2D((w,w), nc_hd*3 => nc_hd, act)
            ])
        end

        append!(layers,
            [
            # offset convolution with padding
            HalfPlaneConv2D((w,w), nc_hd => nc_hd, act),
            # conv-tranpose up, doubling number of channels
            ConvTranspose((2,2), nc_hd => 2*nc_hd, stride=2)
        ])

        lower_level = Chain(layers...)
    end

    # TODO: add in image stacking and unstacking etc
    model = Chain(
        HalfPlaneConv2D((w,w), nc_in => nc_hd, act),
        lower_level,
        HalfPlaneConv2D((w,w), 2*nc_hd => n_out_features, act)
    )
    return model
end

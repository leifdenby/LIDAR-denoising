export HalfPlane, halfplane_offset, SSDN

using Flux: pad_zeros, MaxPool, Conv, SamePad
using Flux: leakyrelu, ConvTranspose, Chain, SkipConnection


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


function SSDN(nc_in, n_levels, n_out_features)
    lower_level = undef

    # convolution stencil size
    w = 3
    # number of hidden layers immediately after input
    nc_hd = 1
    # activation function
    act = leakyrelu

    for n in 1:n_levels
        layers = Vector{Any}([
            # offset convolution with padding
            HalfPlane( Conv((w,w), nc_hd => nc_hd, act, pad=SamePad()), 1),
            # max-pool coarsening
            HalfPlane( MaxPool((2,2), pad=SamePad()), 1)
        ])

        if n > 1
            append!(layers, [
                # skip-connetion with concatenation across lower level
                SkipConnection(lower_level, (mx, x) -> cat(mx, x, dims=3))
                # conv-transpose up in lower level doubled number channels
                # so with concat that is (2+1)*nc_hd. Conv with padding
                # to return to nc_hd channels
                HalfPlane( Conv((w,w), nc_hd*3 => nc_hd, act, pad=SamePad()), 1)
            ])
        end

        append!(layers,
            [
            # offset convolution with padding
            HalfPlane( Conv((w,w), nc_hd => nc_hd, act, pad=SamePad()), 1),
            # conv-tranpose up, doubling number of channels
            ConvTranspose((2,2), nc_hd => 2*nc_hd, stride=2)
        ])

        lower_level = Chain(layers...)
    end

    # TODO: add in image stacking and unstacking etc
    model = Chain(
        HalfPlane( Conv((w,w), nc_in => nc_hd, act, pad=SamePad()), 1),
        lower_level,
        HalfPlane( Conv((w,w), 2*nc_hd => n_out_features, act, pad=SamePad()), 1)

    )
    return model
end

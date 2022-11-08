export HalfPlane, halfplane_offset, SSDN, rotated_stack, unrotate, HalfPlaneShiftOp


"""
    HalfPlane(op, dim)

Turn a Flux operator (e.g. MaxPool or Conv) into a half-plane operation along
dimension `dim`
"""
struct HalfPlane
    op
    dim::Integer
end

Flux.@functor HalfPlane

"""For a HalfPlaneOp work out what offset is needed to make it half-plane"""
function halfplane_offset(stencil_width::Integer)
    n_offset = stencil_width ÷ 2
    # NB: Laine et al 2019 has the offset for stencil width of 3 be 1 here, but
    # I think that is incorrect is that would still include the central
    # pixel, but I'm leaving this for now
    # if stencil_width % 2 == 1
        # n_offset += 1
    # end
    return n_offset
end

struct HalfPlaneShiftOp
    offset::Integer
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

function halfplane_offset(op::HalfPlaneShiftOp, dim::Integer)
    return op.offset
end

HalfPlaneShiftOp(offset::Integer, dim::Integer) = HalfPlane(HalfPlaneShiftOp(offset), dim)


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


function halfplane_offset_reverse(c::HalfPlane, x::AbstractArray{T}) where T
    n_offset = halfplane_offset(c)
    # pad with zeros by the same amount at the beginning
    x_padded = pad_zeros(x, (0, n_offset), dims=[c.dim])

    return selectdim(x_padded, c.dim, n_offset:size(x, c.dim) + n_offset)
end

"""
    (c::HalfPlane)(x)
Apply half-plane operation c to x

NB: Laine 2019 applies crop after convolution, so that the order of operations in Laine
is 1) pad, 2) conv (with samepad) and 3) crop, here we do pad and crop before the wrapped
operation
"""
function (c::HalfPlane)(x::AbstractArray{T}) where T
    # apply halfplane padding
    x_offset = halfplane_offset(c, x)

    # need to create a copy here (and not just a view) otherwise
    # executation requires scalar indexing which isn't allowed on the GPU
    x_conv = c.op(copy(x_offset))

    return x_conv
    n_offset = halfplane_offset(c)
    @show size(x_conv) size(x) n_offset size(x, c.dim)
    return selectdim(x_conv, c.dim, n_offset:size(x, c.dim) + n_offset)
end

function (c::HalfPlaneShiftOp)(x::AbstractArray{T}) where T
    # don't do anything, this is a noop, pad and shift is done in (c::HalfPlane)(x::AbstractArray{T}) call
    return x
end

"""
    rotate_hw(x)

Rotate a batch in the-xy plane by `angle` (only values divisible by 90)
"""
# function rotate_hw(x::AbstractArray{T,4}, angle) where T
    # if angle == 0
        # return x
    # else
        # return mapslices(x_ -> rotr90(x_, angle ÷ 90), x, dims=[1,2])
    # end
# end

"""rotated_stack(x)

Created rotated copies rotated by 0, 90, 180 and 270 in the batch dimension
"""
rotated_stack(x) = cat([rotr90(x, a ÷ 90, axes=(1,2)) for a in [0, 90, 180, 270]]...; dims=4)

# NB: I think this is creating copies, I'm sure this could be improved by
# rotating inplace in memory
unrotate = Chain(
    # split along batch dimension
    x -> [selectdim(x, 4, i) for i in 1:4],
    # rotate each part back by correct amount
    x -> [rotr90(v[:,:,:,:], a ÷ 90, axes=(1,2)) for (v, a) in zip(x, [0, 270, 180, 90])],
    # and concat again along channel dimension
    x -> cat(x..., dims=3)
)

# Base.split(x::AbstractArray{T}, d) where T = [selectdim(x, i, d) for i in 1:size(x, d)]


"""
Create a "Blindspot" UNet where the receptive field of each output pixels
excludes the pixel at the same position in the input

nc_hr: number of hidden layers immediately after input
"""
function BlindspotUNet(n_levels, channels; act=leakyrelu, nc_hd=4)
    if n_levels < 0
        throw("n_levels must non-zero")
    end
    # make "lower" level in the Unet be a no-op by default
    lower_level = Conv((1, 1), nc_hd => 2*nc_hd)

    # convolution stencil size
    w = 3

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
        # add four rotated copies of input images along batch dimension
        rotated_stack,
        HalfPlane( Conv((w,w), channels.first => nc_hd, act, pad=SamePad()), 1),
        lower_level,
        HalfPlaneShiftOp(1, 1),
        unrotate,
        # after unrotating each half-plane goes into a different channel
        # so we gain x4 the number of channels. After this we no longer need
        # half-plane convolutions
        Conv((1,1), 4*2*nc_hd => 2*nc_hd, act, pad=SamePad()),
        Conv((1,1), 2*nc_hd => channels.second, act, pad=SamePad()),
    )
    return model
end

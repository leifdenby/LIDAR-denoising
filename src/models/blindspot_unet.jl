export HalfPlane, halfplane_offset, SSDN, rotate_hw, rotated_stack, unrotate


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


function halfplane_offset_reverse(c::HalfPlane, x::AbstractArray{T}) where T
    n_offset = halfplane_offset(c)

    # pad with zeros by the same amount at the beginning
    x_padded = pad_zeros(x, (0, n_offset), dims=[c.dim])

    # and crop of last n_offset elements (same as cropping to original size) in
    # c.dim direction
    selectdim(x_padded, c.dim, n_offset:size(x, c.dim) + n_offset)
end

"""
    (c::HalfPlane)(x)
Apply half-plane operation c to x
"""
function (c::HalfPlane)(x::AbstractArray{T}) where T
    # need to create a copy here (and not just a view) otherwise
    # executation requires scalar indexing which isn't allowed on the GPU
    return c.op(copy(halfplane_offset(c, x)))
end

"""
    rotate_hw(x)

Rotate a batch in the-xy plane by `angle` (only values divisible by 90)
"""
function rotate_hw(x::AbstractArray{T,4}, angle) where T
    if angle == 0
        return x
    else
        return mapslices(x_ -> rotr90(x_, angle ÷ 90), x, dims=[1,2])
    end
end

"""rotated_stack(x)

Created rotated copies rotated by 0, 90, 180 and 270 in the batch dimension
"""
rotated_stack(x) = cat([rotate_hw(x, a) for a in [0, 90, 180, 270]]...; dims=4)

# NB: I think this is creating copies, I'm sure this could be improved by
# rotating inplace in memory
unrotate = Chain(
    # split along batch dimension
    x -> [selectdim(x, 4, i) for i in 1:4],
    # rotate each part back by correct amount
    x -> [rotate_hw(v[:,:,:,:], a) for (v, a) in zip(x, [0, 270, 180, 90])],
    # and concat again along channel dimension
    x -> cat(x..., dims=3)
)

# Base.split(x::AbstractArray{T}, d) where T = [selectdim(x, i, d) for i in 1:size(x, d)]


"""
Create a "Blindspot" UNet where the receptive field of each output pixels
excludes the pixel at the same position in the input
"""
function BlindspotUNet(n_levels, channels; act=leakyrelu, n_hd=1)
    # make "lower" level in the Unet be a no-op by default
    lower_level = Conv((1, 1), channels)

    # convolution stencil size
    w = 3
    # number of hidden layers immediately after input
    nc_hd = 1
    # activation function

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
        # TODO: need inverse of half-plane offset here
        unrotate,
        # after unrotating each half-plane goes into a different channel
        # so we gain x4 the number of channels. After this we no longer need
        # half-plane convolutions
        Conv((1,1), 4*2*nc_hd => 2*nc_hd, act, pad=SamePad()),
        Conv((1,1), 2*nc_hd => channels.last, act, pad=SamePad()),
    )
    return model
end
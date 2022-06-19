using Test
using LIDARdenoising: HalfPlane, halfplane_offset, SSDN, rotate_hw
using Flux: MaxPool, SamePad, identity, Conv


@test_skip @testset "half-plane generics" begin
    nx = 3
    ny = 3
    ns = 3  # size of filter is 3x3
    no = (ns + 1) ÷ 2  # expected offset is 2 to avoid the center
    ## half-plane operations
    m = Conv((ns,ns), 1=>1, identity)
    d_op = HalfPlane(m, 1)
    v1 = randn(Float32, (nx, ny, 1, 1))
    @test halfplane_offset(d_op) == no
    # check that the offsetting is being done correctly
    @test all(selectdim(halfplane_offset(d_op, v1), 1, 1:no) .== 0)
end

@test_skip @testset "half-plane convolutions nx=$nx" for nx in [3, 5]
    ny = 3
    ns = 3  # size of filter is 3x3

    # check that that only the values in the half-plane contribute to the final value
    v2 = randn(Float32, (nx, ny, 1, 1))
    offset_conv_xdim = HalfPlane( Conv((ns,ns),1=>1, identity, pad=0), 1)
    offset_conv_xdim_pad1 = HalfPlane( Conv((ns,ns),1=>1, identity, pad=1), 1)
    offset_conv_xdim_samepad = HalfPlane( Conv((ns,ns),1=>1, identity, pad=SamePad()), 1)
    offset_conv_ydim = HalfPlane( Conv((ns,ns),1=>1, identity, pad=0),2 )
    # all operators have same stencil size, so we can just calculate the offset once
    no = halfplane_offset(offset_conv_xdim)

    # change the values for the values that sholdn't effect the result
    v2_xmod = copy(v2)
    v2_xmod[nx-no+1:nx,:,:,:] .= 1
    v2_ymod = copy(v2)
    v2_ymod[:,ny-no+1:ny,:,:] .= 1 

    @test offset_conv_xdim(v2) == offset_conv_xdim(v2_xmod)
    @test offset_conv_ydim(v2) == offset_conv_ydim(v2_ymod)
    @test offset_conv_xdim_pad1(v2) == offset_conv_xdim_pad1(v2_xmod)
    @test offset_conv_xdim_samepad(v2) == offset_conv_xdim_samepad(v2_xmod)

    # next we'll check the shapes against just applying the underlying
    # convolutions without offsetting (which creates the half-planes)
    @test size(offset_conv_xdim(v2)) == size(offset_conv_xdim.op(v2))
    @test size(offset_conv_xdim_pad1(v2)) == size(offset_conv_xdim_pad1.op(v2))
    @test size(offset_conv_xdim_samepad(v2)) == size(offset_conv_xdim_samepad.op(v2))
end

@test_skip @testset "half-plane max-pool" begin
    nx = 5
    ny = 4
    ns = 2  # size of filter is 2x2 as in Laine et al 2019

    offset_mp_xdim = HalfPlane( MaxPool((ns, ns), pad=SamePad()), 1)
    offset_mp_ydim = HalfPlane( MaxPool((ns, ns), pad=SamePad()), 2)
    # all operators have same filter width
    no = halfplane_offset(offset_mp_xdim)
    @test no == 1

    # add batch and channel dimensions
    ex(x) = x[:,:,:,:]

    # can't compare against the max-pool values with/without padding because
    # the values will be shifted and the stride will fall on different values,
    # instead I'll just construct arrays which have the padding already applied

    v3 = repeat(1:nx, 1, ny) |> ex
    v3_padded_xdim = vcat(repeat([0], 1, ny), v3[1:(nx-1), :]) |> ex
    v3_padded_ydim = hcat(repeat([0], nx, 1), v3[:, 1:(ny-1)]) |> ex

    @test all( halfplane_offset(offset_mp_xdim, v3) .== v3_padded_xdim )
    @test all( halfplane_offset(offset_mp_ydim, v3) .== v3_padded_ydim )

    @test offset_mp_xdim(v3) == offset_mp_xdim.op(v3_padded_xdim)
    @test offset_mp_ydim(v3) == offset_mp_ydim.op(v3_padded_ydim)
end

@testset "rotation" begin
    a = randn(Float32, (16, 16, 1, 1))
    @test rotate_hw(a, 90) == rotate_hw(a, -270)
end

@testset "ssdn" for n_layers in [1, 2]
    n_features_in, n_features_out = 1, 1
    model = SSDN(n_features_in, n_layers, n_features_out)
    x = randn(Float32, (64, 64, 1, 1))
    # we're using "same"-padding throughout (as in Laine 2019) so the shape of
    # the output should be the same as the input
    @test size(model(x)) == size(x)
end

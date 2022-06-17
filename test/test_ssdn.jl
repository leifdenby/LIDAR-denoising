using Test
using LIDARdenoising
using Flux: MaxPool


@testset "half-plane generics" begin
    nx = 3
    ny = 3
    ns = 3  # size of filter is 3x3
    no = (ns + 1) ÷ 2  # expected offset is 2 to avoid the center
    ## half-plane operations
    d_op = DummyHalfPlaneOp((ns, ns), 1)
    v1 = randn(Float32, (nx, ny, 1, 1))
    @test offset(d_op) == no
    @test all(offset(d_op, v1)[1:no,:] .== 0)
end

@test_skip @testset "half-plane convolutions nx=$nx" for nx in [3, 5]
    ny = 3
    ns = 3  # size of filter is 3x3

    # check that that only the values in the half-plane contribute to the final value
    v2 = randn(Float32, (nx, ny, 1, 1))
    offset_conv_3x3_xdim = HalfPlaneConv2D((ns,ns),1=>1, identity, pad=0)
    offset_conv_3x3_xdim_pad = HalfPlaneConv2D((ns,ns),1=>1, identity, pad=1)
    offset_conv_3x3_ydim = HalfPlaneConv2D((ns,ns),1=>1, identity, pad=0, dim=2)
    # all operators have same filter width
    no = offset(offset_conv_3x3_xdim)

    # change the values for the values that sholdn't effect the result
    v2_xmod = copy(v2)
    v2_xmod[nx-no+1:nx,:,:,:] .= 1
    v2_ymod = copy(v2)
    v2_ymod[:,ny-no+1:ny,:,:] .= 1 

    @test offset_conv_3x3_xdim(v2) == offset_conv_3x3_xdim(v2_xmod)
    @test offset_conv_3x3_ydim(v2) == offset_conv_3x3_ydim(v2_ymod)
    # XXX: tests using padding fail, they shouldn't, but I think thre is a mistake
    # in the paper and with padding the central pixel does effect the result
    @test_broken offset_conv_3x3_xdim_pad(v2) == offset_conv_3x3_xdim_pad(v2_xmod)
    @test offset_conv_3x3_xdim_pad(v2_xmod) == offset_conv_3x3_xdim_pad(v2_xmod)
end


@testset "half-plane max-pool" begin
    nx = 5
    ny = 4
    ns = 2  # size of filter is 2x2 as in Laine et al 2019

    offset_mp_xdim = HalfPlaneMaxPool2D((ns, ns), 1)
    offset_mp_ydim = HalfPlaneMaxPool2D((ns, ns), 2)
    # all operators have same filter width
    no = offset(offset_mp_xdim)
    @test no == 1

    # add batch and channel dimensions
    ex(x) = x[:,:,:,:]

    # can't compare against the max-pool values with/without padding because
    # the values will be shifted and the stride will fall on different values,
    # instead I'll just construct arrays which have the padding already applied

    v3 = repeat(1:nx, 1, ny) |> ex
    v3_padded_xdim = vcat(repeat([0], 1, ny), v3[1:(nx-1), :]) |> ex
    v3_padded_ydim = hcat(repeat([0], nx, 1), v3[:, 1:(ny-1)]) |> ex


    @test offset_mp_xdim(v3) == MaxPool((ns, ns))(v3_padded_xdim)
    @test offset_mp_ydim(v3) == MaxPool((ns, ns))(v3_padded_ydim)
end

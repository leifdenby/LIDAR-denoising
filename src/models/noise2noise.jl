using Distributions: Normal


struct Noise2Noise <: AbstractDenoiser
    model
end
Flux.@functor Noise2Noise

"""
Noise2Noise tries to predict one noisy sample from another usnig a (blind-spot) UNet. The blind-spot isn't actually necessary
"""
function loss(denoiser::Noise2Noise, x1_noisy, x2_noisy)
    Flux.Losses.mse(denoiser.model(x1_noisy), x2_noisy)
end

function (denoiser::Noise2Noise)(noisy_data::AbstractArray{T,4}) where {T}
    return denoiser.model(noisy_data |> device(denoiser)) |> cpu
end



# UNet functionality I copied in, needs replacing

expand_dims(x,n::Int) = reshape(x,ones(Int64,n)...,size(x)...)
function squeeze(x) 
    if size(x)[end] != 1
        return dropdims(x, dims = tuple(findall(size(x) .== 1)...))
    else
        # For the case BATCH_SIZE = 1
        int_val = dropdims(x, dims = tuple(findall(size(x) .== 1)...))
        return reshape(int_val,size(int_val)...,1)
    end
end



function BatchNormWrap(out_ch)
    Chain(x->expand_dims(x,2),
	  BatchNorm(out_ch),
	  x->squeeze(x))
end

function _random_normal(shape...)
    return Float32.(rand(Normal(0.f0,0.02f0),shape...))
end

UNetConvBlock(in_chs, out_chs, kernel = (3, 3)) =
    Chain(Conv(kernel, in_chs=>out_chs,pad = (1, 1);init=_random_normal),
	BatchNormWrap(out_chs),
	x->leakyrelu.(x,0.2f0))

ConvDown(in_chs,out_chs,kernel = (4,4)) =
  Chain(Conv(kernel,in_chs=>out_chs,pad=(1,1),stride=(2,2);init=_random_normal),
	BatchNormWrap(out_chs),
	x->leakyrelu.(x,0.2f0))


struct UNetUpBlock
  upsample
end

Flux.@functor UNetUpBlock

UNetUpBlock(in_chs::Int, out_chs::Int; kernel = (3, 3), p = 0.5f0) = 
    UNetUpBlock(Chain(x->leakyrelu.(x,0.2f0),
       		ConvTranspose((2, 2), in_chs=>out_chs,
			stride=(2, 2);init=_random_normal),
		BatchNormWrap(out_chs),
		Dropout(p)))

function (u::UNetUpBlock)(x, bridge)
  x = u.upsample(x)
  return cat(x, bridge, dims = 3)
end

"""
  Unet(channels::Pair{Int,Int})
  Initializes a [UNet](https://arxiv.org/pdf/1505.04597.pdf) instance with the
  given number of input/output `channels`, for example `1 => 3` for `1` input
  channel and `3` output channels.
"""
struct Unet
  conv_down_blocks
  conv_blocks
  up_blocks
end

Flux.@functor Unet

function UNet(channels)
  conv_down_blocks = Chain(ConvDown(64,64),
		      ConvDown(128,128),
		      ConvDown(256,256),
		      ConvDown(512,512))

  conv_blocks = Chain(UNetConvBlock(channels.first, 3),
		 UNetConvBlock(3, 64),
		 UNetConvBlock(64, 128),
		 UNetConvBlock(128, 256),
		 UNetConvBlock(256, 512),
		 UNetConvBlock(512, 1024),
		 UNetConvBlock(1024, 1024))

  up_blocks = Chain(UNetUpBlock(1024, 512),
		UNetUpBlock(1024, 256),
		UNetUpBlock(512, 128),
		UNetUpBlock(256, 64,p = 0.0f0),
		Chain(x->leakyrelu.(x,0.2f0),
		Conv((1, 1), 128=>channels.second;init=_random_normal)))									  
  Unet(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::Unet)(x::AbstractArray)
  op = u.conv_blocks[1:2](x)

  x1 = u.conv_blocks[3](u.conv_down_blocks[1](op))
  x2 = u.conv_blocks[4](u.conv_down_blocks[2](x1))
  x3 = u.conv_blocks[5](u.conv_down_blocks[3](x2))
  x4 = u.conv_blocks[6](u.conv_down_blocks[4](x3))

  up_x4 = u.conv_blocks[7](x4)

  up_x1 = u.up_blocks[1](up_x4, x3)
  up_x2 = u.up_blocks[2](up_x1, x2)
  up_x3 = u.up_blocks[3](up_x2, x1)
  up_x5 = u.up_blocks[4](up_x3, op)
  tanh.(u.up_blocks[end](up_x5))
end


function Noise2Noise(;channels::Pair{Int, Int}=1=>1)
    model = UNet(channels)
    Noise2Noise(model)
end
module Models

using Flux
using CUDA
using Flux: pad_zeros, MaxPool, Conv, SamePad
using Flux: leakyrelu, ConvTranspose, Chain, SkipConnection
using Base: split


include("ssdn.jl")
include("linear.jl")
include("utils.jl")

export rotr90

end
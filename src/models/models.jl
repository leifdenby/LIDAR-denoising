module Models

using Flux
using CUDA
using MLUtils
using Flux



include("common.jl")
export train!

#include("ssdn.jl")
#include("linear.jl")
#
include("supervised.jl")
export Noise2CleanDenoiser, DnCNN

end
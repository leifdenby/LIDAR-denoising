module Models

using Flux
using CUDA
using MLUtils
using Flux



include("common.jl")
include("utils.jl")
export train!

# partially supervised (requires two noisy samplse for training)
include("blindspot_unet.jl")
include("noise2noise.jl")
export Noise2Noise

# unsupervised models (requires only noisy samples for training)
#include("ssdn.jl")
#include("linear.jl")
include("selfsupervised.jl")
export SelfSupervisedDenoiser

# supervised denoisers (requires noisy-clean samples for training)
include("supervised.jl")
export Noise2CleanDenoiser, DnCNN

end
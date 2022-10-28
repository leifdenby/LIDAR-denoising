module Models

using Flux
using CUDA
using MLUtils
using Flux



include("general.jl")
#include("ssdn.jl")
#include("linear.jl")
include("supervised.jl")

export LinearDenoiser, Noise2CleanDenoiser, Noise2NoiseDenoiser, LaineSelfSupervisedDenoiser
export train!

end
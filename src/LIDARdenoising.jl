module LIDARdenoising

using Distributions
import PyPlot
import Statistics
import Logging: AbstractLogger, with_logger, @info
using Logging: @info, ConsoleLogger, with_logger
using CUDA
import Flux
using Flux: gpu, cpu
using Flux: identity, Conv, Chain, sigmoid, unsqueeze
using NCDatasets
using Random: shuffle
using MLUtils: unsqueeze

# store gpu/cpu device into `_device` variable, can't use name `device` since CUDA exports that
if CUDA.functional()
    @info "CUDA is on"
    _device = gpu
    CUDA.allowscalar(false)
else
    _device = cpu
end
using Statistics: mean, std

include("noise.jl")
include("dataloader.jl")
include("normalization.jl")
include("ncfile.jl")
include("model.jl")
include("train.jl")
include("plot.jl")

export DataLoaderLES


end

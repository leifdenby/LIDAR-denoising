module LIDARdenoising

using Distributions
import Statistics
import Logging: AbstractLogger, with_logger, @info
using Logging: @info, ConsoleLogger, with_logger
using CUDA
using Flux
using NCDatasets
using Random: shuffle
using MLUtils: unsqueeze, splitobs, shuffleobs
using Plots: plot, heatmap, savefig

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
include("models/models.jl")
include("plot.jl")

export DataLoaderLES


end

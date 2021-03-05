using DrWatson
using Test
@quickactivate

include(srcdir("normalization.jl"))

a = rand(Float64, (2,3,4)) .+ 42.0
@test a ≈ denormalize(normalize(a))

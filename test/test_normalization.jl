using Test
using LIDARdenoising


a = rand(Float64, (2,3,4)) .+ 42.0
@test a ≈ LIDARdenoising.denormalize(LIDARdenoising.normalize(a))

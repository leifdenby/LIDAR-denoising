using DrWatson
@quickactivate

include(srcdir("dataloader.jl"))

# generate some fake data to test with
nx, ny, nz = 40, 30, 20
x = rand(Float32, (nz, ny, nx))

# make batchsize larger than x-dimension so we ensure the indexing is being
# done correctly to only pick valid indecies
nbatches = 8
batchsize = 50

dl = DataLoaderLES(x; batchsize=batchsize, nbatches=nbatches)
@test length(dl) == nbatches

for batch in dl
    x_batch, y_batch = batch
    # TODO: currently we always do slices along perpendicular to the y-axis,
    # make this an argument so we can use both
    @test size(x_batch) == (nz, nx, batchsize)
    @test size(y_batch) == (nz, nx, batchsize)
end

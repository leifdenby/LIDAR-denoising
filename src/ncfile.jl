using NCDatasets


"""
Load LES data for training
"""
function load_data(filename; z_max=nothing, dtype=Float32)
    # TODO: don't hardware variable name, support transposed data and data with time dimension
    ds = NCDatasets.Dataset(filename)

    x_grid = ds["xt"]
    y_grid = ds["yt"]
    z_grid = ds["zt"]

    # Assumed shape is (z,y,x,t), take for timestep for now
    da = ds["qv"][:,:,:,1]

    if z_max != nothing
        k_max = argmax(z_grid .> z_max)-1
        z_grid = z_grid[1:k_max,:,:]
        da = da[1:k_max,:,:]
    end
    return dtype.(replace(da, missing => 0.0))
end

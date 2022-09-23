struct GriddedData3D{T} <: AbstractArray{T,3}
    values::AbstractArray{T,3}
    x::AbstractVector{T}
    y::AbstractVector{T}
    z::AbstractVector{T}
end

Base.size(d::GriddedData3D) = size(d.values)
Base.getindex(d::GriddedData3D, key...) = Base.getindex(d.values, key...)

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
    values = dtype.(replace(da, missing => 0.0))

    get_values_1d(v) = dtype.(replace(v[:], missing => 0.0))
    return GriddedData3D(values, get_values_1d(x_grid), get_values_1d(y_grid), get_values_1d(z_grid))
end

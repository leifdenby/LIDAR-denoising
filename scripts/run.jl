using DrWatson: @quickactivate, srcdir
@quickactivate

include(srcdir("train.jl"))
using ArgParse: ArgParseSettings, @add_arg_table!, parse_args
using NCDatasets

"""
Load LES data for training
"""
function load_data(filename; z_max=nothing)
    # TODO: don't hardware variable name, support transposed data and data with time dimension
    ds = NCDatasets.Dataset(filename)

    x_grid = ds["xt"]
    y_grid = ds["yt"]
    z_grid = ds["zt"]

    # Assumed shape is (z,y,x,t), take for timestep for now
    da = ds["qv"][:,:,:,1]

    if z_max != nothing
        k_max = argmax(z_grid .< z_max)
        z_grid = z_grid[1:k_max,:,:]
        da = da[1:k_max,:,:]
    end
    return Float32.(replace(da, missing => 0.0))
end


function main()
    argparser = ArgParseSettings()
    @add_arg_table! argparser begin
        "--file"
            arg_type = String
            required = true
            help = "path to netCDF file containing LES data"
        "--z-max"
            arg_type = Float64
            help = "Maximum altitude to use for training data"
            default = 600.0
    end
    args = parse_args(argparser)

    data = load_data(args["file"]; z_max=args["z-max"])
    trained_model = train_model_on_data(data)
end
main()

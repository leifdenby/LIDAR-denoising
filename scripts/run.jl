using DrWatson: @quickactivate, srcdir
@quickactivate

include(srcdir("train.jl"))
include(srcdir("ncfile.jl"))

using ArgParse: ArgParseSettings, @add_arg_table!, parse_args
using WeightsAndBiasLogger
using Debugger: @bp, @run


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
        "--n-epochs"
            arg_type = Int
            help = "Number of epochs to train"
            default = 2
        "--lr"
            arg_type = Float64
            help = "learning rate"
            default = 0.5
        "--log-to-wandb"
            arg_type = Bool
            help = "Log to Weight & Biases"
            default = false
    end
    args = parse_args(argparser)

    data = load_data(args["file"]; z_max=args["z-max"])
    if args["log-to-wandb"]
        logger = WBLogger(project="LIDARdenoising.jl")
        config!(logger, args)
    end
    trained_model = train_model_on_data(data; lr=args["lr"], n_epochs=args["n-epochs"])
end
@run main()

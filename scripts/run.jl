using DrWatson: @quickactivate, srcdir
@quickactivate

include(srcdir("train.jl"))
include(srcdir("ncfile.jl"))
include(srcdir("plot.jl"))
include(srcdir("model.jl"))

using ArgParse: ArgParseSettings, @add_arg_table!, parse_args
using WeightsAndBiasLogger: config!, wandb, WBLogger
using Logging: NullLogger
using BSON: @load, @save
import Flux


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
            action => :store_true
        "--noise-level"
            arg_type = Float64
            help = "σ_noise level"
            default = 0.5
        "--pretrained-model"
            arg_type = String
            default = nothing
            help = "Load pretrained model as starting point for training"
        "--save-model-to"
            arg_type = String
            default = nothing
            help = "Save trained model to given filename"
    end
    args = parse_args(argparser)

    data = load_data(args["file"]; z_max=args["z-max"])
    if args["log-to-wandb"]
        logger = WBLogger(project="LIDARdenoising.jl")
        config!(logger, args)
    else
        logger = NullLogger()
    end

    Nf = 5  # filter size in model convolutions
    Nc = 6  # number of "channels" in model convolutions
    model = build_model(Nf, Nc)

    if args["pretrained-model"] != nothing
        fn = args["pretrained-model"]
        @load fn model
        @info "model loaded from $fn"
    end

    trained_model = train_model_on_data(model, data; lr=args["lr"], n_epochs=args["n-epochs"], logger=logger, σ_noise=args["noise-level"])

    if args["save-model-to"] != nothing
        fn = args["save-model-to"]
        model_on_cpu = Flux.cpu(model)
        @save fn model_on_cpu
        @info "model saved to $fn"
        if args["log-to-wandb"]
            wandb.save(fn)
        end
    end
end
main()

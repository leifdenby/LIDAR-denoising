begin
    N = 256
    Nf = 5
    Nd = N - 2 * Int((Nf - 1) / 2) * 3
    Nc = 6 # num channels
    Nd
end


"""build denoising neural network with filter size of `Nf` and number intermediate channels of `Nc`"""
function build_model(name="linear_1x1")

    if name == "linear_1x1"
        model = Conv((1, 1), 1 => 1, identity)
    elseif name == "linear_3x3"
        model = Conv((3, 3), 1 => 1, identity, pad=1)
    elseif name == "__foobar__"
        Nf = 5
        Nc = 6
        model = Chain(
            Conv((Nf, Nf), 1 => Nc, sigmoid),
            #Flux.MaxPool((4,4)),
            Conv((Nf, Nf), Nc => Nc, sigmoid),
            Conv((Nf, Nf), Nc => 1, sigmoid),
        )
    else
        error("construction of `$name` model not defined")
    end
    return model
end

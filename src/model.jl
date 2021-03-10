using Flux

begin
    N = 256
    Nf = 5
    Nd = N - 2 * Int((Nf - 1) / 2) * 3
    Nc = 6 # num channels
    Nd
end


"""build denoising neural network with filter size of `Nf` and number intermediate channels of `Nc`"""
function build_model(Nf::Int, Nc::Int)
    model = Flux.Chain(
        # reshape to add channels and batch dimensions
        x -> reshape(x, size(x, 1), size(x, 2), 1, size(x, 3)),

        ## NN interior below
        # first convolution
        Flux.Conv((Nf, Nf), 1 => Nc, Flux.sigmoid),
        #Flux.MaxPool((4,4)),
        Flux.Conv((Nf, Nf), Nc => Nc, Flux.sigmoid),
        Flux.Conv((Nf, Nf), Nc => 1, Flux.sigmoid),
        #x -> reshape(x, :), # flatten
        #Flux.Dense(Nd*Nd, 1)
        #Flux.Conv((Nd, Nd), 1 => 1),
        #x -> Flux.Conv((size(x, 1), size(x, 2)), 1 => 1)(x),

        # flatten, remove channels and batch dimensions
        x -> reshape(x, size(x, 1), size(x, 2), size(x, 4)),
    )
    return model
end

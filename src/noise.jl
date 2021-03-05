import Distributions

function add_noise(x; σ = 0.01)
    dist = Distributions.Normal(0.0, σ)
    x + (typeof(x))(rand(dist))
end

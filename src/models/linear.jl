function Linear(;conv_size=1)
    return Conv((conv_size, conv_size), 1 => 1, identity, pad=1)
end
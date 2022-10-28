function Linear(n_features_in, n_features_out)
    return Conv((1, 1), n_features_in => n_features_out, identity, pad=0)
end
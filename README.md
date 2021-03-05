# LIDAR denoising

[![LIDAR-denoising](https://github.com/leifdenby/LIDAR-denoising/actions/workflows/main.yml/badge.svg)](https://github.com/leifdenby/LIDAR-denoising/actions/workflows/main.yml)

This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> LIDAR denoising

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box.

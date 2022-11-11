using Plots
using Distributions
using Test

"""
https://github.com/cran/kza/blob/master/src/kzf.c#L13
"""
function R_differenced(y::Vector{T}, q::Int) where T
    n = length(y)
    d = Vector{T}(undef, n)
    dprime = Vector{T}(undef, n)
    
	# calculate d = |Z(i+q) - Z(i-q)|
    #R: for (i=0; i<q; i++) {REAL(d)[i] = fabs(REAL(y)[i+q] - REAL(y)[0]);}
	for i in 1:q
        try
            d[i] = abs(y[i+q] - y[1])
        catch
            @info i q
            throw("lol")
        end
    end
    #R: for (i=q; i<n-q; i++) {REAL(d)[i] = fabs(REAL(y)[i+q] - REAL(y)[i-q]);}
	for i in q+1:n-q
        d[i] = abs(y[i+q] - y[i-q])
    end
    #R: for (i=n-q; i<n; i++) {REAL(d)[i] = fabs(REAL(y)[n-1] - REAL(y)[i-q]);}
	for i in n-q+1:n
        d[i] = abs(y[n] - y[i-q])
    end

	# d'(t) = d(i+1)-d(i)
    #R: for(i=0; i<n-1; i++) REAL(dprime)[i] = REAL(d)[i+1]-REAL(d)[i];
	for i in 1:n-1
        dprime[i] = d[i+1]-d[i]
    end
	dprime[n] = dprime[n-1]
    return d, dprime
end

N = 40
N_step = 10
i = 1:N

y_ref = -1.0e-1 .* [(i_ + i_ % N_step)/N for i_ in i]
y_noisy = y_ref .+ 0.05 .* rand(Normal(), N)

@testset "d and dprime (w=$(w))" for w in [3,4,5]
    y_sample = collect(1:20)  # line with unit slope
    d_sample, dprime_sample = R_differenced(y_sample, w)
    @test d_sample[1:w] == reverse(d_sample[end-w+1:end])
    @test all(d_sample[w+1:end-w] .== w*2)
    @test all(dprime_sample[1:w] .== 1)
    @test all(dprime_sample[end-w:end] .== -1)
    @test all(dprime_sample[w+1:end-w-1] .== 0)
end


"""
compute moving average value for array `v` at index `col` with window `w`
skipping any `missing` values in `v`
"""
function mavg1d(v::Vector{T}, col::Int, w::Int) where T
    s::T = 0.0
    z = 0

    startcol = col-w>1 ? col-w : 1
    endcol = col+w<length(v) ? col+w : length(v)

    #@show col w startcol endcol
    
    for i in startcol:endcol
        if !ismissing(v[i])
            z += 1
            s += v[i]
        end
    end
    if (z == 0)
        return missing
    else
        return s/z
    end
end

@testset "moving average" begin
    y_mavg = [mavg1d(y_noisy, i, 9) for i in 1:N]
    # drop 5% of the value, moving average should stay similar
    y_missing = [rand() > 0.95 ? missing : y for y in y_noisy]
    y_missing_mavg = [mavg1d(y_missing, i, 9) for i in 1:N]
    @test maximum(skipmissing(abs.(y_missing - y_missing_mavg))) < 0.2
end

"""
adaptive curvature function
"""
function adaptive(d::T, m::T) where T
	return 1 - d/m
end


"""
v: A vector of the time series
window: The window for the filter.
iterations: The number of iterations.
min_size:  Minimum size of window q.
tolerance: The smallest value to accept as nonzero.

source: https://github.com/cran/kza/blob/master/src/kza.c#L60
"""
function kza1d(v::Vector{T}, window::Integer; iterations=3, minimum_window_length=Int(round(0.05*window)), tolerance=1.0e-5) where T
    n = length(v)
    eps = tolerance
	q = window
	min_window_length = minimum_window_length

    y = kz1d(v, q; iterations=iterations)
    d, dprime = R_differenced(y, q)
 
    m = maximum(skipmissing(d))

    tmp = copy(v)
    ans = Vector{T}(undef, n)

    #for(i=0; i<INTEGER_VALUE(iterations); i++) {
    for i in 1:iterations
    	#for (t=0; t<n; t++) {
        for t in 1:n
		    #if (fabs(REAL(dprime)[t]) < eps) { /* dprime[t] = 0 */
            if abs(dprime[t]) < eps  # dprime[t] = 0
			    qh = Int(floor(q*adaptive(d[t], m)))
			    qt = Int(floor(q*adaptive(d[t], m)))
            elseif dprime[t] < 0.0
                qh = q
    		    qt = Int(floor(q*adaptive(d[t], m)))
            else
		    	qh = Int(floor(q*adaptive(d[t], m)))
			    qt = q;
            end
			qt = (qt < min_window_length) ? min_window_length : qt
			qh = (qh < min_window_length) ? min_window_length : qh
            
            #@info "step" d[t] adaptive(d[t], m) q t qt qh
	
	        # /* check bounds */
        	qh = (qh > n-t) ? n-t : qh; # head past end of serie
            qt = (qt >= t) ? t-1 : qt;  	        		
            #@info "bounds" t-qt t+qh t
   		    ans[t] = mean(skipmissing(tmp[t-qt:t+qh]))
        end
	    # copyVector(tmp, ans);
        tmp[:] = ans[:]
    end
	return ans
end

function kz1d(x::Vector{T}, window::Int; iterations::Int=3) where T
    ans = Vector{T}(undef, length(x))
    tmp = copy(x)
    
    for k in 1:iterations
        for i in 1:length(x)
            ans[i] = mavg1d(tmp, i, window)
        end
        tmp[:] = ans[:]
    end
    return ans
end

mavg1d(y, 1, 3)

N = 600
N_step = N // 2
i = 1:N

y_ref = -1.0e-1 .* [(i_ + i_ % N_step)/N for i_ in i]
y_noisy = y_ref .+ 0.01 .* rand(Normal(), N)

plot(i, y_noisy)
plot!(i, y_ref)
plot!(i, kza1d(y_noisy, 3), iterations=1, color=:red)



yrs = 20
t = range(0, yrs, yrs*365)
m = 365
#noise
e = rand(Normal(), length(t))
e = zeros(length(t))
trend = range(0,-1,length(t))
#signal
bkpt = 3452
brk = [repeat([0.0], bkpt); repeat([0.5], length(t)-bkpt)]
signal = trend + brk
# y = seasonal + trend + break point + noise
y = sin.(2*pi*t) + signal + e

plot(t, y, type=:line, ylim=(-3, 3), label="y")
plot!(t, signal, label="signal")

y_kz = kz1d(y,m)
plot!(t, y_kz, label="KZ")
# kza reconstruction of the signal
y_kza = kza1d(y,m,minimum_window_length=10, iterations=3);
plot!(t, y_kza, label="KZA", color=:red, ylim=(-1, 1))

par(mfrow=c(2,1))
plot(y,type="l", ylim=c(-3,3))
plot(signal,type="l",ylim=c(-3,3),
main="Signal and KZA Reconstruction")
lines(k.kza$kza, col=4)


using NetCDF

sample_arr = transpose(ncread("scripts/sample_long.nc", "WaterVaporMixingRatio"))[:,1:12_000]
sample_arr = transpose(ncread("scripts/sample_long.nc", "WaterVaporMixingRatio"))[:,12_000:24_000]
alt = transpose(ncread("scripts/sample_long.nc", "alt"))[:]
#sample_arr = transpose(ncread("scripts/sample(1).nc", "WaterVaporMixingRatio"))[:,1:6300]
heatmap(sample_arr)

function do_slice(;wi=3000,wl=3500)
    y_sample = sample_arr[10,:]
    plot(y_sample)
    y_kza = kza1d(y_sample, Int(240 // 4))
    plot!(y_kza, color=:red)

    sample_rel = copy(sample_arr)
    for ti in 1:length(y_kza)
        sample_rel[:,ti] .-= y_kza[ti]
    end

    # sliced window with alt and dist
    xt = collect(1:wl+1) * 4 * 10 # [m]
    plot(
        heatmap(xt, alt, sample_arr[:,wi:wi+wl]),
        heatmap(xt, alt, sample_arr[:,wi:wi+wl] .> 15.5),
        heatmap(xt, alt, sample_rel[:,wi:wi+wl], clim=(-1, 1), c=:balance),
        heatmap(xt, alt, sample_rel[:,wi:wi+wl] .> 0.2),
        layout=(4,1),
        size=(1200, 800),
    )
end

do_slice()

y_sample = sample_arr[28,:]
plot(y_sample)
for n in 1:10
    y_kza = kza1d(y_sample, Int((n * 60)//4))
    plot!(y_kza, color=:red)
end
plot!()

using ProgressMeter

windows = 2:240
windows = 120:5:1200
#@showprogress for n in [2:10:600; 600:100:6000]
variances = []
@showprogress for w in windows
    y_kza = kza1d(y_sample, w)
    push!(variances, std(y_kza - y_sample))
end
plot(windows * 4, 1.0./(1.0 .- variances), marker=:dot, xlabel="window [s]", ylabel="1/(1 - σ)")
plot(windows * 4 / 60, 1.0./(1.0 .- variances), marker=:dot, xlabel="window [min]", ylabel="1/(1 - σ)")
plot(windows * 4 / 60 / 60, 1.0./(1.0 .- variances), marker=:dot, xlabel="window [hr]", ylabel="1/(1 - σ)")

nn = 100
plot((windows * 4)[1:nn], 1.0./(1.0 .- variances)[1:nn], marker=:dot, xlabel="window [s]", ylabel="1/(1 - σ)", xscale=:identity)
nn = 60
plot((windows * 4)[1:nn], 1.0./(1.0 .- variances)[1:nn], marker=:dot, xlabel="window [s]", ylabel="1/(1 - σ)", xscale=:identity)


t_mins = collect(1:size(sample_arr, 2)) .* 4 ./ 60
t_hrs = collect(1:size(sample_arr, 2)) .* 4 ./ 60 ./ 60
t = t_hrs
plot(t, y_sample, xlabel="time [hr]")
for w_secs in [60, 120, 240, 480, ]
    y_kza = kza1d(y_sample, Int(w_secs / 4))
    plot!(t, y_kza, color=:red, label="win=$(w_secs)s")
end
plot!()
y_kza = kza1d(y_sample, 100*4)
plot!(y_kza, color=:red)

function f(dx,x) # in-place
    for i in 2:length(x)-1
      dx[i] = x[i-1] - 2x[i] + x[i+1]
    end
    dx[1] = -2x[1] + x[2]
    dx[end] = 0.0#x[end-1] - 2x[end]
    nothing
end

dx = copy(variances)

xx, yy = windows * 4, 1.0./(1.0 .- variances) 
using BSplines
using Interpolations

spl = CubicSplineInterpolation(xx, yy)
plot(xx, spl.(xx))
dydx = [Interpolations.gradient(spl, x)[1] for x in xx]
plot(xx, x -> Interpolations.gradient(spl, x)[1])

dx = copy(yy)
f(dx, yy)
plot(
    #plot(spl),
    plot(xx / 60, spl.(xx)),
    #plot(xx, yy, marker="."),
    #plot(xx[2:end], dx[2:end], marker="."),
    plot(xx, x -> Interpolations.gradient(spl, x)[1]),
    layout=(2,1)
)
### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ bd0b7548-6ae5-11eb-0c35-d5e8291605c1
begin
	import NCDatasets
	import Plots
	import Flux
	import Statistics
end

# ╔═╡ effedcd8-6ae5-11eb-3da8-a17cb77d9331
begin
	fn_qv = "../LES_data/data/training_data/qv.x0.0.nc"
	fn_qv = "../LES_data/data/noshear_br0.05.tn6/rico.qv.nc"

	fn_mu = "../LES_data/data/training_data/cvrxp_p_stddivs.x0.0.nc"
	
	da_qv = NCDatasets.Dataset(fn_qv)
	#da_mu = NCDatasets.Dataset(fn_mu)
	
	zslice = 1:40
	x_grid = da_qv["xt"]
	y_grid = da_qv["yt"]
	z_grid = da_qv["zt"][zslice]
	qv = Float32.(replace(da_qv["qv"][zslice,:,:,1], missing => 0.0))
	
	qv_mean = Statistics.mean(qv)
	qv_std = Statistics.std(qv)
	qv .-= qv_mean
	qv ./= qv_std
end

# ╔═╡ 2950c00a-6ae6-11eb-2350-dda6643de180
begin
	N = 256
	Nf = 5
	Nd = N - 2*Int((Nf-1)/2)*3
	Nc = 6 # num channels
	Nd
end

# ╔═╡ 1cbd694c-6ae6-11eb-26bd-136c79a123c3
model = Flux.Chain(
	# reshape to add channels and batch dimensions
	x -> reshape(x, size(x, 1), size(x, 2), 1, 1),
	
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
	x -> reshape(x, size(x, 1), size(x, 2)), 
)

# ╔═╡ 57b14032-6ae6-11eb-3709-93b97dfe4fe4
begin
	v = rand(Float32, (400, 300))
	model(v)
end

# ╔═╡ 804ceb5c-6ae8-11eb-3c76-33a8c8478064
begin
	Plots.heatmap(x_grid, z_grid, qv[:,:,1])
end

# ╔═╡ 6b02c754-6bb4-11eb-1a80-f7816e5d39fe
function plot_pred(x, y, ŷ)
	#z_grid = da_qv["zt"][6:end-7]
	#x_grid = da_qv["xt"][6:end-7]
	x_grid_conv = x_grid[6:end-7]
	z_grid_conv = z_grid[6:end-7]
	
	p1 = Plots.heatmap(x_grid_conv, z_grid_conv, x[6:end-7,6:end-7], title="x")
	p2 = Plots.heatmap(x_grid_conv, z_grid_conv, y, title="y")
	p3 = Plots.heatmap(x_grid_conv, z_grid_conv, ŷ, title="ŷ")
	
	Plots.plot(p1, p2, p3, layout=(3,1))
end

# ╔═╡ ab0ebcc4-6aef-11eb-1efc-73693ae8f668
begin
	v2 = rand(Float32, (64, 64, 22))
	
	#fn1(x::AbstractArray{T,3}) where T = 
	v2[1]
end

# ╔═╡ 4f4ee288-6aec-11eb-29fb-538c135cda19
begin
	struct DataLoaderLES{D}
		data::AbstractArray{D,3}
		batchsize::Int
		nbatches::Int
	end
	
	function DataLoaderLES(data; batchsize=1, nbatches=1)
		batchsize > 0 || throw(ArgumentError("Need positive batchsize"))
		nbatches > 0 || throw(ArgumentError("Need positive nbatches"))

		DataLoaderLES(data, batchsize, nbatches)
	end
	
	# required functions to support iteration
	Base.length(dl::DataLoaderLES) = dl.batchsize * dl.nbatches
	
	function _getExample(data::AbstractArray{D,3}) where D
		nx, ny, nz = size(data)
		
		data_x_idx = rand(1:nx)
		# TODO: used ShiftedArrays.jl to roll a random amount
		#y_shuffle = rand(1:ny)
		
		ex = data[:,data_x_idx,:]
		
		x = ex
		y = ex[6:end-7,6:end-7]
		
		@show data_x_idx
		
		return (x, y)
	end
	
	function Base.iterate(d::DataLoaderLES, i=0)
		if i >= d.batchsize * d.nbatches
			return nothing
		end
		
		return (_getExample(d.data), i+1)
	end
		# returns data in d.indices[i+1:i+batchsize]
#		
#		nexti = min(i + d.batchsize, d.nobs)
#		ids = d.indices[i+1:nexti]
#		batch = _getobs(d.data, ids)
#		return (batch, nexti)
#	end
end

# ╔═╡ f7e562da-6bb4-11eb-1764-cf6e30d20ac5
begin
	x1, y1 = _getExample(replace(qv, missing => 0.0))
	plot_pred(x1, y1, model(x1))
	#size(x1), size(y1), size(model(x1)), size(x_grid)
end

# ╔═╡ ff83e786-6ae8-11eb-381d-6fb89f7ec6f4
begin
	function loss(x, y)
			ŷ = model(x)
			sum((y .- ŷ).^2)
	end
	
	loss(qv[:,:,1], qv[6:end-7,6:end-7,:,1])
end

# ╔═╡ d66cdaa4-6b92-11eb-280e-3fda42a8d9ae
#Base.Iterators.take(DataLoaderLES(v2), 1)

# ╔═╡ 8f0ee1a8-6b2f-11eb-1bed-e795315b182d
for d in DataLoaderLES(qv)
	x, y = d
	loss(x, y)
end

# ╔═╡ 8feafc4c-6ae9-11eb-081b-877591dee4d4
begin
	opt = Flux.Optimise.Descent(0.01)
	test_x, test_y = _getExample(qv)
	evalcb() = @show(loss(test_x, test_y))

	#Flux.@epochs 22
	Flux.train!(loss, Flux.params(model), DataLoaderLES(qv), opt; cb=evalcb)
end

# ╔═╡ c5307a3e-6bb2-11eb-0512-49e59c06bd96


# ╔═╡ f5efb9ee-6b96-11eb-3344-6fdba3166fdd
begin
	x, y = _getExample(qv)
	loss(x, y)
	plot_pred(x, y, model(x))
end

# ╔═╡ Cell order:
# ╠═bd0b7548-6ae5-11eb-0c35-d5e8291605c1
# ╠═effedcd8-6ae5-11eb-3da8-a17cb77d9331
# ╠═2950c00a-6ae6-11eb-2350-dda6643de180
# ╠═1cbd694c-6ae6-11eb-26bd-136c79a123c3
# ╠═57b14032-6ae6-11eb-3709-93b97dfe4fe4
# ╠═804ceb5c-6ae8-11eb-3c76-33a8c8478064
# ╠═6b02c754-6bb4-11eb-1a80-f7816e5d39fe
# ╠═f7e562da-6bb4-11eb-1764-cf6e30d20ac5
# ╠═ab0ebcc4-6aef-11eb-1efc-73693ae8f668
# ╠═4f4ee288-6aec-11eb-29fb-538c135cda19
# ╠═ff83e786-6ae8-11eb-381d-6fb89f7ec6f4
# ╠═d66cdaa4-6b92-11eb-280e-3fda42a8d9ae
# ╠═8f0ee1a8-6b2f-11eb-1bed-e795315b182d
# ╠═8feafc4c-6ae9-11eb-081b-877591dee4d4
# ╠═c5307a3e-6bb2-11eb-0512-49e59c06bd96
# ╠═f5efb9ee-6b96-11eb-3344-6fdba3166fdd

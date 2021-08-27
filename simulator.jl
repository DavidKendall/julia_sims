### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 248d7979-e8da-408d-88da-97e52d84648e
import Pkg; Pkg.add("Agents")

# ╔═╡ 7b9704ab-5c42-4825-8661-1e391c3a3f26
using Random

# ╔═╡ fb288ecc-468a-4bb5-8f53-6b1b695b75e0
using Statistics

# ╔═╡ 16b91667-fe1d-4173-a301-f7e1694c7bb0
using BenchmarkTools

# ╔═╡ 1d910032-0755-4343-8a7d-530514f7400f
using Plots

# ╔═╡ 533ec0e4-465d-4ac4-ac93-17d95692343f
using LinearAlgebra

# ╔═╡ b246f7ae-a50a-4f1e-8ad4-35a28058f662
using PlutoUI

# ╔═╡ a53416e2-0f0d-4da1-b3e7-53d609291594
using JSON

# ╔═╡ 101d06b9-2e3c-4e62-98e4-a275ad860c61
using Profile

# ╔═╡ eaafe1e8-6873-490a-9caf-339e17d39ebc
include("dsim.jl")

# ╔═╡ 3f0339d3-14a5-43cd-9f36-64d49356f443
begin
    const POS_X = 1
    const POS_Y = 2
    const COH_X = 3
    const COH_Y = 4
    const REP_X = 5
    const REP_Y = 6
    const DIR_X = 7
    const DIR_Y = 8
    const RES_X = 9
    const RES_Y = 10
    const GOAL_X = 11
    const GOAL_Y = 12
    const PRM = 13
    const GAP_X = 14
    const GAP_Y = 15
    const COH_N = 16
    const REP_N = 17
    const DEBUG = 18
    const N_COLS = 18
end

# ╔═╡ 5fbe1265-2a93-431a-88f2-7108723f8995
const default_swarm_params = Dict(
    "cb" => 3.0,
    "rb" => [[2.0,2.0],[2.0,2.0]],
    "kc" => [[0.15,0.15],[0.15,0.15]],
    "kr" => [[50.0,50.0],[50.0,50.0]],
    "kd" => 0.0,
    "kg" => 0.0,
    "scaling" => "linear",
    "exp_rate" => 0.2,
    "speed" => 0.05,
    "stability_factor" => 0.0,
    "perim_coord" => false,
    "rgf" => false
)

# ╔═╡ 13fb8202-56fe-4197-8c78-4e997ba3d86b
function mk_rand_swarm(n; goal=[0. 0.], loc=0., grid=10., seed=nothing)
    b = zeros((n,N_COLS))
    rng = MersenneTwister(seed)
    xs = (rand(rng, Float64, (n,1)) * 2. .- 1.) .* grid .+ loc
    ys = (rand(rng, Float64, (n,1)) * 2. .- 1.) .* grid .+ loc
    b[:,POS_X] = xs
    b[:,POS_Y] = ys
    b[:,GOAL_X:GOAL_Y] .= goal
    return b
end

# ╔═╡ a32771e7-5408-4674-aee9-19428fdae26a
function mk_swarm(xs, ys; goal=[0. 0.])
	  n = length(xs)
    b = zeros((n,N_COLS))
    b[:,POS_X] = xs .+ 25.0
    b[:,POS_Y] = ys .+ 25.0
    b[:,GOAL_X:GOAL_Y] .= goal
    return b
end

# ╔═╡ e274cd80-70f6-4795-b6d3-a17bc91eba0d
function all_pairs_mag(b::Matrix{Float64}, cb::Float64)
	n_agents = size(b)[1]
	b[:,COH_N] .= 0.
	xv = Array{Float64,2}(undef, (n_agents, n_agents))
	yv = Array{Float64,2}(undef, (n_agents, n_agents))
	mag = Array{Float64,2}(undef, (n_agents, n_agents))
	for j in 1:n_agents
		xv[j,j] = 0.
		yv[j,j] = 0.
		mag[j,j] = cb + 1.
		for i in (j+1):n_agents
			xv[i,j] = b[i,POS_X] - b[j,POS_X]
			yv[i,j] = b[i,POS_Y] - b[j,POS_Y]
      mag[i,j] = √(xv[i,j]^2 + yv[i,j]^2)
			xv[j,i] = -xv[i,j]
			yv[j,i] = -yv[i,j]
      mag[j,i] = mag[i,j]
      if mag[i,j] ≤ cb
        b[i,COH_N] += 1
        b[j,COH_N] += 1
      end
    end
  end
	return xv, yv, mag
end

# ╔═╡ 951547d1-36d9-4105-aeab-5b150f43d1b3
function compute_coh(b,xv,yv,mag,cb,kc,p)
	n_agents = size(b)[1]
	Threads.@threads for i in 1:n_agents
		b[i,COH_X] = 0.
		b[i,COH_Y] = 0.
		for j in 1:n_agents
			if j != i && mag[j,i] ≤ cb
				b[i, COH_X] += xv[j,i] * kc[p[i],p[j]]
				b[i, COH_Y] += yv[j,i] * kc[p[i],p[j]]
			end
		end
	end		
end

# ╔═╡ 7c192780-1f9d-4fd3-8e71-edae1560cec3
function compute_coh2(b,xv,yv,mag,cb,kc,p)
	n_agents = size(b)[1]
	Threads.@threads for j in 1:n_agents
		b[j,COH_X] = 0.
		b[j,COH_Y] = 0.
		for i in 1:n_agents
			if i != j && mag[i,j] ≤ cb
				b[j, COH_X] += xv[i,j] * kc[p[j],p[i]]
				b[j, COH_Y] += yv[i,j] * kc[p[j],p[i]]
			end
		end
	end		
end

# ╔═╡ 6e76ec47-b9b6-4644-85d4-8b07116633a6
function compute_rep_linear(b,xv,yv,mag,rb,kr,p)
	n_agents = size(b)[1]
	Threads.@threads for i in 1:n_agents
		b[i,REP_N] = 0.0
		b[i,REP_X] = 0.0
		b[i,REP_Y] = 0.0
		for j in 1:n_agents
			if j != i && mag[j, i] <= rb[p[i],p[j]]
				b[i,REP_N] = b[i,REP_N] + 1
				b[i,REP_X] = b[i,REP_X] + (1. - (rb[p[i],p[j]] / mag[j,i])) * xv[j,i] * kr[p[i],p[j]]
				b[i,REP_Y] = b[i,REP_Y] + (1. - (rb[p[i],p[j]] / mag[j,i])) * yv[j,i] * kr[p[i],p[j]]
			end
		end
	end
end

# ╔═╡ 6013f4fa-c799-4178-8f7d-776bc81843dc
function nbr_sort(nbrs, ang, i)
	n = length(nbrs)
	for j in 1:n
		jmin = j
		for k in j:n
			if ang[:,i][nbrs[k]] < ang[:,i][nbrs[jmin]]
				jmin = k
			end
		end
		if jmin != j
		    nbrs[jmin], nbrs[j] = nbrs[j], nbrs[jmin]
		end
	end
end

# ╔═╡ 3c0e9c83-ee36-4755-a349-1b7a18b762de
function second((a, b))
    return b
end

# ╔═╡ 5c9be6fc-30bc-480b-8386-f2206f86ef5a
function on_perim(b,xv,yv,mag,cb,kg,rgf)
    n_agents = size(b)[1]
    b[:,PRM] .= 0.
    p = ones(Int64, n_agents)
	  Threads.@threads for i in 1:n_agents
		    b[i,GAP_X] = 0.
		    b[i,GAP_Y] = 0.
		    if b[i,COH_N] < 3
			      p[i] = 2
			      b[i, PRM] = 1.
			      continue
        end
        nbrs = findall(mag[:, i] .≤ cb)
        n_nbrs = length(nbrs)
        ang = atan.(yv[nbrs,i], xv[nbrs,i])
    #nbr_sort(nbrs, ang, i, k-1)
    nbrs = sort!(collect(zip(ang, nbrs)))
    #sort!(nbrs, by = get_ang)
    #sort!(nbrs)
		for j in 1:n_nbrs
			k = (j % n_nbrs) + 1
      if mag[nbrs[k][2],nbrs[j][2]] > cb
				p[i] = 2
				b[i,PRM] = 1.
        b[i, GAP_X] += kg * ((0.5 * (b[nbrs[k][2],POS_X] + b[nbrs[j][2],POS_X])) - b[i,POS_X])
        b[i, GAP_Y] += kg * ((0.5 * (b[nbrs[k][2],POS_Y] + b[nbrs[j][2],POS_Y])) - b[i,POS_Y])
				break
			else
          delta = nbrs[k][1] - nbrs[j][1]
				if delta < 0
			        delta += 2π
				end
				if delta > π
					p[i] = 2
					b[i,PRM] = 1.
					if rgf
              b[i, DEBUG] += 1.0
              b[i, GAP_X] += kg * ((0.5 * (b[nbrs[k][2],POS_X] + b[nbrs[j][2],POS_X])) - b[i,POS_X])
              b[i, GAP_Y] += kg * ((0.5 * (b[nbrs[k][2],POS_Y] + b[nbrs[j][2],POS_Y])) - b[i,POS_Y])
					end
					break
				end
			end
		end
	end
	return p
end


# ╔═╡ dc2d4b1d-561b-4339-ae25-2d5b1b08cc2d
function on_perim2(b,xv,yv,mag,cb,kg,rgf)
    n_agents = size(b)[1]
    b[:,PRM] .= 0.
    p = ones(Int64, n_agents)
    ang = atan.(yv,xv)
	  for i in 1:n_agents
		    b[i,GAP_X] = 0.
		    b[i,GAP_Y] = 0.
		    if b[i,COH_N] < 3
			      p[i] = 2
			      b[i, PRM] = 1.
			      continue
        end
        # nbrs = findall(mag[:, i] .≤ cb)
		
		#Alternative to nbrs = findall(mag[:,i] .≤ cb)
		nbrs = zeros(Int64, Int(b[i, COH_N]))
		k = 1
		for j in 1:n_agents
		if j != i && mag[j, i] <= cb
		nbrs[k] = j
		k += 1
			end
		end
        n_nbrs = length(nbrs)
	
		# Only one of the following 2 lines should be uncommented - nbr_sort very slow
        # nbr_sort(nbrs, ang, i)
        sort!(nbrs, by = n -> ang[i,n])
		    for j in 1:n_nbrs
			      k = (j % n_nbrs) + 1
            if mag[nbrs[k],nbrs[j]] > cb
				        p[i] = 2
				        b[i,PRM] = 1.
                b[i, GAP_X] += kg * ((0.5 * (b[nbrs[k],POS_X] + b[nbrs[j],POS_X])) - b[i,POS_X])
                b[i, GAP_Y] += kg * ((0.5 * (b[nbrs[k],POS_Y] + b[nbrs[j],POS_Y])) - b[i,POS_Y])
				        break
			      else
                delta =  ang[nbrs[k],i] - ang[nbrs[j],i]
				        if delta < 0
			              delta += 2π
				        end
				        if delta > π
					          p[i] = 2
					          b[i,PRM] = 1.
					          if rgf
                        b[i, DEBUG] += 1.0
                        b[i, GAP_X] += kg * ((0.5 * (b[nbrs[k],POS_X] + b[nbrs[j],POS_X])) - b[i,POS_X])
                        b[i, GAP_Y] += kg * ((0.5 * (b[nbrs[k],POS_Y] + b[nbrs[j],POS_Y])) - b[i,POS_Y])
					          end
					          break
				        end
			      end
		    end
	  end
	  return p
end



# ╔═╡ b5b64f0d-5f3d-49fc-9d3d-a2289b3e4b8f
function update_resultant(b, stability_factor, speed)
    n_agents = size(b)[1]
    for i in 1:n_agents
        mag_res = √(b[i,RES_X] ^ 2 + b[i,RES_Y] ^ 2)
        if mag_res > stability_factor * speed
            b[i,RES_X] = b[i,RES_X] / mag_res * speed
            b[i,RES_Y] = b[i,RES_Y] / mag_res * speed
        else
            b[i,RES_X] = 0.0
            b[i,RES_Y] = 0.0
		end
	end
end

# ╔═╡ 3a661119-9180-4cf6-9a7b-05cb05b0926c
function compute_step(b; scaling="linear",exp_rate=0.2,speed=0.05,perim_coord=false,stability_factor=0.,cb=3.0, rb=Array{Float64,2}([2. 2.; 2. 2.]),kc=Array{Float64,2}([0.15 0.15; 0.15 0.15]),kr=Array{Float64,2}([50. 50.; 50. 50.]),kd=0.,kg=0.,rgf=false)
	n_agents = size(b)[1]
	xv,yv,mag = all_pairs_mag(b, cb)
	
	p = on_perim(b,xv,yv,mag,cb,kg,rgf)
	
	compute_coh(b,xv,yv,mag,cb,kc,p)
	b[:,COH_X:COH_Y] ./= max.(b[:,COH_N], 1.)
	
	compute_rep_linear(b, xv, yv, mag, rb, kr, p)
	b[:,REP_X:REP_Y] ./= max.(b[:,REP_N], 1.)
	
    # compute the direction vectors
    b[:,DIR_X:DIR_Y] = kd .* (b[:,GOAL_X:GOAL_Y] .- b[:,POS_X:POS_Y])

    # compute the resultant of the cohesion, repulsion and direction vectors
    if perim_coord
        b[:,DIR_X:DIR_Y] .*= b[:,PRM]
	end
    b[:,RES_X:RES_Y] = b[:,COH_X:COH_Y] .+ b[:,GAP_X:GAP_Y] .+ b[:,REP_X:REP_Y] .+ b[:,DIR_X:DIR_Y]

    # normalise the resultant and update for speed, adjusted for stability
    update_resultant(b, stability_factor, speed)
	
	return xv,yv,mag,p
end

# ╔═╡ b1036e43-bb31-429d-94ec-c794a7addb60
function apply_step(b)
    """
    Assuming the step has been computed so that RES fields are up to date, update positions
    """
    #b[:,POS_X:POS_Y] .+= b[:,RES_X:RES_Y]
    #round.(b[:,POS_X:POS_Y], digits=9)
    for i in 1:size(b)[1]
        b[i,POS_X] = round(b[i,POS_X] + b[i,RES_X], digits=9) 
        b[i,POS_Y] = round(b[i,POS_Y] + b[i,RES_Y], digits=9)
    end
end

# ╔═╡ 679bba42-87fc-47b1-8c6c-e201ee923a7e
function load_swarm(path="swarm.json")
    state = JSON.parsefile(path)
	b = mk_swarm(state["agents"]["coords"][1], state["agents"]["coords"][2])
	params = state["params"]
	params = Dict(collect(zip(map(Symbol, collect(keys(params))), values(params))))
	params[:rb] = transpose(reshape(collect(Iterators.flatten(params[:rb])), (2,2)))
	params[:kc] = transpose(reshape(collect(Iterators.flatten(params[:kc])), (2,2)))
	params[:kr] = transpose(reshape(collect(Iterators.flatten(params[:kr])), (2,2)))
	return b, params
end

# ╔═╡ 04195ded-8a7e-4817-b395-cedcf73a4f6e
#b, parameters = load_swarm("/home/dk0/research/swarms/swarm_simulator/experiments/config/paper/base_400.json")


# ╔═╡ 4920cd4f-b6d5-4973-9118-91162e16d9c6
#with_terminal() do
#@btime 
#begin
    #cb = parameters[:cb]
    #xv,yv,mag = all_pairs_mag(b, cb)
#end
#end


# ╔═╡ 6524a573-ab33-4413-b6f1-3e8cbd0e7de5
#with_terminal() do
#@btime begin
	#kg = parameters[:kg]
	#rgf = parameters[:rgf]
	#p, _ = on_perim(b,xv,yv,mag,cb,kg,rgf)
#end
#end


# ╔═╡ 4382b55d-30ce-4a8b-8b50-d6b2ee96f842
#begin
  #kg = parameters[:kg]
  #rgf = parameters[:rgf]
  #p, ang = on_perim(b,xv,yv,mag,cb,kg,rgf)
#end


# ╔═╡ b55daeb4-eba6-4d0b-bdc4-82b20d1d6b6f
#with_terminal() do
#@btime begin
  #kc = parameters[:kc]
  #compute_coh(b,xv,yv,mag,cb,kc,p)
  #b[:,COH_X:COH_Y] ./= max.(b[:,COH_N], 1.)
#end
#end


# ╔═╡ ea86a862-21d5-46fb-b1b9-603e910488d9
#coh_xy = b[:,COH_X:COH_Y]


# ╔═╡ 85bd292f-2822-4015-ad78-200e1375917a
#coh_xy2 = b[:,COH_X:COH_Y]


# ╔═╡ eb21140a-d6fe-4a67-8eb6-95c614f217bf
#begin
	#kc = parameters[:kc]
	#compute_coh(b,xv,yv,mag,cb,kc,p)
	#b[:,COH_X:COH_Y] ./= max.(b[:,COH_N], 1.)
#end


# ╔═╡ d023595b-623f-4578-b843-1673ee9c6aeb
#with_terminal() do
#@btime begin
	#rb = parameters[:rb]
	#kr = parameters[:kr]
	#compute_rep_linear(b, xv, yv, mag, rb, kr, p)
	#b[:,REP_X:REP_Y] ./= max.(b[:,REP_N], 1.)
#end
#end


# ╔═╡ 2c8d3adf-b8a8-4a0b-bbd3-22c1a0e8f6ae
#begin
	#rb = parameters[:rb]
	#kr = parameters[:kr]
	#compute_rep_linear(b, xv, yv, mag, rb, kr, p)
	#b[:,REP_X:REP_Y] ./= max.(b[:,REP_N], 1.)
#end


# ╔═╡ 43c97abc-92ac-4aee-9e21-2f1e927c15b1
#with_terminal() do
#@btime begin
	#kd = parameters[:kd]
	#perim_coord = parameters[:perim_coord]
	#stability_factor = parameters[:stability_factor]
	#speed = parameters[:speed]
    ## compute the direction vectors
    #b[:,DIR_X:DIR_Y] = kd .* (b[:,GOAL_X:GOAL_Y] .- b[:,POS_X:POS_Y])

    ## compute the resultant of the cohesion, repulsion and direction vectors
    #if perim_coord
        #b[:,DIR_X:DIR_Y] .*= b[:,PRM]
	#end
    #b[:,RES_X:RES_Y] = b[:,COH_X:COH_Y] .+ b[:,GAP_X:GAP_Y] .+ b[:,REP_X:REP_Y] .+ b[:,DIR_X:DIR_Y]

    ## normalise the resultant and update for speed, adjusted for stability
    #update_resultant(b, stability_factor, speed)
#end
#end


# ╔═╡ 5d502ee8-bead-4900-bb58-cd92b96d88c4
b1, parameters1 = load_swarm("/home/dk0/research/swarms/swarm_simulator/experiments/config/paper/inner_400.json")


# ╔═╡ 906185ae-0c7a-4fb9-8375-0619e5f24853
# with_terminal() do
# @btime 
begin
	for i in 1:2000
			compute_step(b1; parameters1...)
			apply_step(b1)
	end
	prm = findall(b1[:,PRM] .> 0.)
	scatter(b1[:,POS_X],b1[:,POS_Y]; legend=false, markersize=2, markercolor=:black, aspect_ratio=:equal)
	scatter!(b1[prm,POS_X],b1[prm,POS_Y]; legend=false, markersize=2, markercolor=:red, aspect_ratio=:equal)
end
# end


# ╔═╡ 35f359e5-d9f8-462a-97b3-63d81c21d4a4
#[i for i in 1:size(b1[:,DEBUG])[1] if b1[i, DEBUG] > 0]

# ╔═╡ 65639045-43a7-40e0-be5d-8cdd9b6a7297
Threads.nthreads()

# ╔═╡ 5762d1a7-8800-43b5-a038-92e0d19fae91
#with_terminal() do
#@profile begin
	#compute_step(b1; parameters1...)
	#apply_step(b1)
#end
#end


# ╔═╡ d93216a4-1052-4cd8-b15e-9348fa3454f7
#with_terminal() do
	#Profile.print()
#end


# ╔═╡ b4981a7e-b6ec-4cc4-9d50-02c31c950ce7
#b[10:20,COH_X]


# ╔═╡ 16b77410-b927-460f-aba2-562a90b3c19a
#findall(v -> v == 2, p)


# ╔═╡ 009b73cb-c2ed-4050-886c-a7e593cc7f24
#size(b)[1]


# ╔═╡ Cell order:
# ╠═7b9704ab-5c42-4825-8661-1e391c3a3f26
# ╠═fb288ecc-468a-4bb5-8f53-6b1b695b75e0
# ╠═16b91667-fe1d-4173-a301-f7e1694c7bb0
# ╠═1d910032-0755-4343-8a7d-530514f7400f
# ╠═533ec0e4-465d-4ac4-ac93-17d95692343f
# ╠═b246f7ae-a50a-4f1e-8ad4-35a28058f662
# ╠═a53416e2-0f0d-4da1-b3e7-53d609291594
# ╠═3f0339d3-14a5-43cd-9f36-64d49356f443
# ╠═5fbe1265-2a93-431a-88f2-7108723f8995
# ╠═13fb8202-56fe-4197-8c78-4e997ba3d86b
# ╠═a32771e7-5408-4674-aee9-19428fdae26a
# ╠═e274cd80-70f6-4795-b6d3-a17bc91eba0d
# ╠═951547d1-36d9-4105-aeab-5b150f43d1b3
# ╠═7c192780-1f9d-4fd3-8e71-edae1560cec3
# ╠═6e76ec47-b9b6-4644-85d4-8b07116633a6
# ╠═6013f4fa-c799-4178-8f7d-776bc81843dc
# ╠═3c0e9c83-ee36-4755-a349-1b7a18b762de
# ╠═5c9be6fc-30bc-480b-8386-f2206f86ef5a
# ╠═dc2d4b1d-561b-4339-ae25-2d5b1b08cc2d
# ╠═b5b64f0d-5f3d-49fc-9d3d-a2289b3e4b8f
# ╠═3a661119-9180-4cf6-9a7b-05cb05b0926c
# ╠═b1036e43-bb31-429d-94ec-c794a7addb60
# ╠═679bba42-87fc-47b1-8c6c-e201ee923a7e
# ╠═04195ded-8a7e-4817-b395-cedcf73a4f6e
# ╠═4920cd4f-b6d5-4973-9118-91162e16d9c6
# ╠═6524a573-ab33-4413-b6f1-3e8cbd0e7de5
# ╠═4382b55d-30ce-4a8b-8b50-d6b2ee96f842
# ╠═b55daeb4-eba6-4d0b-bdc4-82b20d1d6b6f
# ╠═ea86a862-21d5-46fb-b1b9-603e910488d9
# ╠═85bd292f-2822-4015-ad78-200e1375917a
# ╠═eb21140a-d6fe-4a67-8eb6-95c614f217bf
# ╠═d023595b-623f-4578-b843-1673ee9c6aeb
# ╠═2c8d3adf-b8a8-4a0b-bbd3-22c1a0e8f6ae
# ╠═43c97abc-92ac-4aee-9e21-2f1e927c15b1
# ╠═5d502ee8-bead-4900-bb58-cd92b96d88c4
# ╠═906185ae-0c7a-4fb9-8375-0619e5f24853
# ╠═35f359e5-d9f8-462a-97b3-63d81c21d4a4
# ╠═65639045-43a7-40e0-be5d-8cdd9b6a7297
# ╠═101d06b9-2e3c-4e62-98e4-a275ad860c61
# ╠═5762d1a7-8800-43b5-a038-92e0d19fae91
# ╠═d93216a4-1052-4cd8-b15e-9348fa3454f7
# ╠═b4981a7e-b6ec-4cc4-9d50-02c31c950ce7
# ╠═16b77410-b927-460f-aba2-562a90b3c19a
# ╠═009b73cb-c2ed-4050-886c-a7e593cc7f24
# ╠═eaafe1e8-6873-490a-9caf-339e17d39ebc
# ╠═248d7979-e8da-408d-88da-97e52d84648e

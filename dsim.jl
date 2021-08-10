using Agents, JSON

mutable struct DAgent <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    speed::Float64
    cb::Float64
    rb::Matrix{Float64}
    kc::Matrix{Float64}
    kr::Matrix{Float64}
    kg::Float64
    rgf::Bool
    prm::UInt8
    debug_counter::Float64
end

function d_load_swarm(path="swarm.json")
    state = JSON.parsefile(path)
    coords::Vector{NTuple{2,Float64}} = collect(zip(state["agents"]["coords"][1], state["agents"]["coords"][2]))
    coords = map(c -> (c[1] + 25.0, c[2] + 25.0), coords)
	  params = state["params"]
	  params = Dict(collect(zip(map(Symbol, collect(keys(params))), values(params))))
    params[:rb] = collect(transpose(reshape(collect(Iterators.flatten(params[:rb])), (2,2))))
    params[:kc] = collect(transpose(reshape(collect(Iterators.flatten(params[:kc])), (2,2))))
    params[:kr] = collect(transpose(reshape(collect(Iterators.flatten(params[:kr])), (2,2))))
	  return coords, params
end

function d_mk_swarm(coords::Vector{NTuple{2,Float64}}, params::Dict{Symbol, Any}, extent=(100,100))
    space2d = ContinuousSpace(extent)
    model = ABM(DAgent, space2d, scheduler = Schedulers.randomly)
    for i in 1:length(coords)
        agent = DAgent(i, coords[i], params[:speed], params[:cb], params[:rb], params[:kc],
                       params[:kr], params[:kg], params[:rgf], 1, 0.0)
        add_agent_pos!(agent, model)
    end
    return model
end

function distance(pos, pos_)
    return hypot(pos[1] - pos_[1], pos[2] - pos_[2])
end

function all_nearby_ids(ag, swarm)
    return filter(id -> id != ag, [i  for i in 1:nagents(swarm) if distance(swarm[ag].pos, swarm[i].pos) ≤ swarm[ag].cb])
end

function agent_step!(b, swarm)
    #nids = collect(nearby_ids(b, swarm, b.cb, exact=true))
    nids = all_nearby_ids(b.id, swarm)
    sensor_data = [(swarm[id].pos, swarm[id].prm) for id in nids]
    n_nids = length(nids)
    pos = view([v[1] for v in sensor_data], 1:n_nids)
    xp = view([v[1][1] for v in sensor_data], 1:n_nids)
    yp = view([v[1][2] for v in sensor_data], 1:n_nids)
    ps = view([v[2] for v in sensor_data], 1:n_nids)
    xv = xp .- b.pos[1]
    yv = yp .- b.pos[2]
    v_gap = (0.0, 0.0)
    v_coh = (0.0, 0.0)
    v_rep = (0.0, 0.0)
    v_res = (0.0, 0.0)
    b.prm = 1
    ang = sort!(collect(zip(atan.(yv, xv), 1:n_nids)))
    if n_nids < 3
        b.prm = 2
    else
        for j in 1:n_nids
            k = (j % n_nids) + 1
            ij = ang[j][2]
            ik = ang[k][2]
            #if edistance(pos[ik], pos[ij], swarm) > b.cb
            if distance(pos[ik], pos[ij]) > b.cb
                b.prm = 2
                v_gap = b.kg .* ((0.5 .* (pos[ik] .+ pos[ij])) .- b.pos)
                #b.debug_counter += 1.0
                break
            else
                δ = ang[k][1] - ang[j][1]
                #b.debug_counter += 1.0
                #println("δ is $δ")
                if δ < 0
                    δ += 2π
                end
                if δ > π
                    b.prm = 2
                    #b.debug_counter += 1.0
                    if b.rgf
                        b.debug_counter += 1.0
                        v_gap = b.kg .* ((0.5 .* (pos[ik] .+ pos[ij])) .- b.pos)
                    end
                    break
                end
            end
        end
    end
    n_reps = 0
    for i in 1:n_nids
        bb_ = pos[i] .- b.pos
        v_coh = v_coh .+ b.kc[b.prm, ps[i]] .* bb_
        ri = b.rb[b.prm,ps[i]]
        #mag_bb_ = edistance(b.pos, pos[i], swarm)
        mag_bb_ = distance(b.pos, pos[i])
        if mag_bb_ ≤ ri
            n_reps += 1
            v_rep = v_rep .+ b.kr[b.prm,ps[i]] * (1.0 - ri / mag_bb_) .* bb_
        end
    end
    v_coh ./ max.(n_nids, 1.0)
    v_rep ./ max.(n_reps, 1.0)
    v_res = v_coh .+ v_gap .+ v_rep
    mag_res = hypot(v_res[1], v_res[2])
    if mag_res > 0.0
        v_res = v_res ./ mag_res .* b.speed
    else
        v_res = (0.0, 0.0)
    end
    move_agent!(b, b.pos .+ v_res, swarm)
end
#function agent_step!(b, swarm)
    #nbrs = collect(nearby_agents(b, swarm, b.cb, exact=true))
    #n_nbrs = length(nbrs)
    #xv = map(nbr -> nbr.pos[1] - b.pos[1], nbrs)
    #yv = map(nbr -> nbr.pos[2] - b.pos[2], nbrs)
    #mag = hypot.(xv, yv)
    #ang = atan.(yv, xv) .+ π
    #s_ang = sort!(collect(zip(ang, 1:n_nbrs)))
    #for j in 1:n_nbrs
        #k = (j % n_nbrs) + 1
        #if mag[s_ang[k][2]] 
    #coh_vec = sum(collect.(map(nbr -> (nbr.pos .- b.pos) .* b.kc[b.prm, nbr.prm], nbrs)), dims=1) ./ n_nbrs
    #return xv, yv, mag, ang, nbrs, s_ang, coh_vec
#end

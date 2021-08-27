using Agents, JSON, InteractiveDynamics, GLMakie

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
    v_coh::NTuple{2,Float64}
    v_gap::NTuple{2,Float64}
    v_rep::NTuple{2,Float64}
    v_res::NTuple{2,Float64}
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
                       params[:kr], params[:kg], params[:rgf], 1, (0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0), 0.0)
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


#function agent_step!(b, swarm)
    ##nids = collect(nearby_ids(b, swarm, b.cb, exact=true))
    #nids = all_nearby_ids(b.id, swarm)
    #sensor_data = [(swarm[id].pos, swarm[id].prm) for id in nids]
    #n_nids = length(nids)
    #pos = view([v[1] for v in sensor_data], 1:n_nids)
    #xp = view([v[1][1] for v in sensor_data], 1:n_nids)
    #yp = view([v[1][2] for v in sensor_data], 1:n_nids)
    #ps = view([v[2] for v in sensor_data], 1:n_nids)
    #xv = xp .- b.pos[1]
    #yv = yp .- b.pos[2]
    #b.v_gap = (0.0, 0.0)
    #b.v_coh = (0.0, 0.0)
    #b.v_rep = (0.0, 0.0)
    #b.v_res = (0.0, 0.0)
    #b.prm = 1
    #ang = sort!(collect(zip(atan.(yv, xv), 1:n_nids)))
    ##ang = sort!(collect(zip([α ≥ 0 ? α : α+2π for α in atan.(yv, xv)], 1:n_nids)))
    #if n_nids < 3
        #b.prm = 2
    #else
        #for j in 1:n_nids
            #k = (j % n_nids) + 1
            #ij = ang[j][2]
            #ik = ang[k][2]
            ##if edistance(pos[ik], pos[ij], swarm) > b.cb
            #if distance(pos[ik], pos[ij]) > b.cb
                #b.prm = 2
                #b.v_gap = b.kg .* ((0.5 .* (pos[ik] .+ pos[ij])) .- b.pos)
                ##b.debug_counter += 1.0
                #break
            #else
                #δ = ang[k][1] - ang[j][1]
                ##b.debug_counter += 1.0
                ##println("δ is $δ")
                #if δ < 0
                    #δ += 2π
                #end
                #if δ > π
                    #b.prm = 2
                    ##b.debug_counter += 1.0
                    #if b.rgf
                        #b.debug_counter += 1.0
                        #b.v_gap = b.kg .* ((0.5 .* (pos[ik] .+ pos[ij])) .- b.pos)
                    #end
                    #break
                #end
            #end
        #end
    #end
    #n_reps = 0
    #for i in 1:n_nids
        #bb_ = pos[i] .- b.pos
        ##bb_ = (xv[i], yv[i])
        #b.v_coh = b.v_coh .+ (b.kc[b.prm, ps[i]] .* bb_)
        #ri = b.rb[b.prm,ps[i]]
        ##mag_bb_ = edistance(b.pos, pos[i], swarm)
        #mag_bb_ = distance(b.pos, pos[i])
        #if mag_bb_ ≤ ri
            #n_reps += 1
            #b.v_rep = b.v_rep .+ (1.0 - (ri / mag_bb_)) .* bb_ .* b.kr[b.prm,ps[i]]
        #end
    #end
    #b.v_coh = b.v_coh ./ max.(n_nids, 1.0)
    #b.v_rep = b.v_rep ./ max.(n_reps, 1.0)
    #b.v_res = b.v_coh .+ b.v_gap .+ b.v_rep
    #mag_res = hypot(b.v_res[1], b.v_res[2])
    #if mag_res > 0.0
        #b.v_res = b.v_res ./ mag_res .* b.speed
    #else
        #b.v_res = (0.0, 0.0)
    #end
    ##move_agent!(b, b.pos .+ b.v_res, swarm)
#end

function observe_state(b, swarm)
    nids = all_nearby_ids(b.id, swarm)
    sensor_data = [(swarm[id].pos, swarm[id].prm) for id in nids]
    n_nids = length(nids)
    pos = view([v[1] for v in sensor_data], 1:n_nids)
    xp = view([v[1][1] for v in sensor_data], 1:n_nids)
    yp = view([v[1][2] for v in sensor_data], 1:n_nids)
    ps = view([v[2] for v in sensor_data], 1:n_nids)
    xv = xp .- b.pos[1]
    yv = yp .- b.pos[2]
    return nids, pos, xv, yv, ps
end

function compute_perimeter_status!(b, pos, xv, yv)
    b.prm = 1
    n_nids = length(pos)
    ang = sort!(collect(zip(atan.(yv, xv), 1:n_nids)))
    b.v_gap = (0.0, 0.0)
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
                b.v_gap = b.kg .* ((0.5 .* (pos[ik] .+ pos[ij])) .- b.pos)
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
                        b.v_gap = b.kg .* ((0.5 .* (pos[ik] .+ pos[ij])) .- b.pos)
                    end
                    break
                end
            end
        end
    end
    return nothing
end

function compute_cohesion!(b, xv, yv, ps)
    b.v_coh = (0.0, 0.0)
    n_nids = length(xv)
    for i in 1:n_nids
        b.v_coh = b.v_coh .+ (b.kc[b.prm, ps[i]] .* (xv[i], yv[i]))
    end
    b.v_coh = b.v_coh ./ max.(n_nids, 1.0)
end

function compute_repulsion!(b, pos, xv, yv, ps)
    b.v_rep = (0.0, 0.0)
    n_reps = 0
    for i in 1:length(pos)
        ri = b.rb[b.prm,ps[i]]
        mag_bb_ = distance(b.pos, pos[i])
        if mag_bb_ ≤ ri 
            n_reps += 1
            b.v_rep = b.v_rep .+ (1.0 - (ri / mag_bb_)) .* (xv[i], yv[i]) .* b.kr[b.prm,ps[i]]
        end
    end
    b.v_rep = b.v_rep ./ max.(n_reps, 1.0)
end

function compute_resultant!(b)
    b.v_res = b.v_coh .+ b.v_gap .+ b.v_rep
    mag_res = hypot(b.v_res[1], b.v_res[2])
    if mag_res > 0.0
        b.v_res = b.v_res ./ mag_res .* b.speed
    else
        b.v_res = (0.0, 0.0)
    end
end

function agent_step!(b, swarm)
    nids, pos, xv, yv, ps = observe_state(b, swarm)
    compute_perimeter_status!(b, pos, xv, yv)
    compute_cohesion!(b, xv, yv, ps)
    compute_repulsion!(b, pos, xv, yv, ps)
    compute_resultant!(b)
    move_agent!(b, round.(b.pos .+ b.v_res, digits=9), swarm)
end

ac(a) = a.prm == 2 ? "#ff0000" : "#000000"

function make_video(path, n_frames)
    d_coords, d_params = d_load_swarm(path)
    swarm = d_mk_swarm(d_coords, d_params, (50,50))
    name = splitext(splitdir(path)[2])[1]
		abm_video(
        "./mp4/$name.mp4", swarm, agent_step!;
        framerate = 20, frames = n_frames,
        title=name, ac=ac, as=4
    )
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

#include("simulator.jl")

#function step_in_sync(swarm, b, parameters)
    #xv, yv, mag, p = compute_step(b; parameters...)
    #apply_step(b)
    #for j in 1:nagents(swarm)
        #agent_step!(swarm[j], swarm)
    #end
    #for j in 1:nagents(swarm)
        #move_agent!(swarm[j], swarm[j].pos .+ swarm[j].v_res, swarm)
    #end
    #return xv, yv, mag, p
#end

#function run_test(path; n_steps=1000, eps=10.0e-9)
    #d_coords, d_params = d_load_swarm(path)
    #swarm = d_mk_swarm(d_coords, d_params, (50,50))
    #b, parameters = load_swarm(path)
    #step_number = 0
    #bpos = collect(zip(b[:,POS_X], b[:,POS_Y])) 
    ##while [i for i in 1:nagents(swarm) if abs(swarm[i].pos[1] - bpos[i][1]) > eps || abs(swarm[i].pos[2] - bpos[i][2]) > eps] == [] && step_number <= n_steps
    ##while [swarm[i].pos for i in 1:nagents(swarm)] == bpos && step_number <= n_steps
    #local xv, yv, mag, p
    #while [i for i in 1:nagents(swarm) if abs(swarm[i].pos[1] - b[i,POS_X]) > eps || abs(swarm[i].pos[2] - b[i,POS_Y]) > eps] == [] && step_number < n_steps
        #xv, yv, mag, p = step_in_sync(swarm, b, parameters)
        #step_number += 1
    #end
    #return step_number, swarm, b, parameters, xv, yv, mag, p
#end

#function one_step_test(path)
    #d_coords, d_params = d_load_swarm(path)
    #swarm = d_mk_swarm(d_coords, d_params, (50,50))
    #b, parameters = load_swarm(path)
    #xv, yv, mag, p = compute_step(b; parameters...)
    #for i in 1:nagents(swarm)
        #nids, pos, xv, yv, ps = observe_state(swarm[i], swarm)
        #compute_perimeter_status!(swarm[i], pos, xv, yv)
    #end
    #for i in 1:nagents(swarm)
        #nids, pos, xv, yv, ps = observe_state(swarm[i], swarm)
        #compute_cohesion!(swarm[i], xv, yv, ps)
        #compute_repulsion!(swarm[i], pos, xv, yv, ps)
        #compute_resultant!(swarm[i])
    #end
    #return swarm, b, parameters, xv, yv, mag, p
#end

#function d_compute_step(swarm)
    #for i in 1:nagents(swarm)
        #nids, pos, xv, yv, ps = observe_state(swarm[i], swarm)
        #compute_perimeter_status!(swarm[i], pos, xv, yv)
    #end
    #for i in 1:nagents(swarm)
        #nids, pos, xv, yv, ps = observe_state(swarm[i], swarm)
        #compute_cohesion!(swarm[i], xv, yv, ps)
        #compute_repulsion!(swarm[i], pos, xv, yv, ps)
        #compute_resultant!(swarm[i])
    #end
#end

#function d_apply_step(swarm)
    #for i in 1:nagents(swarm)
        ##move_agent!(swarm[i], swarm[i].pos .+ swarm[i].v_res, swarm)
        #swarm[i].pos = round.(swarm[i].pos .+ swarm[i].v_res, digits=9)
    #end
#end

#function consistent_states(swarm, b, eps)
    #result = true
    #for i in 1:nagents(swarm)
        #if any(abs.(swarm[i].pos .- Tuple(b[i,POS_X:POS_Y])) .> eps)
            #result = false
            #break
        #end
        #if any(abs.(swarm[i].v_coh .- Tuple(b[i,COH_X:COH_Y])) .> eps)
            #result = false
            #break
        #end
        #if any(abs.(swarm[i].v_rep .- Tuple(b[i,REP_X:REP_Y])) .> eps)
            #result = false
            #break
        #end
        #if any(abs.(swarm[i].v_gap .- Tuple(b[i,GAP_X:GAP_Y])) .> eps)
            #result = false
            #break
        #end
        #if swarm[i].prm != Int(b[i,PRM] + 1.0)
            #result = false
            #break
        #end
    #end
    #return result
#end

#function n_step_test(path, n_steps=1000, eps=10e-9)
    #d_coords, d_params = d_load_swarm(path)
    #swarm = d_mk_swarm(d_coords, d_params, (50,50))
    #b, parameters = load_swarm(path)
    #local xv, yv, mag, p 
    #while  consistent_states(swarm, b, eps) && n_steps > 0
        #n_steps -= 1
        #xv, yv, mag, p = compute_step(b; parameters...)
        #apply_step(b)
        #d_compute_step(swarm)
        #d_apply_step(swarm)
    #end
    #return n_steps, swarm, b, parameters, xv, yv, mag, p
#end

#function compare_coh_x(swarm, b)
    #test_t = collect(zip([swarm[i].v_coh[1] for i in 1:nagents(swarm)], b[:,COH_X]))
    #anom = [i for i in 1:nagents(swarm) if test_t[i][1] != test_t[i][2]]
    #return test_t, anom, test_t[anom]
#end

#function compare_coh_y(swarm, b)
    #test_t = collect(zip([swarm[i].v_coh[2] for i in 1:nagents(swarm)], b[:,COH_Y]))
    #anom = [i for i in 1:nagents(swarm) if test_t[i][1] != test_t[i][2]]
    #return test_t, anom, test_t[anom]
#end

#function compare_gap_x(swarm, b)
    #test_t = collect(zip([swarm[i].v_gap[1] for i in 1:nagents(swarm)], b[:,GAP_X]))
    #anom = [i for i in 1:nagents(swarm) if test_t[i][1] != test_t[i][2]]
    #return test_t, anom, test_t[anom]
#end

#function compare_gap_y(swarm, b)
    #test_t = collect(zip([swarm[i].v_gap[2] for i in 1:nagents(swarm)], b[:,GAP_Y]))
    #anom = [i for i in 1:nagents(swarm) if test_t[i][1] != test_t[i][2]]
    #return test_t, anom, test_t[anom]
#end

#function compare_rep_x(swarm, b)
    #test_t = collect(zip([swarm[i].v_rep[1] for i in 1:nagents(swarm)], b[:,REP_X]))
    #anom = [i for i in 1:nagents(swarm) if test_t[i][1] != test_t[i][2]]
    #return test_t, anom, test_t[anom]
#end

#function compare_rep_y(swarm, b)
    #test_t = collect(zip([swarm[i].v_rep[2] for i in 1:nagents(swarm)], b[:,REP_Y]))
    #anom = [i for i in 1:nagents(swarm) if test_t[i][1] != test_t[i][2]]
    #return test_t, anom, test_t[anom]
#end

#function compare_res_x(swarm, b)
    #test_t = collect(zip([swarm[i].v_res[1] for i in 1:nagents(swarm)], b[:,RES_X]))
    #anom = [i for i in 1:nagents(swarm) if test_t[i][1] != test_t[i][2]]
    #return test_t, anom, test_t[anom]
#end

#function compare_res_y(swarm, b)
    #test_t = collect(zip([swarm[i].v_res[2] for i in 1:nagents(swarm)], b[:,RES_Y]))
    #anom = [i for i in 1:nagents(swarm) if test_t[i][1] != test_t[i][2]]
    #return test_t, anom, test_t[anom]
#end

#function compare_pos_x(swarm, b)
    #test_t = collect(zip([swarm[i].pos[1] for i in 1:nagents(swarm)], b[:,POS_X]))
    #anom = [i for i in 1:nagents(swarm) if test_t[i][1] != test_t[i][2]]
    #return test_t, anom, test_t[anom]
#end

#function compare_pos_y(swarm, b)
    #test_t = collect(zip([swarm[i].pos[2] for i in 1:nagents(swarm)], b[:,POS_Y]))
    #anom = [i for i in 1:nagents(swarm) if test_t[i][1] != test_t[i][2]]
    #return test_t, anom, test_t[anom]
#end

#function testeps(t; eps = 10.0 ^ -9)
    #return [i for i in 1:length(t) if abs(t[i][1] - t[i][2]) > eps]
#end

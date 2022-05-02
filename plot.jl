

using DataStructures
using Dates
using DrWatson
using Plots
# using PlotlyJS
include("project2_jl/algo_zeroth_order.jl")
include("project2_jl/algo_penalty_barrier.jl")
include("project2_jl/helpers.jl")
include("project2_jl/simple.jl")

time_now = Dates.format(now(), "Y-mm-dd-HH:MM:SS")
outputdir(args...) = projectdir("output", args...)
outputdir_subpath = outputdir(time_now)
mkdir(outputdir_subpath)

simple1_plot(x1, x2) = -x1 * x2 + 2.0 / (3.0 * sqrt(3.0))
simple1_constraint_plot(x1, x2) = [x1 + x2^2 - 1, -x1 - x2]
simple2_plot(x1, x2) = (1.0 - x1)^2 + 100.0 * (x2 - x1^2)^2
simple2_constraint_plot(x1, x2) = [(x1-1)^3 - x2 + 1, x1 + x2 - 2]

function get_x_hist(prob_name, f, ∇f, c, x0, n)
    x_hists = Dict()
    method = nothing
    if prob_name == "simple1"
        #* Solver 1
        method = HookeJeevesDynamic(α=0.4, γ=0.3)
        penalties = [penalty_l0, penalty_l2]
        weights = [1.5, 2.0]
        multipliers = [2.0, 2.0]
        _, x_hist  = penalty_method(method, f, ∇f, c, penalties, x0, 15, "_"; weights=weights, multipliers=multipliers)
        x_hists[string(typeof(method))] = x_hist

        #* Solver 2
        d = length(x0)
        μ = copy(x0)
        Σ = 1 * ones(d,d)
        Σ[diagind(Σ)] .= 2
        method = CEM(pop_size=100, elite_size=20, P = MvNormal(μ, Σ))
        penalties = [penalty_l0, penalty_l2]
        weights = [1.5, 2.0]
        multipliers = [2.0, 2.0]
        _, x_hist  = penalty_method(method, f, ∇f, c, penalties, x0, 15, 5; weights=weights, multipliers=multipliers)
        x_hists[string(typeof(method))] = x_hist

    elseif prob_name == "simple2" 
        #* Solver 1
        method = HookeJeevesDynamic(α=1.0, γ=0.5, ϵ=5e-1)
        penalties = [penalty_l0, penalty_l2]
        weights = [2.0, 2.0]
        multipliers = [2.0, 2.0]
        _, x_hist = penalty_method(method, f, ∇f, c, penalties, x0, 10, "_"; weights=weights, multipliers=multipliers)
        x_hists[string(typeof(method))] = x_hist

        #* Solver 2
        d = length(x0)
        μ = copy(x0)
        Σ = 2 * ones(d,d)
        Σ[diagind(Σ)] .= 5

        method = CEM(pop_size=72, elite_size=15, P = MvNormal(μ, Σ))
        penalties = [penalty_l0, penalty_l2]
        weights = [4.0, 4.0]
        multipliers = [2.0, 2.0]
        _, x_hist = penalty_method(method, f, ∇f, c, penalties, x0, 3, 3; weights=weights, multipliers=multipliers)
        x_hists[string(typeof(method))] = x_hist
    end

    return x_hists
end


for (prob_name, (f, ∇f, c, x_init_fn, n)) in PROBS
    (prob_name == "simple3") && continue

    data = DefaultDict(Vector)
    for seed in 1:3
        x0 = x_init_fn()
        x_hists = get_x_hist(prob_name, f, ∇f, c, x0, n)
        for (solver_name, x_hist) in x_hists
            f_hist = [f(x) for x in x_hist]
            c_hist = [maximum(c(x)) for x in x_hist]
            
            # if prob_name == "simple1"
            #     println("$(maximum(f_hist)), $(minimum(f_hist))")
            # end

            push!(
                data[solver_name], 
                (
                    seed = seed,
                    iteration=collect(1:length(f_hist)),
                    x=x_hist,
                    f=f_hist,
                    c=c_hist
                )
            )
        end
    end

    # Value Plots
    if prob_name == "simple2"
        for (solver_name, seed_runs) in data
            title = "$(uppercasefirst(string(f))) w/ $(solver_name)"
            for plot_type in ["output", "constraint"]
                ylabel = plot_type == "output" ? "f(x)" : "c(x)"
                for (i, (seed, iteration, _, f, c)) in enumerate(seed_runs)
                    y = plot_type == "output" ? f : c
                    i == 1 && plot(iteration, y, title=title, xlabel = "Iteration", ylabel = ylabel, label="Init 1")
                    i > 1 && plot!(iteration, y, label="Init $i")
                end
                savefig(outputdir(outputdir_subpath, "$(plot_type)_$(prob_name)_$(solver_name).png"))
            end
        end
    end

    # Contour Plot
    xr = -3:0.1:3
    yr = -3:0.1:3
    if prob_name in ["simple1", "simple2"]
        f_plot = (prob_name == "simple1") ? simple1_plot : simple2_plot
        c_plot = (prob_name == "simple1") ? simple1_constraint_plot : simple2_constraint_plot
        levels = (prob_name == "simple1") ? collect(range(-10,10,100)) : [10,25,50,100,200,250,300]

        c = map(c_plot, xr, yr)
        for (solver_name, seed_runs) in data
            title = "$(uppercasefirst(string(f))) w/ $(solver_name)"
            contour(xr, yr, f_plot,
                levels = levels, colorbar = false, c=cgrad(:viridis, rev = true), legend = false, title=title,
                xlims =(-3,3), ylims =(-3,3), xlabel = "x₁", ylabel = "x₂", aspectratio = :equal, clim =(2,500)
            )
            plot!([c[j][1] for j = 1:length(c)], [c[j][2] for j = 1:length(c)])

            for (i, (seed, iteration, x, _, _)) in enumerate(seed_runs)
                i == 1 && plot!([x[j][1] for j = 1:length(x)], [x[j][2] for j = 1:length(x)], color = :black, label="Init 1")
                i > 1 &&  plot!([x[j][1] for j = 1:length(x)], [x[j][2] for j = 1:length(x)], color = :black, label="Init $i")
            end
            savefig(outputdir(outputdir_subpath, "contour_$(prob_name)_$(solver_name).png"))
        end
    end
end

include("helpers.jl")
#TODO Reason over stoppping conditions for both methods, max_iters? evals_per_iter?

"""
Encourage infeasible solutions to go towards feasible set
"""
function penalty_method(M, f, ∇f, c, penalties, x, outer_max_iters, inner_max_iters; num_eval_termination = true, weights=[1.0], multipliers=[2.0])
    x_hist = []
    for i in 1:outer_max_iters
        unconstrained_form = x -> f(x) + sum(weight * fn(x,c) for (fn, weight) in zip(penalties, weights))
        x, x_hist_inner = solve!(M, unconstrained_form, x, inner_max_iters)
        weights .*= multipliers

        append!(x_hist, x_hist_inner)
        if penalties[1](x, c) == 0 # this assumes the first penalty function is count-based penalty
            break
        end
        # (i < outer_max_iters) && pop!(x_hist)
    end
    return x, x_hist
end

penalty_l0(x, c) = sum(c(x) .> 0)
penalty_l2(x, c) = sum((max.(c(x), 0)) .^ 2)


"""
Encourage feasible solutions to stay feasible
"""
function barrier_method(M, f, ∇f, c, barrier, x, inner_max_iters; num_eval_termination = true, weight=1.0, multiplier=2.0, ϵ=1e-3)
    #* need to intialize with feasible point by solving quadratic penalty
    x, _ = solve!(M, x -> penalty_l2(x,c), x, inner_max_iters)

    δ = Inf
    while δ > ϵ
        unconstrained_form = x -> f(x) + barrier(x, c) / weight
        x′, _ = solve!(M, unconstrained_form, x, inner_max_iters)
        δ = norm(x′ - x)
        x = x′
        weight *= multiplier
    end
    return x
end

barrier_inverse(x, c) = -sum(1 ./ c(x))
barrier_log(x, c) = -sum(y ≥ -1 ? log(-y) : 0 for y in c(x))

include("helpers.jl")

# Hooke Jeeves Dynamic w/ Eager execution
basis(i, n) = [k == i ? 1.0 : 0.0 for k in 1:n]
abstract type ZerothOrder end
Base.@kwdef mutable struct HookeJeevesDynamic <: ZerothOrder
    α = 1e-2
    ϵ = 1e-4
    γ = 0.5
    n = nothing
    evals_per_iter = nothing
    D = nothing # directions to search in, need to store it for permutating order
end
function init!(M::HookeJeevesDynamic, x)
    M.n = length(x)
    M.evals_per_iter = 2 * M.n
    M.D = [sgn * basis(i, M.n) for i in 1:M.n for sgn in (-1, +1)]
end
function step!(M::HookeJeevesDynamic, f, x, y, idx_best_prev)
    α, ϵ, γ = M.α, M.ϵ, M.γ
    improved, terminate, idx_best = false, false, 1

    x_best, y_best = x, y 
    d_best_prev = M.D[idx_best_prev]
    M.D = pushfirst!(deleteat!(M.D, idx_best_prev), d_best_prev)
    xs_new = [x + α * d for d in M.D]

    for (idx, x_new) in enumerate(xs_new)
        y_new = f(x_new)
        if y_new < y_best
            x_best, y_best = x_new, y_new
            improved = true
            idx_best = idx
            break
        end
    end

    M.α *= (!improved ? γ : 1)
    terminate = (M.α <= ϵ ? true : false)
    return x_best, y_best, terminate, idx_best
end

function solve(M::HookeJeevesDynamic, f, x, max_iters; num_eval_termination=true)
    init!(M, x)
    x_hist = [x]
    y, terminate, idx_best = f(x), false, 1

    while !terminate
        x, y, terminate, idx_best = step!(M, f, x, y, idx_best)
        push!(x_hist, x)

        if num_eval_termination && (count(f) >= max_iters - M.evals_per_iter)
            break
        end
    end
    return x, x_hist
end
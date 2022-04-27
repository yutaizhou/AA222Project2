using Distributions
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

function solve!(M::HookeJeevesDynamic, f, x, max_iters; num_eval_termination=true)
    init!(M, x)
    x_hist = [x]
    y, terminate, idx_best = f(x), false, 1

    while !terminate
        x, y, terminate, idx_best = step!(M, f, x, y, idx_best)
        push!(x_hist, x)

        # if num_eval_termination && (count(f) >= max_iters - M.evals_per_iter)
        #     break
        # end
    end
    return x, x_hist
end

#* Cross Entropy
Base.@kwdef mutable struct CEM <: ZerothOrder
    pop_size = 100
    elite_size = 10
    d = nothing # sample dimension
    P = nothing # proposal distribution

    evals_per_iter = pop_size
end
function init!(M::CEM, x; P = nothing)
    d = length(x)
    M.d = d

    if P === nothing
        # μ = copy(x)
        # Σ = Matrix(2I,d,d)
        # μ = [2/3, 1/√3]
        μ = [0, 0]
        Σ = [
            1.0 0.2
            0.2 2.0
        ]
        M.P = MvNormal(μ, Σ)
    else
        M.P = P
    end

    return M
end

function step!(M::CEM, f)
    P, pop_size, elite_size = M.P, M.pop_size, M.elite_size

    population = rand(P, pop_size)
    performance = [f(population[:,i]) for i in 1:pop_size]
    order = sortperm(performance)
    elites = population[:, order[1:elite_size]]  
    M.P = fit(typeof(P), elites)

    return M.P.μ
end

function solve!(M::CEM, f, x, max_iters; P = nothing, num_eval_termination = false)
    init!(M, x; P=P)

    x_hist = [x]
    for _ in max_iters
        x = step!(M, f)
        push!(x_hist, x)

        # if num_eval_termination && (count(f) >= max_iters - M.evals_per_iter)
        #     break
        # end
    end
    return x, x_hist
end
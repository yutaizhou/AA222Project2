#=
        project1.jl -- This is where the magic happens!

    All of your code must either live in this file, or be `include`d here.
=#

#=
    If you want to use packages, please do so up here.
    Note that you may use any packages in the julia standard library
    (i.e. ones that ship with the julia language) as well as Statistics
    (since we use it in the backend already anyway)
=#

# Example:
using LinearAlgebra
using Distributions

#=
    If you're going to include files, please do so up here. Note that they
    must be saved in project1_jl and you must use the relative path
    (not the absolute path) of the file in the include statement.

    [Good]  include("somefile.jl")
    [Bad]   include("/pathto/project1_jl/somefile.jl")
=#

# Example
# include("myfile.jl")
include("algo_zeroth_order.jl")
include("algo_penalty_barrier.jl")
include("algo_simplex.jl")


"""
    optimize(f, g, c, x0, n, prob)

Arguments:
    - `f`: Function to be optimized
    - `g`: Gradient function for `f`
    - `c`: Constraint function for 'f'
    - `x0`: (Vector) Initial position to start from
    - `n`: (Int) Number of evaluations allowed. Remember `g` costs twice of `f`
    - `prob`: (String) Name of the problem. So you can use a different strategy for each problem. E.g. "simple1", "secret2", etc.

Returns:
    - The location of the minimum
"""
function optimize(f, ∇f, c, x0, n, prob_name)
    if prob_name == "simple1"
        method = HookeJeevesDynamic(α=0.4, γ=0.3)
        penalties = [penalty_l0, penalty_l2]
        weights = [1.5, 2.0]
        multipliers = [2.0, 2.0]
        x = penalty_method(method, f, ∇f, c, penalties, x0, 15, "_"; weights=weights, multipliers=multipliers)
        return x

    elseif prob_name == "simple2"
        method = HookeJeevesDynamic(α=1.0, γ=0.5, ϵ=5e-1)
        penalties = [penalty_l0, penalty_l2]
        weights = [2.0, 2.0]
        multipliers = [2.0, 2.0]
        x = penalty_method(method, f, ∇f, c, penalties, x0, 10, "_"; weights=weights, multipliers=multipliers)
        return x

    elseif prob_name == "simple3"
        method = HookeJeevesDynamic(α=0.3, γ=0.3)
        penalties = [penalty_l0, penalty_l2]
        weights = [2.0, 2.0]
        multipliers = [2.0, 2.0]
        x = penalty_method(method, f, ∇f, c, penalties, x0, 20, "_"; weights=weights, multipliers=multipliers)    
        return x

    elseif prob_name == "secret1"
        method = HookeJeevesDynamic(α=6.0, γ=0.5, ϵ=5e-3)
        penalties = [penalty_l0, penalty_l2]
        weights = [4.0, 4.0]
        multipliers = [2.0, 2.0]
        x = penalty_method(method, f, ∇f, c, penalties, x0, 4, "_"; weights=weights, multipliers=multipliers)
        return x

    
    elseif prob_name == "secret2"
        method = HookeJeevesDynamic(α=6.0, γ=0.5, ϵ=5e-3)
        penalties = [penalty_l0, penalty_l2]
        weights = [4.0, 4.0]
        multipliers = [2.0, 2.0]
        x = penalty_method(method, f, ∇f, c, penalties, x0, 4, "_"; weights=weights, multipliers=multipliers)
        return x
    end
end
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
    if prob_name == "simple1" #* 0.0645 - rank 7
        method = HookeJeevesDynamic(α=0.4, γ=0.3)
        penalties = [penalty_l0, penalty_l2]
        weights = [1.5, 2.0]
        multipliers = [2.0, 2.0]
        x = penalty_method(method, f, ∇f, c, penalties, x0, 15, 15; weights=weights, multipliers=multipliers)
        return x

    elseif prob_name == "simple2" #* 45.29 - rank 19 do better
        method = HookeJeevesDynamic(α=0.3, γ=0.3)
        barrier = barrier_inverse
        weight = 2.0
        multiplier = 2
        x = barrier_method(method, f, ∇f, c, barrier, x0, n; weight=weight, multiplier=multiplier)
        return x

    elseif prob_name == "simple3" #* 0.173 - rank 10
        method = HookeJeevesDynamic(α=0.3, γ=0.3)
        penalties = [penalty_l0, penalty_l2]
        weights = [1.75, 1.75]
        multipliers = [2.0, 2.0]
        x = penalty_method(method, f, ∇f, c, penalties, x0, 20, n; weights=weights, multipliers=multipliers)    
        return x

    elseif prob_name == "secret1" #* 0.1207 - rank 10
        method = CEM(pop_size = 40, elite_size = 10)
        penalties = [penalty_l0, penalty_l2]
        weights = [1.75, 1.75]
        multipliers = [2.0, 2.0]
        x = penalty_method(method, f, ∇f, c, penalties, x0, 20, n; weights=weights, multipliers=multipliers)    
        return x
    
    elseif prob_name == "secret2"
        
    end

end
using LinearAlgebra

Base.@kwdef mutable struct LinearProgram
    A
    b
    c
end

"""
B: partition
LP: Linear program in equality form
"""
function get_vertex(B, LP)
    A, b, c = LP.A, LP.b, LP.c

    B_idc = sort!(collect(B))
    Aᵦ = A[:, B_idc]
    xᵦ = Aᵦ \ b

    x = zeros(length(c))
    x[B_idc] = xᵦ

    return x
end

"""
B: partition
LP: Linear program in equality form
q: entering index
"""
function edge_transition(LP, B, q)
    A = LP.A

    n = size(A,2)
    B_idc = sort(B)
    n_idc = sort!(setdiff(1:n, B))

    Aᵦ = A[:, B_idc]
    Aq = A[:, n_idc[q]]    
    d = Aᵦ \ Aq
    xᵦ = Aᵦ \ b

    p, xq′ = 0, Inf
    for i in 1:length(d)
        if d[i] > 0
            v = xᵦ[i] / d[i]
            if v < xq′
                p, xq′ = i, v
            end
        end
    end
 
    return (p, xq′)
end

"""
One step of simplex optimization using Greedy heuristic
"""
function simplex_step!(B, LP)
    A, b, c = LP.A, LP.b, LP.c

    n = size(A,2)
    B_idc = sort(B)
    n_idc = sort!(setdiff(1:n, B))
    Aᵦ = A[:, B_idc]
    Aᵥ = A[:, n_idc]

    # xᵦ = Aᵦ \ b
    cᵦ = c[B_idc]
    λ = tr(Aᵦ) \ cᵦ
    cᵥ = c[n_idc]
    μᵥ = cᵥ - tr(Aᵥ) * λ 

    q, p, xq′, Δ = 0, 0, Inf, Inf
    for i in 1:length(μᵥ)
        if μᵥ[i] < 0 
            pᵢ, xᵢ′ = edge_transition(LP, B, i)
            Δᵢ =  μᵥ[i] * xᵢ′
            if Δᵢ < Δ
                q, p , xq′, Δ = i, pᵢ, xᵢ′, Δᵢ
            end
        end
    end
    if q == 0
        return (B, true) # optimal vertex found!
    end

    if isinf(xq′)
        error("Unbounded!")
    end
    
    j = findfirst(isequal(B_idc[p]), B)
    B[j] = n_idc[q] # swap indices
    return (B, false) # new vertex but not optimal
end

# Simplex solver when initial partition is known
function simplex_solve!(B, LP)
    done = false
    while !done
        B, done = simplex_step!(B, LP)
    end
    return B
end

# Simplex solver when initial partition is unknown
function simplex_solve(LP)
    A, b, = LP.A, LP.b

    # Find the initial parition
    m, n = size(A)
    z = ones(m)
    Z = Matrix(Diagonal([j ≥ 0 ? 1: -1 for j in b]))

    A′ = hcat(A,Z)
    b′ = b
    c′ = vcat(zeros(n), z)

    LP_aux = LinearProgram(A′, b′, c′)
    B = collect(1:m) .+ n
    simplex_solve!(B, LP_aux)

    if any(i -> i > n, B)
        error("infeasible")
    end

    # Use the intial parition
    A′′ = [A,          Matrix(1.0I, m, m);
          zeros(m,n)  Matrix(1.0I, m, m)]
    b′′ = vcat(b, zeros(m))
    c′′ = c′
    
    LP = LinearProgram(A′′, b′′, c′′)
    simplex_solve!(B, LP)
    return get_vertex(B, LP)[1:n]
end

function dual_certificate(LP, x, λ, ϵ=1e-6)
    A, b, c = LP.A, LP.b, LP.c

    prime_feasible = all(x.≥ 0) && (A*b ≈ b)
    dual_feasible  = all(tr(A) * λ .≤ c)
    gap_feasible   = isapprox(c ⋅ x, b ⋅ λ, atol=ϵ)

    return prime_feasible && dual_feasible && gap_feasible
end
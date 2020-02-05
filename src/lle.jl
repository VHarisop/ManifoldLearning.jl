# Locally Linear Embedding (LLE)
# ------------------------
# Nonlinear dimensionality reduction by locally linear embedding,
# Roweis, S. & Saul, L., Science 290:2323 (2000)

using Logging

#### LLE type
struct LLE{T <: Real} <: AbstractDimensionalityReduction
    k::Int
    λ::AbstractVector{T}
    proj::Projection{T}

    LLE{T}(k::Int, λ::AbstractVector{T}, proj::Projection{T})  where T = new(k, λ, proj)
end

## properties
outdim(R::LLE) = size(R.proj, 1)
eigvals(R::LLE) = R.λ
neighbors(R::LLE) = R.k

## show
summary(io::IO, R::LLE) = print(io, "LLE(outdim = $(outdim(R)), neighbors = $(neighbors(R)))")

## interface functions
function fit(::Type{LLE}, X::AbstractMatrix{T};
             maxoutdim::Int=2, k::Int=12, tol::Real=1e-5, use_naive=false) where {T<:Real}
    # Construct NN graph
    D, E = find_nn(X, k)
    _, C = largest_component(SimpleWeightedGraph(adjmat(D,E)))
    X = X[:, C]
    n = length(C)

    # Correct indexes of neighbors if more then one connected component
    Ec = E
    if size(E,2) != n
        R = Dict(zip(C, collect(1:n)))
        Ec = zeros(Int,k,n)
        for i in 1 : n
            Ec[:,i] = map(j->get(R,j,C[i]), E[:,C[i]])
        end
    end

    if k > maxoutdim
        @warn("k > maxoutdim: regularization will be used")
    else
        tol = 0
    end

    # Reconstruct weights and compute embedding:
    # M = (I - w)'(I - w) = I - w'I - Iw + w'w
    M = spdiagm(0 => fill(one(T), n))
    Ones = fill(one(T), k, 1)
    for i in 1 : n
        J = Ec[:,i]
        Z = view(X, :, J) .- view(X, :, i)
        G = transpose(Z)*Z
        G += I * tol # regularize
        w = vec(G \ Ones)
        w ./= sum(w)
        ww = w*transpose(w)
        for (l, j) in enumerate(J)
            M[i,j] -= w[l]
            M[j,i] -= w[l]
            for (m, jj) in enumerate(J)
                M[j,jj] = ww[l,m]
            end
        end
    end

    λ, V = specDecompose(M, tol, maxoutdim, use_naive)
    return LLE{T}(k, λ, rmul!(transpose(V), sqrt(n)))
end

transform(R::LLE) = R.proj


function specDecompose(M::AbstractArray, tol::Real, dim::Int, use_naive::Bool)
    @info("Entering: specDecompose")
    n, _ = size(M)
    V = Matrix(qr(randn(n, dim+1))); _, Q, Λ = schur(Symmetric(V' * Matrix(M * V)))
    V[:] = V * Q; gap = diff(sort(Λ))[1]; itIdx = 0
    while true
        itIdx += 1
        V = qr(M \ V)
        ϵ₂, ϵComp, Λ[:] = _2infPert(M, V, gap=gap)
        @info("it: $(itIdx) - ϵ₂: $(ϵ₂) - ϵI: $(ϵComp)")
        if use_naive && (ϵ₂ ≤ tol * sqrt(n))
            break
        end
        (ϵComp ≤ tol) && break
    end
    idx = sortperm(Λ)[2:(dim+1)]
    return Λ[idx], V[:, idx]
end


"""
    2infPert(A, V; gap=1.0)

Compute the 2→∞ residual, substituting `V` for the true eigenvector
matrix. Return both residuals as well the current Ritz values.
"""
function _2infPert(A, V; gap=1.0)
    # Rayleigh-Ritz
    _, Q, D = schur(V' * Matrix(A * V)); V[:] = V * Q
    E = Matrix(A * V) - D' .* V
    # compute residuals
    ϵ₂    = opnorm(E)
    # ϵ₂    = first(svds(E, nsv=1, ritzvec=false)[1].S)
    ϵInf  = sqrt(maximum(sum(E.^2, dims=2)))
    ϵFac  = (ϵ₂ / gap)
    vv2E  = E - V * (V'E)
    nrm0  = sqrt(maximum(sum(V.^2, dims=2)))
    nrm1  = sqrt(maximum(sum(vv2E.^2, dims=2)))
    ϵComp = 8 * nrm0 * ϵFac^2 + 2 * (nrm1 / gap) * (1 + 2 * ϵFac)
    return ϵFac, min(ϵComp, ϵ₂), D
end


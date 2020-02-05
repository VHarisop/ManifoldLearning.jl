"Generate k-nearest neighborhood graph edges with distances"
function find_nn(X::AbstractMatrix{T}, k::Int=12) where T<:Real
    m, n = size(X)
    r = Array{T}(undef, (n, n))
    d = Array{T}(undef, k, n)
    e = Array{Int}(undef, k, n)

    mul!(r, transpose(X), X)
    sa2 = sum(X.^2, dims=1)

    @inbounds for j = 1 : n
        @inbounds for i = 1 : j-1
             r[i,j] = r[j,i]
        end
        r[j,j] = 0
        @inbounds for i = j+1 : n
            v = sa2[i] + sa2[j] - 2 * r[i,j]
            r[i,j] = isnan(v) ? NaN : sqrt(max(v, 0.))
        end
        e[:, j] = sortperm(r[:,j])[2:k+1]
        d[:, j] = r[e[:, j],j]
    end

    return (d, e)
end

function adjmat(D::AbstractMatrix{T}, E::AbstractMatrix{<:Integer}) where {T<:Real}
    @assert size(D) == size(E) "Distance and edge matrix must be of the same size"
    n = size(D, 2)
    W = spzeros(T, n, n)
    @inbounds for II in CartesianIndices(D)
        d  = D[II]
        ii = E[II]
        j = II[2]
        W[ii, j] = d
        W[j, ii] = d
    end
    return W
end

"Crate a graph with largest connected component of adjacency matrix `W`"
function largest_component(G)
    CC = connected_components(G)
    if length(CC) > 1
        @warn "Found $(length(CC)) connected components. Largest component is selected."
        C = CC[argmax(map(length, CC))]
        G = first(induced_subgraph(G, C))
    else
        C = first(CC)
    end
    return G, C
end

"Generate a swiss roll dataset"
function swiss_roll(n::Int = 1000, noise::Float64=0.05)
    t = (3 * pi / 2) * (1 .+ 2 * rand(n, 1))
    height = 30 * rand(n, 1)
    X = [t .* cos.(t) height t .* sin.(t)] + noise * randn(n, 3)
    labels = vec(rem.(sum([round.(Int, t / 2) round.(Int, height / 12)], dims=2), 2))
    return collect(transpose(X)), labels
end

"Perform spectral decomposition for Ax=λI"
function decompose(M::AbstractMatrix{<:Real}, d::Int)
    W = isa(M, AbstractSparseArray) ? Symmetric(Matrix(M)) : Symmetric(M)
    F = eigen!(W)
    idx = sortperm(F.values)[2:d+1]
    return F.values[idx], F.vectors[:,idx]
end

"Perform spectral decomposition for Ax=λB"
function decompose(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real}, d::Int)
    AA = isa(A, AbstractSparseArray) ? Symmetric(full(A)) : Symmetric(A)
    BB = isa(B, AbstractSparseArray) ? Symmetric(full(B)) : Symmetric(B)
    F = eigen(AA, BB)
    idx = sortperm(F.values)[2:d+1]
    return F.values[idx], F.vectors[:,idx]
end


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


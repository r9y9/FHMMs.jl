# Completely factorized variational inference

using NumericExtensions

const ϵ = 1.0e-64

# update variational paramter `θ` based on eq. (9a), (9b)
function _updateE!(fhmm::FHMM,
                   Y::AbstractMatrix,
                   θ::Array{Float64, 3},   # shape (K, M, T)
                   C⁻¹::Matrix{Float64},    # shape (D, D)
                   logP::Array{Float64, 3}, # shape (K, K, M)
                   Δ::Matrix{Float64},     # shape (K, M)
                   Ỹ::Array{Float64, 3}     # shape (D, T)
                   )
    const D, T = size(Y)
    # alias
    W = fhmm.W

    for m=1:fhmm.M
        Δ[:,m] = diag(W[:,:,m]'*C⁻¹*W[:,:,m])
    end

    # Eq. (9b)
    ΣWθ = zeros(Float64, D)
    for m=1:fhmm.M
        for t=1:T
            fill!(ΣWθ, 0)
            for l=1:fhmm.M
                if m != l
                    ΣWθ += W[:,:,l] * θ[:,l,t]
                end
            end
            Ỹ[:,m,t] = Y[:,t] - ΣWθ
        end
    end

    # Eq. (9a)
    for m=1:fhmm.M
        WᵗC⁻¹ = W[:,:,m]'C⁻¹
        for t=2:T-1
            θ[:,m,t] = softmax(WᵗC⁻¹*Ỹ[:,m,t] - 0.5Δ[:,m]
                + logP[:,:,m]*θ[:,m,t-1]
                + logP[:,:,m]'*θ[:,m,t+1])
        end
        # special cases
        θ[:,m,1] = softmax(WᵗC⁻¹*Ỹ[:,m,1] - 0.5Δ[:,m]
                + logP[:,:,m]'*θ[:,m,2])
        θ[:,m,T] = softmax(WᵗC⁻¹*Ỹ[:,m,T] - 0.5Δ[:,m]
                + logP[:,:,m]*θ[:,m,T-1])
    end

    @assert !any(isnan(θ))

    return θ
end

# update variational parameters until convergence
function updateE!(fhmm::FHMM,
                  Y::AbstractMatrix,
                  θ::Array{Float64, 3}; # shape (K, M, T)
                  maxiter::Int=10,
                  tol::Float64=1.0e-4,
                  verbose::Bool=false)
    const D, T = size(Y)

    C⁻¹::Matrix{Float64} = fhmm.C^-1
    logP::Array{Float64, 3} = log(fhmm.P + ϵ)
    Δ = Array(Float64, fhmm.K, fhmm.M)
    Ỹ = Array(Float64, D, fhmm.M, T)

    θᵗ = copy(θ)
    for iter=1:maxiter
        # update expectations
        _updateE!(fhmm, Y, θᵗ, C⁻¹, logP, Δ, Ỹ)

        # check if converged
        diff = vecnorm(θᵗ-θ)
        if diff < tol
            if verbose
                println("$(diff): converged at #$(iter).")
            end
            break
        end

        # update θ with new one
        θ = copy(θᵗ)
    end

    return θ
end

# M-step
function updateM!(fhmm::FHMM,
                  Y::AbstractMatrix,
                  θ::Array{Float64, 3};
                  fzero::Bool=false)
    const D, T = size(Y)

    # Eq. (A.4) update initial state prob.
    fhmm.π[:,:] = θ[:,:,1]

    # Eq. (A.5) update transition matrices.
    for m=1:fhmm.M
        for i=1:fhmm.K
            for j=1:fhmm.K
                fhmm.P[i,j,m] = sum([θ[i,m,t]*θ[j,m,t-1] for t=2:T])
            end
        end
        θᵐ = reshape(θ[:,m,1:T-1], fhmm.K, T-1)
        # avoid zero division
        fhmm.P[:,:,m] = fhmm.P[:,:,m] ./ (sum(θᵐ, 2) + ϵ)'
    end

    # Eq. (A.3) update observation means.
    θʳ = reshape(θ, fhmm.K*fhmm.M, T) # slight allocations
    fhmm.W[:,:,:] = reshape((Y*θʳ') * pinv(θʳ*θʳ'), D, fhmm.K, fhmm.M)

    # sparseness contraint for minimum norm col of W
    # ** not showen in the original paper
    if fzero
        for m=1:fhmm.M
            minidx = indmin([norm(fhmm.W[:,k,m]) for k=1:fhmm.K])
            fhmm.W[:,minidx,m] .*= 0.99
        end
    end

    # Eq. (A.7) update C
    s = zeros(D, D)
    for t=1:T
        for m=1:fhmm.M
            s += fhmm.W[:,:,m]*θ[:,m,t]*Y[:,t]'
        end
    end
    fhmm.C[:,:] = 1/T * (Y*Y' - s)

    nothing
end

function randθ(K::Int, M::Int, T::Int)
    return softmax(rand(K, M, T), 1)
end

# TODO derive correct lower bound
function bound(fhmm::FHMM,
               Y::AbstractMatrix,
               θ::Array{Float64, 3})
    const D, T = size(Y)
    likelihood::Float64 = 0.0
    logθ::Array{Float64, 3} = log(θ + ϵ)
    logP::Array{Float64, 3} = log(fhmm.P + ϵ)

    for t=1:T
        for m=1:fhmm.M
            likelihood += (θ[:,m,t]'logθ[:,m,t])[1]
        end
    end

    likelihood -= T*D*log(pi)
    d = det(fhmm.C)
    if d > 0
        likelihood -= 0.5T*log(d)
    end

    C⁻¹::Matrix{Float64} = fhmm.C^-1
    for t=1:T
        ws = sum([fhmm.W[:,:,m]*θ[:,m,t] for m=1:fhmm.M])
        likelihood -= (0.5*(Y[:,t] - ws)'*C⁻¹*(Y[:,t] - ws))[1]
    end

    for m=1:fhmm.M
        likelihood += (θ[:,m,1]'log(fhmm.π[:,m] + ϵ))[1]
    end

    for t=2:T
        for m=1:fhmm.M
            likelihood += (θ[:,m,t]'logP[:,:,m]*θ[:,m,t-1])[1]
        end
    end

    @assert !isnan(likelihood)

    return likelihood
end

# Result of completely factorized variational (CFV) inference
immutable CFVInferenceResult
    likelihoods::Vector{Float64}
    θ::Array{Float64, 3}
end

function fit!(fhmm::FHMM,
              Y::AbstractMatrix; # Y: observation matrix, shape (D, T)
              maxiter::Int=100,
              tol::Float64=1.0e-5,
              fzero::Bool=false,
              verbose::Bool=false)
    const D, T = size(Y)

    # initialize variational parameter
    # means of state variables, shape (K, M, T)
    θ = randθ(fhmm.K, fhmm.M, T)

    likelihood::Vector{Float64} = zeros(1)

    # Roop of EM algorithm
    for iter=1:maxiter
        # compute bound
        score = bound(fhmm, Y, θᵗ)

        # update expectations
        θ = updateE!(fhmm, Y, θ, verbose=verbose)

        # update parameters of FHMM
        updateM!(fhmm, Y, θ, fzero=fzero)

        improvement = (score - likelihood[end]) / abs(likelihood[end])

        if verbose
            println("#$(iter): bound $(likelihood[end])
                    improvement: $(improvement)")
        end

        # check if converged
        #=
        if improvement < 1.0e-7
            if verbose
                println("#$(iter) converged")
            end
            break
        end
        =#

        push!(likelihood, score)
    end

    # remove initial zero
    shift!(likelihood)

    return CFVInferenceResult(likelihood, θ)
end

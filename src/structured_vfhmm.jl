# Structured variational inference

using NumericExtensions

# update variational parameter `h` given expectation `γ`
function _update_svp!(fhmm::FHMM,
                      Y::AbstractMatrix,
                      h::Array{Float64, 3}, # shape (K, T, M)
                      γ::Array{Float64, 3}, # shape (K, T, M) 
                      C⁻¹::Matrix{Float64},  # shape (D, D)
                      Δ::Matrix{Float64}, # shape (K, M)
                      Ỹ::Array{Float64, 3} # shape (K, M, T)
                      )
    const D, T = size(Y)
    # alias
    W = fhmm.W

    for m=1:fhmm.M
        Δ[:,m] = diag(W[:,:,m]'*C⁻¹*W[:,:,m])
    end

    # Eq. (12b)
    ΣWγ = zeros(Float64, D)
    for m=1:fhmm.M
        for t=1:T
            fill!(ΣWγ, 0.0)
            for l=1:fhmm.M
                if m != l
                    ΣWγ += W[:,:,l] * γ[:,t,l]
                end
            end
            Ỹ[:,m,t] = Y[:,t] - ΣWγ
        end
    end
    @assert !any(isnan(ΣWγ))

    # Eq. (12a)
    for m=1:fhmm.M
        WᵗC⁻¹ = W[:,:,m]'C⁻¹
        for t=1:T
            h[:,t,m] = WᵗC⁻¹*Ỹ[:,m,t] - 0.5Δ[:,m]
            # h[:,t,m] .-= maximum(h[:,t,m])
            # h[:,t,m] = softmax(h[:,t,m])
        end
    end
    h = exp(h)
    
    @assert !any(isnan(h))

    return h
end

# update expectations using forward-backward algorithm
function _updateE!(fhmm::FHMM,
                   m::Int,
                   Y::AbstractMatrix,    # shape: (D, T)
                   α::Array{Float64, 3},   # shape: (K, T, M)
                   β::Array{Float64, 3},   # shape: (K, T, M)
                   γ::Array{Float64, 3},   # shape: (K, T, M)
                   ξ::Array{Float64, 4}, # shape: (K, K, T-1, M)
                   B::Array{Float64, 3})   # shape: (K, T, M)
    const D, T = size(Y)
    const K = fhmm.K
    
    # scaling paramter
    c = Array(Float64, T)

    # forward recursion
    α[:,1,m] = fhmm.π[:,m] .* B[:,1,m]
    α[:,1,m] /= (c[1] = sum(α[:,1,m]) + ϵ)
    for t=2:T
        @inbounds α[:,t,m] = (fhmm.P[:,:,m]' * α[:,t-1,m]) .* B[:,t,m]
        @inbounds α[:,t,m] /= (c[t] = sum(α[:,t,m]) + ϵ)
    end
    @assert !any(isnan(α[:,:,m]))

    # backword recursion
    β[:,T,m] = 1.0
    for t=T-1:-1:1
        β[:,t,m] = fhmm.P[:,:,m] * β[:,t+1,m] .* B[:,t+1,m] ./ c[t+1]
    end
    @assert !any(isnan(β[:,:,m]))

    γ[:,:,m] = α[:,:,m] .* β[:,:,m]

    for t=1:T-1
        ξ[:,:,t,m] = fhmm.P[:,:,m] .* α[:,t,m] .* β[:,t+1,m]' .* 
            B[:,t+1,m]' ./ c[t+1]
    end

    return γ, ξ
end

# update variational paramter until convergence and then update expectations.
function updateE!(fhmm::FHMM,
                  Y::AbstractMatrix,
                  α::Array{Float64, 3}, # inplace
                  β::Array{Float64, 3}, # 
                  γ::Array{Float64, 3}, # shape (K, T, M)
                  ξ::Array{Float64, 4}, # inplace
                  h::Array{Float64, 3};  # shape (K, T, M)
                  maxiter::Int=10,
                  tol::Float64=1.0e-4,
                  verbose::Bool=false)
    const D, T = size(Y)
    const K = fhmm.K
    const M = fhmm.M

    C⁻¹::Matrix{Float64} = fhmm.C^-1
    Δ = Array(Float64, K, M)
    Ỹ = Array(Float64, D, M, T)

    hᵗ = copy(h)
    for iter=1:maxiter
        # update variational parameters
        hᵗ = _update_svp!(fhmm, Y, hᵗ, γ, C⁻¹, Δ, Ỹ)
        
        # check if converged
        diff = vecnorm(hᵗ-h)
        if diff < tol
            if verbose
                println("$(diff): converged at #$(iter).")
            end
            break
        end

        # update h with new one
        h = copy(hᵗ)
    end
    
    # update expectations using forward-backward algorithm for 
    # each factorized HMM
    for m=1:M
        γ, ξ = _updateE!(fhmm, m, Y, α, β, γ, ξ, h)
    end

    return γ, ξ, h
end

# M-step
function updateM!(fhmm::FHMM,
                  Y::AbstractMatrix,
                  γ::Array{Float64, 3}, # shape (K, T, M)
                  ξ::Array{Float64, 4}; # shape (K, K, T-1, M)
                  fzero::Bool=false)
    const D, T = size(Y)
    const M = fhmm.M
    const K = fhmm.K

    # Eq. (A.4) update initial state prob.
    for m=1:M
        fhmm.π[:,m] = γ[:,1,m] / sum(γ[:,1,m] + ϵ)
    end
    
    # Eq. (A.5) update transition matrices
    for m=1:fhmm.M
        fhmm.P[:,:,m] = sum(ξ[:,:,:,m], 3) ./ (sum(γ[:,1:end-1,m], 2) + ϵ)
        # fhmm.P[:,:,m] ./= sum(fhmm.P[:,:,m], 1)
    end

    # Eq. (A.3) update observation means
    γʳ = Array(Float64, K*M, T)
    for t=1:T
       γʳ[:,t] = reshape(γ[:,t,:], K*M)
    end
    fhmm.W[:,:,:] = reshape((Y*γʳ') * pinv(γʳ*γʳ'), D, K, M)

    # sparseness contraint for minimum norm col of W
    if fzero
        for m=1:fhmm.M
            minidx = indmin([norm(fhmm.W[:,k,m]) for k=1:fhmm.K])
            fhmm.W[:,minidx,m] .*= 0.99
        end
    end

    # Eq. (A,7) update C
    s = zeros(D, D)
    for t=1:T
        for m=1:fhmm.M
            s += fhmm.W[:,:,m]*γ[:,t,m]*Y[:,t]'
        end
    end    
    fhmm.C[:,:] = 1/T * (Y*Y' - s)

    nothing
end

# TODO derive correct lower bound
function bound_sv(fhmm::FHMM,
                  Y::AbstractMatrix,
                  γ::Array{Float64, 3},
                  ξ::Array{Float64, 4})
    const D, T = size(Y)
    likelihood::Float64 = 0.0
    logγ::Array{Float64, 3} = log(γ + ϵ)
    logP::Array{Float64, 3} = log(fhmm.P + ϵ)
    
    for t=1:T
        for m=1:fhmm.M
            likelihood += (γ[:,t,m]'logγ[:,t,m])[1]
        end
    end

    likelihood -= T*D*log(pi)
    d = det(fhmm.C)
    if d > 0
        likelihood -= 0.5T*log(d)
    end

    C⁻¹::Matrix{Float64} = fhmm.C^-1
    for t=1:T
        ws = sum([fhmm.W[:,:,m]*γ[:,t,m] for m=1:fhmm.M])
        likelihood -= (0.5*(Y[:,t] - ws)'*C⁻¹*(Y[:,t] - ws))[1]
    end

    for m=1:fhmm.M
        likelihood += (γ[:,1,m]'log(fhmm.π[:,m] + ϵ))[1]
    end

    #=
    for t=2:T
        for m=1:fhmm.M
            likelihood += (θ[:,m,t]'logP[:,:,m]*θ[:,m,t-1])[1]
        end
    end
    =#

    @assert !isnan(likelihood)

    return likelihood
end

# Result of structured variational (SV) inference
immutable SVInferenceResult
    likelihoods::Vector{Float64}
    h::Array{Float64, 3}
    α::Array{Float64, 3}
    β::Array{Float64, 3}
    γ::Array{Float64, 3}
    ξ::Array{Float64, 4}
end

function fit_sv!(fhmm::FHMM,
                 Y::AbstractMatrix; # observation matrix, shape (D, T)
                 maxiter::Int=100,
                 tol::Float64=1.0e-5,
                 fzero::Bool=false,
                 verbose::Bool=false)
    const D, T = size(Y)
    const M = fhmm.M
    const K = fhmm.K
        
    # means of observation probability
    α = Array(Float64, K, T, M)
    β = Array(Float64, K, T, M)
    γ = ones(Float64, K, T, M)/K
    ξ = ones(Float64, K, K, T-1, M)/K
    h = ones(Float64, K, T, M)/K

    likelihood::Vector{Float64} = zeros(1)

    # Roop of EM algorithm
    for iter=1:maxiter
        # compute bound
        score = bound_sv(fhmm, Y, γ, ξ)
        
        # update expectations
        γ, ξ, h = updateE!(fhmm, Y, α, β, γ, ξ, h, verbose=verbose)

        # update parameters of FHMM
        updateM!(fhmm, Y, γ, ξ, fzero=fzero)

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

    return SVInferenceResult(likelihood, h, α, β, γ, ξ)
end

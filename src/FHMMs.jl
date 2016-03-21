module FHMMs

# Factorial Hidden Markov Models (FHMM)

# Reference:
# [Ghahramani1997] Ghahramani, Zoubin, and Michael I. Jordan.
# "Factorial hidden Markov models."
# Machine learning 29.2-3 (1997): 245-273.

export FHMM, fit!, fit_sv!

# Factorial HMM (FHMM)
type FHMM
    K::Int # number of states
    M::Int # number of factors
    # contributions to the means of Gaussian, shape (D, K, M)
    W::Array{Float64, 3}
    # initial state probability, shape (K, M)
    π::Array{Float64, 2}
    # transition maticies, shape (K, K, M)
    P::Array{Float64, 3}
    # covariance matrix, shape (D, D)
    C::Array{Float64, 2}

    function FHMM(D::Int, K::Int, M::Int)
        new(K, M,
            rand(D, K, M),
            ones(K, M) ./ (K*M),
            ones(K, K, M) ./ K,
            rand(D, D))
    end
end

for fname in ["completely_factorized_vfhmm",
              "structured_vfhmm"]
    include(string(fname, ".jl"))
end

end # module

# TensorOps/measurements.jl

"""
    measure_energy(state::MPSState)

Compute ⟨ψ|H|ψ⟩ for an MPSState by contracting the full MPS-MPO-MPS sandwich.

The MPS should be in canonical form. The energy is computed by sweeping
left-to-right, building the left environment, and reading off the scalar
at the end.
"""
function measure_energy(state::MPSState)
    mps = state.mps
    mpo = state.mpo
    N = length(mps.tensors)
    Tenv = promote_type(eltype(mps.tensors[1]), eltype(mpo.tensors[1]))

    # Start with trivial left boundary
    L = ones(Tenv, 1, 1, 1)

    # Contract site by site from left to right
    for i in 1:N
        L = _contract_left_environment(L, mps.tensors[i], mpo.tensors[i])
    end

    # L is now (1,1,1) — the scalar ⟨ψ|H|ψ⟩
    energy = real(L[1, 1, 1])
    return energy
end

"""
    measure_norm(state::MPSState)

Compute ⟨ψ|ψ⟩ by contracting MPS with itself (no MPO).
"""
function measure_norm(mps::MPS)
    N = length(mps.tensors)
    T = eltype(mps.tensors[1])

    # Transfer matrix contraction: start with (1,1) identity
    C = ones(T, 1, 1)

    for i in 1:N
        A = mps.tensors[i]
        # C[a,b] * conj(A)[a,s,c] * A[b,s,d] → C_new[c,d]
        @tensoropt C_new[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * A[4, 5, -2]
        C = C_new
    end

    return real(C[1, 1])
end

"""
    measure_local_observable(mps::MPS, op::Matrix, site_idx::Int)

Compute ⟨ψ|O_site|ψ⟩ for operator `op` acting on site `site_idx`.
The MPS should be normalized.
"""
function measure_local_observable(mps::MPS, op::Matrix, site_idx::Int)
    N = length(mps.tensors)
    @assert 1 ≤ site_idx ≤ N "site_idx must be in 1:$N"
    T = promote_type(eltype(mps.tensors[1]), eltype(op))

    # Build left environment up to site_idx - 1 (just overlap, no MPO)
    C = ones(T, 1, 1)
    for i in 1:site_idx-1
        A = mps.tensors[i]
        @tensoropt C_new[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * A[4, 5, -2]
        C = C_new
    end

    # Insert operator at site_idx
    A = mps.tensors[site_idx]
    op_T = convert(Matrix{T}, op)
    @tensoropt C_op[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * op_T[5, 6] * A[4, 6, -2]
    C = C_op

    # Continue contracting to the right (overlap only)
    for i in site_idx+1:N
        A = mps.tensors[i]
        @tensoropt C_new[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * A[4, 5, -2]
        C = C_new
    end

    return C[1, 1]
end

"""
    measure_correlation(mps::MPS, op_L::Matrix, site_L::Int, op_R::Matrix, site_R::Int)

Compute ⟨ψ|O_L(site_L) O_R(site_R)|ψ⟩ for sites site_L < site_R.
"""
function measure_correlation(mps::MPS, op_L::Matrix, site_L::Int, op_R::Matrix, site_R::Int)
    N = length(mps.tensors)
    @assert 1 ≤ site_L < site_R ≤ N "Need site_L < site_R in 1:$N"
    T = promote_type(eltype(mps.tensors[1]), eltype(op_L), eltype(op_R))

    # Build left environment up to site_L - 1
    C = ones(T, 1, 1)
    for i in 1:site_L-1
        A = mps.tensors[i]
        @tensoropt C_new[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * A[4, 5, -2]
        C = C_new
    end

    # Insert op_L at site_L
    A = mps.tensors[site_L]
    op_L_T = convert(Matrix{T}, op_L)
    @tensoropt C_op[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * op_L_T[5, 6] * A[4, 6, -2]
    C = C_op

    # Propagate (overlap) from site_L+1 to site_R-1
    for i in site_L+1:site_R-1
        A = mps.tensors[i]
        @tensoropt C_new[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * A[4, 5, -2]
        C = C_new
    end

    # Insert op_R at site_R
    A = mps.tensors[site_R]
    op_R_T = convert(Matrix{T}, op_R)
    @tensoropt C_op2[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * op_R_T[5, 6] * A[4, 6, -2]
    C = C_op2

    # Continue to the end
    for i in site_R+1:N
        A = mps.tensors[i]
        @tensoropt C_new[-1, -2] := C[3, 4] * conj(A)[3, 5, -1] * A[4, 5, -2]
        C = C_new
    end

    return C[1, 1]
end

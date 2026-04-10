include(joinpath(@__DIR__, "..", "src", "DMRGOpt.jl"))
using .DMRGOpt

# --- Parameters ---
N       = 20       # number of sites
d       = 2        # spin-1/2
chi_max = 32       # max bond dimension
Jxy     = 1.0
Jz      = 1.0      # isotropic Heisenberg
n_sweeps = 10
cutoff   = 1e-10    # SVD truncation cutoff

# --- Build model ---
mpo = build_xxz_mpo(N, Jxy, Jz; d=d)
mps = random_mps(N, d, chi_max)
state = MPSState(mps, mpo)

# --- Solver & schedule ---
krylov_dim = 4
max_iter   = 100
solver     = LanczosSolver(krylov_dim, max_iter)
schedule = SweepSchedule(chi_max, n_sweeps; cutoff_final=cutoff)

# --- DMRG loop ---
for sweep in 1:n_sweeps
    opts = DMRGOptions(schedule.maxdims[sweep], schedule.cutoffs[sweep], d)
    e_R  = dmrg_sweep(state, solver, opts, :right)
    e_L  = dmrg_sweep(state, solver, opts, :left)
    E    = measure_energy(state)
    println("Sweep $sweep  χ=$(schedule.maxdims[sweep])  E = $E")
end

println("\nFinal energy: ", measure_energy(state))
println("Final norm:   ", measure_norm(state.mps))

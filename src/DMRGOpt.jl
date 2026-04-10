module DMRGOpt

using LinearAlgebra
using TensorOperations

export
    # Core types
    TensorNetwork,
    MPS,
    MPO,
    MPDO,
    Environment,
    MPSState,
    DMRGOptions,
    TDVPOptions,
    SweepSchedule,

    # Sites
    SpinSite,
    SuperSite,

    # Measurements
    measure_energy,
    measure_norm,
    measure_local_observable,
    measure_correlation,

    # Canonicalization
    make_canonical,
    is_left_orthogonal,
    is_right_orthogonal,
    is_orthogonal,

    # Decomposition utilities
    entropy,
    truncation_error,

    # Solvers
    EffectiveHamiltonian,
    OneSiteEffectiveHamiltonian,
    TwoSiteEffectiveHamiltonian,
    ZeroSiteEffectiveHamiltonian,
    LanczosSolver,
    KrylovExponential,

    # DMRG
    dmrg_sweep,

    # Models
    build_xxz_mpo,
    update_fields!,
    random_mps

# Include files in dependency order
include("Core/types.jl")
include("Core/site.jl")
include("TensorOps/canonicalization.jl")
include("TensorOps/decomposition.jl")
include("TensorOps/environment.jl")
include("TensorOps/measurements.jl")
include("Core/states.jl")
include("Models/random_mps.jl")
include("Models/xxz.jl")
include("Algorithms/solvers.jl")
include("Algorithms/dmrg.jl")

end # module DMRGOpt

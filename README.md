# Constrained DMRG

Ground-state optimization of quantum spin Hamiltonians with site-resolved magnetization constraints, implemented from scratch in Julia.

## Overview

Standard DMRG finds the ground state of a Hamiltonian $\hat{H}$. This project extends it to solve the constrained problem:

$$\min_{|\psi\rangle} \langle \psi | \hat{H} | \psi \rangle \quad \text{s.t.} \quad \langle \psi | \hat{S}^z_i | \psi \rangle = m_i, \quad i = 1, \dots, N$$

The constraints are enforced via Lagrange multipliers $\lambda_i$ that act as site-local magnetic fields in an effective Hamiltonian $\hat{H}_{\text{eff}} = \hat{H} - \sum_i \lambda_i \hat{S}^z_i$. The multiplier fields are injected directly into the MPO on-site blocks with no increase in bond dimension.

## Dependencies

- [Julia](https://julialang.org/) (tested on 1.10+)
- [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) &mdash; tensor contraction via `@tensoropt`
- LinearAlgebra (stdlib)

## Project Structure

```
Constrained_DMRG/
├── src/
│   ├── DMRGOpt.jl                  # Module definition and exports
│   ├── Core/
│   │   ├── types.jl                # MPS, MPO, MPDO, Environment, MPSState,
│   │   │                           #   DMRGOptions, TDVPOptions, SweepSchedule
│   │   ├── site.jl                 # SpinSite & SuperSite types, spin_ops(d)
│   │   └── states.jl               # MPSState constructor (canonicalize + build env)
│   ├── TensorOps/
│   │   ├── canonicalization.jl     # Mixed canonical form, orthogonality checks
│   │   ├── decomposition.jl        # SVD with truncation, entropy, truncation error
│   │   ├── environment.jl          # Left/right environment construction and updates
│   │   └── measurements.jl         # Energy, norm, local observables, correlators
│   ├── Models/
│   │   ├── xxz.jl                  # XXZ Heisenberg MPO with injectable Zeeman fields
│   │   └── random_mps.jl           # Random MPS initialization
│   └── Algorithms/
│       ├── solvers.jl              # Lanczos eigensolver, Krylov exponential (TDVP)
│       └── dmrg.jl                 # Two-site DMRG sweep
├── scripts/
│   └── run_dmrg.jl                 # Entry point: unconstrained DMRG on Heisenberg chain
└── docs/
    └── dmrg_local_magnetization_constraints.[md|pdf]   # Working notes on constraint strategies
```

### Core

- **`types.jl`** &mdash; All fundamental types: `MPS`, `MPO`, `MPDO` (subtypes of `TensorNetwork`), `Environment`, `MPSState` (bundles MPS + MPO + environments + orthogonality center), and option/schedule structs. `SweepSchedule` ramps bond dimension linearly over the first half of sweeps and tightens the SVD cutoff.
- **`site.jl`** &mdash; `SpinSite(S)` builds spin-$S$ operators ($S^x, S^y, S^z, S^+, S^-$) for arbitrary spin. `SuperSite(n)` bundles $n$ spin-1/2 sites into a single supersite of dimension $2^n$ with sublattice operators built via Kronecker products.
- **`states.jl`** &mdash; `MPSState(mps, mpo)` canonicalizes the MPS in-place and builds the matching environment, returning a ready-to-sweep state object.

### TensorOps

- **`canonicalization.jl`** &mdash; SVD-based gauge moves (`_move_orthogonality_left/right`) and `make_canonical(mps, center)` for mixed canonical form. Includes orthogonality verification utilities.
- **`decomposition.jl`** &mdash; `_svd_truncate` with both bond-dimension cap and relative singular-value cutoff. Entanglement entropy and truncation error from singular values.
- **`environment.jl`** &mdash; Left and right environment tensor contractions, full environment construction around the orthogonality center, and incremental updates during sweeps.
- **`measurements.jl`** &mdash; `measure_energy` (full MPS-MPO-MPS contraction), `measure_norm`, `measure_local_observable` (single-site $\langle O_i \rangle$), and `measure_correlation` (two-point $\langle O_i O_j \rangle$).

### Models

- **`xxz.jl`** &mdash; Builds the bond-dimension-5 MPO for the XXZ chain: $H = \sum_i \left[\frac{J_{xy}}{2}(S^+_i S^-_{i+1} + \text{h.c.}) + J_z S^z_i S^z_{i+1}\right] + \sum_i h_i S^z_i$. The field vector `h` is where Lagrange multipliers enter ($h_i = -\lambda_i$). `update_fields!(mpo, h)` patches the on-site MPO blocks in-place without rebuilding.
- **`random_mps.jl`** &mdash; Normalized random MPS with bond dimensions that respect exact Hilbert-space growth from both edges, capped at the target $\chi$.

### Algorithms

- **`solvers.jl`** &mdash; `LanczosSolver` for ground-state eigenvalue problems (DMRG) with thick-restart Lanczos. `KrylovExponential` for real/imaginary time evolution (TDVP) via Krylov subspace exponentiation. Effective Hamiltonian types (`OneSite`, `TwoSite`, `ZeroSite`) with `_apply` methods for matrix-free Hamiltonian-vector products.
- **`dmrg.jl`** &mdash; Two-site DMRG sweep (left-to-right and right-to-left) with SVD truncation and incremental environment updates.

## Usage

```bash
cd Constrained_DMRG
julia scripts/run_dmrg.jl
```

This runs 10 DMRG sweeps on a 20-site spin-1/2 isotropic Heisenberg chain ($J_{xy} = J_z = 1$) with bond dimension ramping up to $\chi = 32$.

## Future Plans

### Constrained optimization loop

The core infrastructure for constrained DMRG is in place (field injection via `update_fields!`, local observable measurements, correlators for susceptibility). What remains is wiring up the outer optimization loop over the Lagrange multipliers. Five strategies are documented in detail in [`docs/dmrg_local_magnetization_constraints.md`](docs/dmrg_local_magnetization_constraints.md):

1. **Two-loop (DMRG-as-black-box)** &mdash; Fully converge DMRG at each $\boldsymbol{\lambda}$, then update multipliers. Clean but wasteful; useful as a reference.
2. **Uzawa (per-sweep updates)** &mdash; Update $\lambda_i$ after each DMRG sweep. Default first attempt.
3. **Per-site primal-dual** &mdash; Update $\lambda_i$ during the sweep at each site visit. Fastest convergence per sweep, hardest to debug.
4. **Augmented Lagrangian** &mdash; Add a linearized quadratic penalty to damp oscillations. Workhorse for hard cases.
5. **Newton / quasi-Newton** &mdash; Use the static susceptibility $\chi_{ij} = \langle S^z_i S^z_j \rangle_c$ for second-order multiplier updates. Final polishing step.

### Additional planned features

- **TDVP time evolution** &mdash; The Krylov exponential solver and zero-site effective Hamiltonian are implemented; the TDVP sweep loop (1-site and 2-site) needs to be added.
- **Additional Hamiltonians** &mdash; Transverse-field Ising, Hubbard, and generic nearest-neighbor models.
- **Entanglement diagnostics** &mdash; Per-bond entanglement entropy tracking during sweeps.
- **Convergence monitoring** &mdash; Automated tracking of energy, constraint residual $\|\mathbf{g}\|$, and truncation error across sweeps.

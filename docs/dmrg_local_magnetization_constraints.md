# DMRG with Local Magnetization Constraints

A working note on enforcing site-resolved magnetization targets in DMRG via Lagrange multipliers. The note lays out the problem, the basic effective-Hamiltonian construction, and then a sequence of progressively more flexible strategies for coupling the multiplier updates to the DMRG optimization. No single recipe is declared "the answer" — each has tradeoffs.

---

## 1. Problem Definition

Given a quantum spin Hamiltonian $\hat{H}$ on $N$ sites, find the lowest-energy state $|\psi\rangle$ subject to a prescribed local magnetization profile:

$$\min_{|\psi\rangle} \langle \psi | \hat{H} | \psi \rangle \quad \text{s.t.} \quad \langle \psi | \hat{S}^z_i | \psi \rangle = m_i, \quad i = 1, \dots, N, \quad \langle \psi | \psi \rangle = 1$$

This is $N$ simultaneous equality constraints — one per site — and is strictly stronger than fixing only the total $S^z_{\text{tot}} = \sum_i m_i$. Unlike the global case (which is a $U(1)$ symmetry sector and can be enforced exactly in DMRG block structure), local magnetization is **not** a symmetry of generic $\hat{H}$, so it must be imposed as a soft optimization constraint.

## 2. Lagrangian and Effective Hamiltonian

Introduce one Lagrange multiplier $\lambda_i$ per site:

$$\mathcal{L}[\psi, \boldsymbol{\lambda}] = \langle \psi | \hat{H} | \psi \rangle - \sum_{i=1}^{N} \lambda_i \left( \langle \psi | \hat{S}^z_i | \psi \rangle - m_i \right)$$

Stationarity in $\langle \psi |$ yields the eigenvalue problem for an **inhomogeneously Zeeman-shifted** Hamiltonian:

$$\hat{H}_{\text{eff}}(\boldsymbol{\lambda}) = \hat{H} - \sum_{i=1}^{N} \lambda_i \hat{S}^z_i$$

Each $\lambda_i$ acts as a local magnetic field at site $i$. Because each term is a single-site operator, the modification to the MPO touches only the on-site block — **no increase in bond dimension**. The dual function $\mathcal{D}(\boldsymbol{\lambda}) = \min_\psi \mathcal{L}[\psi, \boldsymbol{\lambda}]$ is concave in $\boldsymbol{\lambda}$, so the outer problem is a well-posed concave maximization (saddle point of $\mathcal{L}$).

The constraint residual and its sensitivity (the static susceptibility):

$$g_i(\boldsymbol{\lambda}) \equiv \langle \hat{S}^z_i \rangle_{\psi_{\boldsymbol{\lambda}}} - m_i, \qquad \chi_{ij} = \frac{\partial \langle \hat{S}^z_i \rangle}{\partial \lambda_j} = \langle \hat{S}^z_i \hat{S}^z_j \rangle_c$$

True energy is recovered as $E = \langle \hat{H} \rangle = E_{\text{eff}} + \sum_i \lambda_i m_i$ at the converged $\boldsymbol{\lambda}^*$.

---

## 3. Strategies

The strategies below are ordered from the most rigid (clean two-loop separation) to the most tightly interleaved (multiplier updates inside the local site optimization). Roughly: rigidity is easy to implement and debug but slow; interleaving is fast but harder to control.

### Strategy 1 — Brute force two-loop (DMRG-as-black-box)

The simplest possible thing.

**Procedure.**
1. Initialize $\boldsymbol{\lambda}^{(0)} = 0$.
2. **Inner loop:** run DMRG to full convergence on $\hat{H}_{\text{eff}}(\boldsymbol{\lambda}^{(k)})$, returning $|\psi^{(k)}\rangle$.
3. Measure $g_i^{(k)} = \langle \hat{S}^z_i \rangle_{\psi^{(k)}} - m_i$.
4. **Outer update:**
   - *Gradient ascent on the dual:* $\lambda_i^{(k+1)} = \lambda_i^{(k)} + \eta\, g_i^{(k)}$
   - *Newton (using susceptibility):* $\boldsymbol{\lambda}^{(k+1)} = \boldsymbol{\lambda}^{(k)} - \chi^{-1} \mathbf{g}^{(k)}$, with $\chi_{ij} = \langle \hat{S}^z_i \hat{S}^z_j \rangle_c$ measured from $|\psi^{(k)}\rangle$
5. Repeat until $\|\mathbf{g}\| < \epsilon$.

**Pros.**
- Conceptually clean; DMRG remains an unmodified black box.
- Easy to debug — at every outer step you have a true ground state of a well-defined Hamiltonian.
- Newton variant has fast (quadratic) local convergence if $\chi$ is well-conditioned.

**Cons.**
- **Extremely wasteful.** Each inner DMRG fully converges to the ground state of a Hamiltonian whose $\boldsymbol{\lambda}$ is about to change, so most of the inner work is thrown away.
- Near the saddle point the inner problem is ill-conditioned: small $\Delta\boldsymbol{\lambda}$ can move the MPS substantially, so each fresh DMRG re-does work the previous one already did.
- **In practice it often fails to converge** — the outer gradient steps overshoot, the inner DMRG dutifully chases, and the residual oscillates without decreasing.
- Newton variant requires $O(N^2)$ correlator measurements per outer step, and $\chi$ may be near-singular (e.g. soft modes, gapless systems), forcing regularization.

**When to use.** Pedagogical baseline, or when $N$ is small enough that the waste is tolerable, or when you want to *check* a more aggressive method against a clean reference.

---

### Strategy 2 — Update $\boldsymbol{\lambda}$ after each DMRG sweep (Uzawa)

Don't wait for DMRG to converge. After every full sweep (or every left-to-right half-sweep), do one multiplier step and then continue sweeping with the updated MPO.

**Procedure.**
1. Initialize $\boldsymbol{\lambda} = 0$ and a random/product MPS.
2. Perform one DMRG sweep on the current $\hat{H}_{\text{eff}}(\boldsymbol{\lambda})$.
3. Measure $\langle \hat{S}^z_i \rangle$ from the resulting MPS.
4. Update $\lambda_i \leftarrow \lambda_i + \eta_k (\langle \hat{S}^z_i \rangle - m_i)$ and refresh the on-site MPO terms.
5. Repeat from step 2 until both energy and residual are converged.

This is **Uzawa's algorithm** (alternating primal/dual gradient steps) applied to MPS optimization. Neither $|\psi\rangle$ nor $\boldsymbol{\lambda}$ is ever fully optimized in isolation — they evolve together toward the saddle point of $\mathcal{L}$.

**Pros.**
- Trivial modification of any DMRG code: insert one measurement + multiplier update between sweeps.
- No wasted inner sweeps. Total cost is comparable to a single unconstrained DMRG run, *not* (number of outer iterations) × (full DMRG).
- Convergence theory is standard (Uzawa, Arrow–Hurwicz): for convex-concave saddle problems with appropriate step size, it converges.

**Cons.**
- Step size $\eta_k$ must be chosen carefully. Constant $\eta$ may oscillate; a decreasing schedule like $\eta_k \sim \eta_0/\sqrt{k}$ is safer but slower.
- During early sweeps the MPS is far from any ground state, so the measured $\langle \hat{S}^z_i \rangle$ is noisy and the multiplier updates can wander before settling.
- No quadratic convergence — it's first-order in $\boldsymbol{\lambda}$.

**When to use.** Default first attempt. If it works, you're done; if it stalls or oscillates, escalate to Strategy 4 (augmented Lagrangian).

---

### Strategy 3 — Update $\lambda_i$ inside the local site optimization

Even more aggressive: update each $\lambda_i$ during the DMRG site sweep itself, at the moment DMRG visits site $i$.

**Procedure (one site visit during a sweep).**
1. At site $i$: build the local effective Hamiltonian $\hat{H}_{\text{eff}}^{\text{loc}}$ from environment tensors, including the current $\lambda_i \hat{S}^z_i$ on-site term.
2. Solve the local eigenvalue problem for the new tensor $A_i$.
3. Compute $\langle \hat{S}^z_i \rangle$ from $A_i$ (essentially free given the local tensor).
4. Update $\lambda_i \leftarrow \lambda_i + \eta(\langle \hat{S}^z_i \rangle - m_i)$.
5. *(Optional)* Re-solve the local problem with the updated $\lambda_i$, one or two micro-iterations.
6. Move to site $i+1$ and repeat.

Conceptually this is a **primal–dual sweep**: each site visit is one primal step (local eigensolve on the MPS tensor) plus one dual step (coordinate update of $\lambda_i$). Since $\lambda_i$ couples only to the on-site operator at site $i$, the multiplier update is local and cheap.

**Pros.**
- Finest possible coupling between multiplier and wavefunction updates — fastest convergence in number of sweeps when it works.
- No extra global measurements: $\langle \hat{S}^z_i \rangle$ falls out of the local tensor for free.
- Natural lockstep coordinate descent: DMRG already does coordinate descent on MPS tensors, this just adds a parallel coordinate descent on $\boldsymbol{\lambda}$.

**Cons.**
- After leaving site $i$, neighboring updates re-entangle and $\langle \hat{S}^z_i \rangle$ drifts away from $m_i$. The constraint is only satisfied "in passing." Subsequent sweeps must revisit and correct, which they do, but it can look like the residual is non-monotone within a sweep.
- Tightly couples to DMRG truncation: a poorly converged local eigensolve produces a noisy $\langle \hat{S}^z_i \rangle$, which produces a noisy $\lambda_i$ update, which biases the next site's local Hamiltonian. Bugs are subtle.
- Per-sweep behavior depends on sweep direction (left-to-right vs right-to-left), breaking the symmetry of the problem in a way that can introduce small biases.

**When to use.** When Strategy 2 is too slow (e.g. very large $N$, or you can't afford many sweeps because bond dimension is huge), and when you trust your DMRG implementation enough to debug subtle interactions.

---

### Strategy 4 — Augmented Lagrangian (linearized)

When Strategies 2 or 3 oscillate or stall, add a quadratic penalty on top of the Lagrangian:

$$\mathcal{L}_\rho[\psi, \boldsymbol{\lambda}] = \langle \hat{H} \rangle - \sum_i \lambda_i (\langle \hat{S}^z_i \rangle - m_i) + \frac{\rho}{2} \sum_i (\langle \hat{S}^z_i \rangle - m_i)^2$$

The penalty term is **not** a single-site operator on the wavefunction — it's quadratic in expectation values, so it does not fit directly into an MPO. The standard fix is to linearize around the current expectation value: at iteration $k$, replace

$$\frac{\rho}{2}(\langle \hat{S}^z_i \rangle - m_i)^2 \;\longrightarrow\; \rho (\langle \hat{S}^z_i \rangle^{(k)} - m_i) \, \hat{S}^z_i + \text{const}$$

so the effective field becomes

$$\tilde{\lambda}_i^{(k)} = \lambda_i^{(k)} - \rho \, (\langle \hat{S}^z_i \rangle^{(k)} - m_i)$$

and the multiplier update follows the standard ALM rule:

$$\lambda_i^{(k+1)} = \lambda_i^{(k)} - \rho \, (\langle \hat{S}^z_i \rangle^{(k)} - m_i)$$

In practice this looks identical to Strategy 2 but with an effective step size $\eta + \rho$, and with the theoretical convergence guarantees of the augmented Lagrangian method.

**Pros.**
- **Most robust** of the lot. ALM converges even for non-strictly-convex problems and is forgiving about $\rho$.
- Quadratic penalty damps oscillations that plague pure gradient methods.
- Reduces sensitivity to the initial $\boldsymbol{\lambda}$ guess.
- Fits into Strategy 2's sweep cadence with no structural change to the algorithm — just a different update rule.

**Cons.**
- One more hyperparameter ($\rho$). Rule of thumb: $\rho$ comparable to the local gap or the typical exchange scale.
- Linearization is only first-order accurate in the residual; very far from feasibility, the quadratic penalty's true effect is underestimated.
- Same caveat as Strategy 2 about noisy early-sweep measurements.

**When to use.** As soon as Strategy 2 shows oscillation or stall. In practice this is the workhorse for hard cases.

---

### Strategy 5 — Newton with susceptibility (and quasi-Newton variants)

If you're willing to invest in measuring the susceptibility matrix $\chi_{ij} = \langle \hat{S}^z_i \hat{S}^z_j \rangle_c$, you can take Newton steps in $\boldsymbol{\lambda}$:

$$\boldsymbol{\lambda}^{(k+1)} = \boldsymbol{\lambda}^{(k)} - \chi^{-1} \mathbf{g}^{(k)}$$

This can be combined with any of Strategies 1–3 (full Newton in the outer loop, or Newton after every sweep instead of gradient).

**Pros.**
- Quadratic local convergence near the saddle point.
- $\chi$ has direct physical meaning (static spin susceptibility) and can be useful as a diagnostic.
- Quasi-Newton variants (BFGS on the dual) avoid explicit $\chi$ measurement and accumulate curvature info from successive gradients.

**Cons.**
- Measuring $\chi$ requires $O(N^2)$ two-point correlators per Newton step — expensive at large $N$.
- $\chi$ may be ill-conditioned or near-singular near gapless points or magnetization plateaus, requiring regularization (e.g. $\chi + \mu I$, Levenberg–Marquardt-style).
- Far from the saddle, Newton steps can overshoot wildly — needs trust-region or line search safeguards.

**When to use.** Final polishing once Strategy 2 or 4 has gotten close to feasibility, to crank the residual down quickly. Or when you specifically want $\chi$ for physical reasons anyway.

---

## 4. Practical Recommendations

- **Always warm-start.** When $\boldsymbol{\lambda}$ updates, start the next DMRG (or sweep) from the *current* MPS, never from scratch. The MPS for nearby $\boldsymbol{\lambda}$ values is nearly identical.
- **Monitor two things, not one.** Track both the energy $\langle \hat{H} \rangle$ and the residual $\|\mathbf{g}\|$. Healthy convergence shows both decreasing. If energy drops but residual grows, $\eta$ is too small or you're in a wrong-sector basin. If residual drops but energy oscillates, $\eta$ is too large.
- **Suggested progression.** Start with **Strategy 2**, modest $\eta \sim 0.1$–$0.5$ in units of the typical exchange. If it stalls or oscillates, switch to **Strategy 4** with $\rho$ comparable to the local gap. Use **Strategy 3** only if entanglement-limited and full sweeps are too expensive. Use **Strategy 5** as a final polishing step or when $\chi$ is wanted anyway.
- **Reserve Strategy 1 as a sanity check** — when in doubt about whether an interleaved method is finding the right saddle, run a few brute outer iterations and confirm they agree.

## 5. Feasibility / v-Representability

Not every target profile $\{m_i\}$ is realizable as the constrained ground state of *any* $\hat{H} - \sum_i \lambda_i \hat{S}^z_i$. The set of achievable profiles is the image of the map $\boldsymbol{\lambda} \mapsto \langle \hat{\mathbf{S}}^z \rangle_{\psi_{\boldsymbol{\lambda}}}$, which is the lattice analog of the v-representability question in DFT (and this whole construction is essentially Levy constrained search for $S^z$). Symptoms of infeasibility:

- $\boldsymbol{\lambda}$ runs away to large values without the residual shrinking.
- Residual stalls at a finite value no matter the strategy.
- Different strategies converge to *different* $\boldsymbol{\lambda}$ but the same (non-zero) residual.

In practice, smooth profiles compatible with the lattice and obeying $|m_i| \le S$ are almost always reachable. Sharp discontinuities, profiles violating local sum rules, or targets inside a first-order magnetization jump are the typical failure modes.

## 6. Conceptual Takeaway

The crucial mental shift is to stop treating DMRG as a black-box ground-state solver inside a constraint loop. Think of the whole thing as **one coupled saddle-point optimization** over $(|\psi\rangle, \boldsymbol{\lambda})$. DMRG sweeps and multiplier updates are then just two flavors of coordinate steps on the same Lagrangian, and they can be interleaved at whatever granularity the problem demands — full DMRG, per-sweep, or per-site.

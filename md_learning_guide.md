# Molecular Dynamics Learning Guide
## Frenkel & Smit — Chapters 4, 6–11

*A structured self-study plan with reading goals, coding projects, and deliverables for each chapter. Every coding project builds on the JAX velocity-Verlet engine from the GitHub repo and produces a notebook you can bring back for discussion.*

---

## How to Use This Guide

Each chapter follows the same rhythm:

1. **Read** the chapter and take notes (half a day).
2. **Code** the projects listed below — extend the MD engine, run simulations, produce plots (one to two days).
3. **Discuss** — bring notebooks here so we can walk through the physics, check your results against known benchmarks, and make sure the math clicks.

The coding projects deliberately span four material classes — **Lennard-Jones liquids/gases**, **crystals (FCC/BCC)**, **polymers/chain molecules**, and **simple molecular fluids** — so that by the end you've built intuition across very different physical regimes.

---

## Prerequisite: Know Your Engine

Before starting Chapter 4, make sure you understand the existing codebase:

- The `MD` class in `md.py`: velocity-Verlet integrator, JAX autograd for forces, `run()` with chunked progress vs. single-scan performance mode.
- Currently the engine only has a `HarmonicPotential` and treats particles independently (no pair interactions, no periodic boundary conditions).
- Everything below asks you to *extend* this engine. Keep each extension in its own module (e.g., `potentials.py`, `thermostats.py`, `analysis_tools.py`) and import into notebooks.

---

## Chapter 4 — Molecular Dynamics Simulations

**Pages 83–130 (48 pages) · Estimated study time: 3–4 days**

### Reading Goals

After reading this chapter you should be able to:

- Explain how an MD simulation is analogous to a real experiment (sample preparation, equilibration, measurement).
- Write down the equipartition theorem and use it to define an instantaneous kinetic temperature T(t) from particle velocities (Eq. 4.1.1–4.1.2).
- Derive the Verlet algorithm from a Taylor expansion and show that the position error is O(Δt⁴) while the velocity error (central difference) is O(Δt²) (Eq. 4.2.3–4.2.4).
- Explain the velocity-Verlet variant and why it is preferred in practice (positions, velocities, and forces all available at the same time step).
- Articulate the difference between short-time and long-time energy conservation and why the Verlet family is preferred over higher-order predictor-corrector schemes for long runs.
- State what Lyapunov instability means for MD trajectories — why two initially close trajectories diverge exponentially, and why this does *not* invalidate statistical predictions (Section 4.3.4).
- Define the mean-squared displacement (MSD), the velocity autocorrelation function (VACF), and the self-diffusion coefficient D, and connect them via the Green-Kubo relation and the Einstein relation (Eq. 4.4.8, 4.6.2).
- Explain minimum-image convention and periodic boundary conditions (PBC) for pair interactions.
- Describe the order-n algorithm for efficiently computing time-correlation functions over long lag times (Section 4.4.2).

### Coding Projects

#### Project 4-A: Lennard-Jones Pair Potential + Periodic Boundary Conditions

**Goal:** Transform the engine from independent particles to an interacting N-body system.

**What to build:**

- Implement the LJ potential V(r) = 4ε[(σ/r)¹² − (σ/r)⁶] as a pair interaction.
- Add a pairwise force routine: loop over all unique pairs, apply the minimum-image convention, and accumulate forces. Use a cutoff rc = 2.5σ and apply the standard tail corrections for energy and pressure.
- Implement periodic boundary conditions (wrap positions back into the box after each step).
- Initialize an FCC lattice at reduced density ρ* = 0.844 and reduced temperature T* = 0.72 (near the LJ triple point).

**What to analyze / deliver (notebook):**

- Plot total energy (KE + PE) vs. time step number. Verify energy conservation: the drift ΔU should be < 10⁻⁴ per particle per time step for dt = 0.004 (reduced units).
- Vary Δt from 0.001 to 0.02 and plot energy drift vs. Δt. Confirm the O(Δt²) scaling of the energy fluctuation.
- Plot the instantaneous temperature. Discuss the ~1/√N_f fluctuations you observe.

#### Project 4-B: Transport Properties — Diffusion in the LJ Fluid

**Goal:** Measure self-diffusion from an MD trajectory using both the MSD and the VACF routes.

**What to build:**

- Write an analysis module that computes the MSD ⟨|r(t) − r(0)|²⟩ averaged over particles and multiple time origins. Handle PBC unwrapping correctly (store unwrapped coordinates).
- Compute the VACF ⟨v(0)·v(t)⟩.
- Implement the order-n algorithm (Section 4.4.2) for efficient long-time correlation functions.

**What to analyze / deliver (notebook):**

- Run at two state points: (a) a dense liquid (ρ* = 0.844, T* = 1.0) and (b) a dilute gas (ρ* = 0.05, T* = 2.0).
- Plot MSD vs. t. Identify the ballistic (∝ t²) and diffusive (∝ t) regimes. Extract D from the slope.
- Plot the VACF. Note the negative dip in the liquid (cage effect) vs. the monotonic decay in the gas. Integrate the VACF to get D via Green-Kubo and compare with the Einstein result.
- Compare your D values against published LJ data (e.g., Meier et al., J. Chem. Phys. 121, 3671, 2004).

#### Project 4-C: Lyapunov Instability Demonstration

**Goal:** Visualize trajectory divergence directly.

**What to build:**

- Run two simulations of the same LJ liquid (64 particles, ρ* = 0.844, T* = 1.0) with initial conditions differing by δv = 10⁻¹⁰ on a single particle.
- At every time step compute Δ(t) = Σᵢ |rᵢ(t) − r'ᵢ(t)|².

**What to analyze / deliver (notebook):**

- Plot log(Δ) vs. t. Extract the largest Lyapunov exponent λ from the linear growth regime.
- Show that despite exponential divergence, macroscopic observables (temperature, pressure, MSD) are indistinguishable between the two runs.

---

## Chapter 6 — Molecular Dynamics in Various Ensembles

**Pages 159–186 (28 pages) · Estimated study time: 3–4 days**

### Reading Goals

After reading this chapter you should be able to:

- Explain why a standard MD simulation samples the NVE (microcanonical) ensemble and why NVT and NPT are often more useful.
- Describe the Andersen thermostat: stochastic velocity reassignment from the Maxwell-Boltzmann distribution, controlled by a collision frequency ν. Understand why this disrupts velocity autocorrelations and makes it unsuitable for measuring transport properties.
- Derive the Nosé-Hoover equations of motion from the extended Lagrangian (Eq. 6.1.3–6.1.8). Explain the role of the coupling parameter Q and its effect on temperature fluctuation dynamics.
- Explain the "flying ice cube" problem and ergodicity failure of a single Nosé-Hoover thermostat for small or stiff systems (the harmonic oscillator pathology). Describe how Nosé-Hoover chains fix this.
- Derive the equations of motion for constant-pressure (NPT) MD using the extended-Lagrangian approach (Andersen barostat). Understand the role of the fictitious piston mass W.
- Describe the Parrinello-Rahman scheme and why allowing the cell shape to fluctuate is essential for studying displacive phase transitions in solids.

### Coding Projects

#### Project 6-A: Thermostat Comparison on a LJ Liquid

**Goal:** Implement three thermostats and understand their impact on dynamics.

**What to build:**

- Implement the Andersen thermostat (stochastic velocity reassignment).
- Implement the Nosé-Hoover thermostat (single chain, with the extended variable ξ).
- Implement velocity rescaling (the simple Berendsen-style approach for comparison, even though it doesn't produce the canonical ensemble rigorously).

**What to analyze / deliver (notebook):**

- Equilibrate a 256-particle LJ liquid at ρ* = 0.75, target T* = 1.0 using each thermostat.
- Plot T(t) over 10⁴ steps for each. For Nosé-Hoover, show the effect of Q on the oscillation period and damping (try Q = 0.1, 1.0, 10.0 in reduced units).
- Histogram the velocity distribution and overlay the Maxwell-Boltzmann prediction. Verify all three yield the correct distribution.
- Compute the VACF under each thermostat. Show that Andersen (high ν) destroys correlations, while Nosé-Hoover preserves them.

#### Project 6-B: Nosé-Hoover Chains on a Harmonic Oscillator

**Goal:** Demonstrate that a single Nosé-Hoover thermostat fails for a 1D harmonic oscillator, and chains fix it.

**What to build:**

- Implement a single 1D harmonic oscillator coupled to a Nosé-Hoover thermostat.
- Implement Nosé-Hoover chains (M = 2 and M = 3 chain lengths).

**What to analyze / deliver (notebook):**

- Run for 10⁵ steps. Plot the phase-space trajectory (x, v).
- For single NH: show the trajectory traces a ring (non-ergodic) — positions and velocities do *not* sample the full Gaussian.
- For NH chains: show the trajectory fills the expected 2D Gaussian. Histogram x and v separately and confirm Boltzmann statistics.

#### Project 6-C: NPT Simulation — Thermal Expansion of a Crystal

**Goal:** Implement constant-pressure MD and study a solid.

**What to build:**

- Implement the Andersen barostat (isotropic volume changes via a rescaled box variable, coupled with a Nosé-Hoover thermostat for temperature control).
- Initialize a 4×4×4 FCC lattice of LJ particles (256 atoms).

**What to analyze / deliver (notebook):**

- Run NPT at P* = 1.0 and several temperatures: T* = 0.5, 1.0, 1.5, 2.0.
- Plot ⟨V⟩ vs. T. Extract the thermal expansion coefficient α = (1/V)(∂V/∂T)_P.
- At T* = 2.0, check whether the crystal has melted by computing the radial distribution function g(r). Compare the sharp FCC peaks at low T with the liquid-like structure at high T.
- Discuss: at what approximate T* does melting occur at this pressure? How does this relate to the known LJ phase diagram?

---

## Chapter 7 — Free Energy Calculations

**Pages 187–220 (34 pages) · Estimated study time: 3–4 days**

### Reading Goals

After reading this chapter you should be able to:

- Explain why free energies cannot be measured directly as simple ensemble averages (unlike energy or pressure) and why special techniques are needed.
- Derive the thermodynamic integration (TI) formula: F(λ=1) − F(λ=0) = ∫₀¹ ⟨∂U/∂λ⟩_λ dλ (Eq. 7.1.6). Understand the linear coupling path and how to choose a smooth λ-schedule.
- Derive the Widom test-particle insertion formula for the excess chemical potential μ_ex (Eq. 7.2.5). Explain when it works (dilute systems) and when it fails (dense liquids, large molecules).
- State the free energy perturbation (FEP) identity (Zwanzig equation): ΔF = −k_BT ln⟨exp(−βΔU)⟩₀. Explain why this is an exact identity but has terrible statistical efficiency when the phase spaces of the reference and target systems don't overlap.
- Describe histogram reweighting / multiple histogram methods (Section 7.3) and how they extract free energy differences from overlapping energy distributions.
- Explain finite-size corrections to the chemical potential (Eq. 7.2.10).

### Coding Projects

#### Project 7-A: Thermodynamic Integration — Turning On the LJ Interaction

**Goal:** Compute the excess Helmholtz free energy of the LJ fluid by "growing" the interaction from an ideal gas.

**What to build:**

- Implement a λ-dependent potential: U(λ) = λ·U_LJ. Use soft-core modifications to avoid singularities at small r when λ → 0 (look up the Beutler soft-core form if needed).
- Run NVT simulations at 10–15 values of λ between 0 and 1 (ρ* = 0.844, T* = 1.5, 108 particles).
- At each λ, measure ⟨∂U/∂λ⟩_λ after equilibration.

**What to analyze / deliver (notebook):**

- Plot ⟨∂U/∂λ⟩ vs. λ. Perform numerical integration (trapezoidal or Gauss-Legendre quadrature).
- Compare your F_ex with the Johnson-Zollweg-Gubbins equation of state for LJ (a standard reference).
- Estimate the statistical error on F_ex by block averaging. Discuss how many λ-points are needed.

#### Project 7-B: Widom Insertion — Chemical Potential of LJ at Various Densities

**Goal:** Measure μ_ex using test-particle insertion and understand where the method breaks down.

**What to build:**

- During an NVT run, periodically insert a ghost particle at a random position, compute ΔU, and accumulate the Boltzmann factor ⟨exp(−βΔU)⟩.
- Run at T* = 1.5 and densities ρ* = 0.1, 0.3, 0.5, 0.7, 0.85.

**What to analyze / deliver (notebook):**

- Plot μ_ex vs. ρ*. Show convergence as a function of the number of insertions.
- Demonstrate that at high density (ρ* = 0.85) the insertion acceptance is vanishingly small and the estimate has huge variance. Discuss why.
- Compare with the TI result from Project 7-A at the overlapping state point.

#### Project 7-C: Free Energy Perturbation Between Two Simple Fluids

**Goal:** Use FEP (Zwanzig equation) to compute ΔF between two systems with slightly different potentials.

**What to build:**

- Define two LJ potentials with σ₁ = 1.0 and σ₂ = 1.05 (a 5% size change at the same ε).
- Run a simulation with potential 1 and at each saved frame evaluate ΔU = U₂ − U₁. Compute ΔF = −k_BT ln⟨exp(−βΔU)⟩₁.
- Also do the reverse (simulate system 2, perturb toward system 1).

**What to analyze / deliver (notebook):**

- Report ΔF from both directions. If they agree well, the phase-space overlap is good.
- Now try σ₂ = 1.20 (a 20% change). Show that ΔF from the two directions diverges — the perturbation is too large.
- Discuss how one would bridge this gap (staging, BAR, or MBAR).

---

## Chapter 8 — The Gibbs Ensemble

**Pages 221–244 (24 pages) · Estimated study time: 2–3 days**

### Reading Goals

After reading this chapter you should be able to:

- Explain why locating liquid-vapor coexistence in a small simulation box by direct simulation is impractical (interface energy dominates, Table 8.1).
- Describe the three trial moves of the Gibbs Ensemble Monte Carlo (GEMC): particle displacement, volume exchange, and particle transfer between two boxes.
- Derive the acceptance criteria for each move from detailed balance in the combined NVT ensemble of two boxes (Eq. 8.3.2–8.3.7).
- Understand the partition-function argument for why the Gibbs ensemble samples coexisting densities without requiring an explicit interface.
- Discuss when GEMC fails: dense liquids near the critical point (poor particle-transfer acceptance), and solid-liquid coexistence (particle insertion into a crystal is essentially impossible).
- Describe the Gibbs-Duhem integration technique as an alternative for tracing coexistence curves (Section 8.5).

### Coding Projects

#### Project 8-A: Gibbs Ensemble MC for LJ Vapor-Liquid Equilibrium

**Goal:** Map the liquid-vapor coexistence curve of the LJ fluid.

**What to build (this is MC, not MD — a separate script/notebook):**

- Implement the three GEMC moves: displacement (Metropolis in each box), volume change (logarithmic sampling of the volume ratio), and particle swap.
- Use 256 total LJ particles, rc = 2.5σ with standard long-range corrections.
- Run at T* = 0.9, 1.0, 1.1, 1.2, 1.3 (the critical temperature is T*_c ≈ 1.31).

**What to analyze / deliver (notebook):**

- At each T, plot the density histograms for both boxes. Show that one equilibrates to a liquid density and the other to a vapor density.
- Plot the coexistence curve (ρ_liquid, ρ_vapor vs. T) and overlay published LJ coexistence data.
- Apply the law of rectilinear diameters and the scaling law ρ_L − ρ_V ∝ (T_c − T)^β to estimate T_c and ρ_c from your data.
- Discuss the acceptance rate for particle swaps. Show it drops rapidly as T decreases (the liquid becomes too dense for random insertions).

#### Project 8-B: Finite-Size Effects in the Gibbs Ensemble

**Goal:** Understand how system size affects the coexistence curve.

**What to analyze / deliver (notebook):**

- Repeat one temperature (e.g. T* = 1.0) with N_total = 128, 256, 512.
- Plot the measured coexisting densities vs. 1/N and discuss extrapolation to the thermodynamic limit.
- Relate back to the interface-fraction table (Table 8.1) in the chapter.

---

## Chapter 9 — Other Methods to Study Coexistence

**Pages 245–260 (16 pages) · Estimated study time: 2 days**

### Reading Goals

After reading this chapter you should be able to:

- Describe the semigrand ensemble: one component has its chemical potential fixed while the identity of particles can switch between species. Explain when this is useful (binary mixtures, alloys).
- Derive the acceptance rule for an identity-swap move in the semigrand ensemble.
- Explain Gibbs-Duhem integration: once one coexistence point is known, the Clausius-Clapeyron equation (or its generalized form) can be integrated numerically to trace the entire coexistence curve.
- Understand why GDI is particularly powerful for solid-liquid coexistence, where the Gibbs ensemble fails.

### Coding Projects

#### Project 9-A: Semigrand Ensemble for a Binary LJ Mixture

**Goal:** Map the phase diagram of a symmetric binary LJ mixture by fixing Δμ and measuring composition.

**What to build:**

- Implement a semigrand MC simulation for a binary LJ mixture: A-A and B-B have the same ε but σ_BB = 1.2σ_AA, and the cross-interaction uses Lorentz-Berthelot rules.
- Trial moves: standard Metropolis displacement plus identity swaps (A → B or B → A) at fixed Δμ = μ_B − μ_A.

**What to analyze / deliver (notebook):**

- At T* = 1.0 and ρ* = 0.7, run at several values of Δμ. Plot ⟨x_B⟩ vs. Δμ.
- If there is a first-order demixing transition, ⟨x_B⟩ will jump discontinuously. Identify the coexisting compositions.
- Discuss the physics: the size asymmetry drives demixing because the entropy of mixing cannot overcome the packing penalty.

#### Project 9-B: Gibbs-Duhem Integration for LJ Solid-Liquid Coexistence

**Goal:** Trace the melting curve starting from a known coexistence point.

**What to build:**

- Start from the known LJ solid-liquid coexistence point at P* ≈ 1.0, T* ≈ 0.77 (from the literature or from your Project 10-A result).
- Implement the predictor-corrector integration of the Clausius-Clapeyron equation: dP/dT = ΔS/ΔV = ΔH/(TΔV), where ΔH and ΔV are measured from separate NPT simulations of liquid and solid at the current (P, T).

**What to analyze / deliver (notebook):**

- Trace the melting curve from P* = 1 up to P* = 20 in steps of ΔP* = 1.
- Plot the melting curve T_m(P) and compare with published LJ phase diagrams.
- Discuss convergence: how large must the NPT runs be at each point to get ΔH and ΔV accurately enough for the integration to not drift?

---

## Chapter 10 — Free Energies of Solids

**Pages 261–288 (28 pages) · Estimated study time: 3–4 days**

### Reading Goals

After reading this chapter you should be able to:

- Explain why free energies of solids cannot be obtained by the Gibbs ensemble or simple particle insertion, and why a different reference system is needed.
- Describe the Einstein crystal method (Frenkel-Ladd): tether each particle to its lattice site with a harmonic spring of strength α, then use TI to switch from the interacting solid to the Einstein crystal, whose free energy is known analytically (Eq. 10.2.1–10.2.3).
- Explain the constraint on the center of mass and why it is necessary (to avoid the crystal drifting).
- Derive the condition for choosing optimal spring constants α_i (Eq. 10.2.4) — the mean-squared displacement in the interacting and Einstein crystals should match.
- Describe how the method extends to molecular solids (Section 10.3): additional orientational degrees of freedom, the interacting-Einstein-crystal approach, and why orientationally disordered solids require special care.
- Explain how the Frenkel-Ladd free energy is combined with the equation of state (P vs. ρ) to locate the solid-liquid coexistence point.

### Coding Projects

#### Project 10-A: Frenkel-Ladd Free Energy of the LJ FCC Solid

**Goal:** Compute the absolute Helmholtz free energy of the LJ FCC solid at a single state point.

**What to build:**

- Set up a 4×4×4 FCC lattice of LJ particles (256 atoms) at ρ* = 1.04 (compressed solid), T* = 0.72.
- Implement a λ-dependent potential: U(λ) = U_LJ + λ Σᵢ αᵢ(rᵢ − r₀ᵢ)² and constrain the center of mass.
- Choose αᵢ using the condition from Eq. 10.2.4 (run a short NVT simulation of the unconstrained solid and measure ⟨(rᵢ − r₀ᵢ)²⟩).
- Run NVT simulations at 10–15 values of λ from 0 to λ_max and measure ⟨∂U/∂λ⟩.

**What to analyze / deliver (notebook):**

- Plot ⟨∂U/∂λ⟩ vs. λ. Perform the TI integral.
- Add the analytically known free energy of the Einstein crystal to obtain F_total.
- Compare with published Frenkel-Ladd results for the LJ FCC solid (e.g., Mastny & de Pablo, J. Chem. Phys. 127, 104504, 2007).
- Discuss: what happens if α is too small (crystal drifts/melts during the integration) or too large (poor overlap with the interacting system)?

#### Project 10-B: Solid-Liquid Coexistence from Free Energy + EOS

**Goal:** Use the free energy to locate the LJ melting point.

**What to build / analyze:**

- From Project 10-A you have F_solid at one (ρ, T). Run a series of NVT simulations at different ρ along the same isotherm to get the equation of state P(ρ) for the solid.
- From your liquid simulations (Projects 7-A/B) or new runs, get P(ρ) and F for the liquid at the same T.
- Find the coexistence point by the common-tangent construction on F(V) or equivalently by finding the (P, μ) crossing.

**What to analyze / deliver (notebook):**

- Plot F/N vs. V/N for both phases with the common tangent.
- Report the coexisting densities ρ_solid and ρ_liquid and the coexistence pressure. Compare with literature (ρ_solid ≈ 1.04, ρ_liquid ≈ 0.94 at T* ≈ 0.77).
- Discuss the sensitivity of the result to the accuracy of the free energy and the EOS.

---

## Chapter 11 — Free Energy of Chain Molecules

**Pages 289–308 (20 pages) · Estimated study time: 2–3 days**

### Reading Goals

After reading this chapter you should be able to:

- Explain why Widom particle insertion fails catastrophically for chain molecules (the probability of inserting a whole chain into a dense fluid without overlap is vanishingly small).
- Describe the chemical potential as reversible work: break the insertion of a chain into elementary steps and sum the free energy contributions (Section 11.1).
- Derive the Rosenbluth sampling method for chain molecules: grow the chain segment by segment, choosing each segment direction from a set of k trial orientations biased by the Boltzmann weight. The ratio of the Rosenbluth weight of the real chain to that of the ideal chain gives Q/Q_id (Eq. 11.2.11–11.2.14).
- Explain the extension to continuously deformable (flexible) molecules: internal and external potential energy, the conformational partition function Q_id, and how the Rosenbluth approach still applies (Section 11.2.2).
- Describe the recursive algorithm and when it is more efficient than the Rosenbluth approach.

### Coding Projects

#### Project 11-A: Rosenbluth Chain Insertion in a LJ Solvent

**Goal:** Measure the excess chemical potential of a short chain molecule dissolved in a LJ fluid.

**What to build:**

- Model a short chain (ℓ = 8 segments) with fixed bond lengths, a bending potential U_bend = k_θ(θ − θ₀)², and LJ non-bonded interactions (segment-segment and segment-solvent).
- Implement the Rosenbluth sampling algorithm: grow the chain segment by segment from a random starting point, choosing each orientation from k = 10 trial directions, weighted by the Boltzmann factor of the external energy.
- Run in a box of ~200 LJ solvent particles at ρ* = 0.6, T* = 1.5 (NVT with Nosé-Hoover from Chapter 6).

**What to analyze / deliver (notebook):**

- Accumulate the average Rosenbluth weight ⟨W⟩ over many trial insertions and compute μ_ex = −k_BT ln(⟨W⟩/⟨W_id⟩).
- Plot convergence of μ_ex vs. number of insertions.
- Vary chain length ℓ = 4, 8, 12, 16 and plot μ_ex vs. ℓ. Discuss the linear scaling expected for long chains (each additional segment contributes roughly the same free energy increment).

#### Project 11-B: Thermodynamic Integration for a Polymer in Solvent

**Goal:** Cross-validate the Rosenbluth result using TI to "grow" the chain interactions.

**What to build:**

- Place the 8-segment chain in the solvent. Define U(λ) = U_solvent-solvent + λ · U_chain-solvent, so at λ = 0 the chain doesn't interact with the solvent (ideal chain in a cavity), and at λ = 1 it's fully interacting.
- Run NVT at 10 values of λ.

**What to analyze / deliver (notebook):**

- Plot ⟨∂U/∂λ⟩ vs. λ and integrate to get ΔF.
- Add the ideal-chain contribution (analytically known from the bond/angle potential) to get μ_ex.
- Compare with the Rosenbluth result from Project 11-A. Discuss which method is more efficient for this chain length.

#### Project 11-C: Chain Length and Solvent Quality

**Goal:** Go beyond the textbook — explore how solvent quality affects chain conformation and chemical potential.

**What to build / analyze:**

- Define a "good solvent" (ε_chain-solvent = 1.0 ε) and a "poor solvent" (ε_chain-solvent = 0.5 ε).
- Run MD simulations of a single 16-segment chain dissolved in each solvent at T* = 1.5. Use your Nosé-Hoover thermostat.
- Measure the radius of gyration R_g and the end-to-end distance R_ee as a function of time.

**What to analyze / deliver (notebook):**

- Plot R_g(t) and ⟨R_g⟩ for both solvents. Show that the chain is expanded in the good solvent and collapsed in the poor solvent (the coil-to-globule transition).
- Compute the Flory exponent ν from ⟨R_ee²⟩ ∝ ℓ^(2ν) by running chains of length ℓ = 8, 16, 32 in the good solvent. Expect ν ≈ 0.588 (3D self-avoiding walk) — though with short chains the scaling won't be perfect.
- Discuss the connection to the free energy: a collapsed chain has lower excess chemical potential (more favorable solvent contacts per unit volume).

---

## Summary of Deliverables by Material Class

| Material | Projects |
|----------|----------|
| **LJ liquid / gas** | 4-A, 4-B, 4-C, 6-A, 7-A, 7-B, 7-C, 8-A, 8-B |
| **FCC / BCC crystal** | 6-C, 10-A, 10-B |
| **Binary mixture / alloy** | 9-A |
| **Polymer / chain molecule** | 11-A, 11-B, 11-C |
| **Coexistence / phase boundary** | 8-A, 8-B, 9-B, 10-B |

---

## Suggested Order and Pacing

| Week | Chapter | Projects | Key new engine features |
|------|---------|----------|------------------------|
| 1 | Ch 4 | 4-A, 4-B, 4-C | LJ potential, PBC, force loop, MSD/VACF analysis |
| 2 | Ch 6 | 6-A, 6-B, 6-C | Thermostats (Andersen, NH, NH-chains), NPT barostat, RDF |
| 3 | Ch 7 | 7-A, 7-B, 7-C | TI framework, Widom insertion, FEP |
| 4 | Ch 8 | 8-A, 8-B | GEMC (MC code, not MD) |
| 5 | Ch 9 | 9-A, 9-B | Semigrand MC, Gibbs-Duhem integration |
| 6 | Ch 10 | 10-A, 10-B | Einstein crystal TI, EOS integration, common tangent |
| 7 | Ch 11 | 11-A, 11-B, 11-C | Chain potentials, Rosenbluth sampling, R_g analysis |

*Each "week" is roughly 1 day reading + 2 days coding. Adjust as needed — some chapters (4, 6, 7) are heavier than others (9).*

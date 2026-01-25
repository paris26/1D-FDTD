Alright, let's build this from the ground up, Feynman-style.

---

## Part 1: What You Already Know — Waves in Vacuum

Feynman's picture of EM waves in vacuum comes down to two coupled equations (in 1D, with E pointing in x and wave traveling in z):

$$\frac{\partial E}{\partial z} = -\mu_0 \frac{\partial H}{\partial t}$$

$$\frac{\partial H}{\partial z} = -\varepsilon_0 \frac{\partial E}{\partial t}$$

The physical story: a changing E-field creates a curling H-field, and a changing H-field creates a curling E-field. They chase each other through space at speed $c = 1/\sqrt{\mu_0 \varepsilon_0}$.

Think of it like a row of dominoes, except each domino falling makes the next one fall *and* the previous one wants to stand back up — you get a wave, not a collapse.

---

## Part 2: Adding a Simple Material (No Dispersion Yet)

When you put a material in the way, the atoms respond to the electric field. Electrons get tugged, creating tiny dipoles. This collective response is the **polarization** P.

The total "electric effect" is now:

$$D = \varepsilon_0 E + P$$

For simple materials, P is proportional to E:

$$P = \varepsilon_0 \chi E$$

where χ (chi) is the susceptibility. So:

$$D = \varepsilon_0(1 + \chi)E = \varepsilon_0 \varepsilon_r E = \varepsilon E$$

The wave equations become the same, just with ε instead of ε₀. Wave speed drops to:

$$v = \frac{c}{n} = \frac{c}{\sqrt{\varepsilon_r}}$$

Glass has n ≈ 1.5, so light slows to about 200,000 km/s inside it.

**So far, nothing complicated.** You just replace ε₀ with ε everywhere.

---

## Part 3: The Problem — Real Materials Are Dispersive

Here's where it gets interesting. In real materials, the electrons aren't infinitely fast. They're like masses on springs:

```
        Spring (restoring force)
           |
    ~~~~~~●~~~~~~  ← Electron
           |
        Damping (friction)
```

When you drive a mass-spring system at different frequencies:
- **Low frequency**: The mass follows the driving force easily (large response)
- **At resonance**: Huge response (this is where materials absorb strongly)
- **High frequency**: The mass can't keep up (small response)

This means **ε depends on frequency**: ε(ω)

**Numerical example:**

Water has a resonance around 20 GHz (microwave). Below that frequency, ε_r ≈ 80 (water is very polarizable). Above ~100 GHz, ε_r drops toward ~4. This is why microwaves heat water effectively — you're driving near resonance.

---

## Part 4: The Lorentz Model — Your First Dispersion Model

The driven harmonic oscillator gives the **Lorentz model**:

$$\varepsilon(\omega) = \varepsilon_\infty + \frac{\Delta\varepsilon \cdot \omega_0^2}{\omega_0^2 - \omega^2 + i\gamma\omega}$$

Where:
- ε_∞ = high-frequency limit (electrons can't respond)
- Δε = ε_static - ε_∞ (the "strength" of the resonance)  
- ω₀ = resonance frequency
- γ = damping rate

**Analogy**: Think of pushing a kid on a swing.
- Push slowly (low ω) → swing follows your hand
- Push at natural frequency → huge oscillations
- Push super fast → swing barely moves

The complex part (iγω) represents energy loss — the material absorbs some wave energy as heat.

---

## Part 5: Why This Creates a Problem for Simulation

In the **frequency domain**, dispersion is easy. You have a single frequency ω, you compute ε(ω), done:

$$D(\omega) = \varepsilon(\omega) E(\omega)$$

But FDTD works in the **time domain**. You're updating E and H step by step in time. A multiplication in frequency domain becomes a **convolution** in time domain:

$$D(t) = \varepsilon_\infty E(t) + \int_0^t \chi(t-\tau) E(\tau) \, d\tau$$

That integral means: "the polarization now depends on the *entire history* of the electric field."

**Analogy**: Imagine you're pushing that swing, but the swing has memory. Its position now depends on every push you've ever given it, weighted by how long ago each push was. You can't just look at "force now → position now."

This is the core computational challenge.

---

## Part 6: The Solution — Auxiliary Differential Equation (ADE)

Here's the clever trick. Instead of computing that ugly convolution, we introduce an auxiliary variable — the **polarization current** J_p — that evolves according to its own differential equation.

For the Lorentz model, the polarization P satisfies:

$$\frac{\partial^2 P}{\partial t^2} + \gamma \frac{\partial P}{\partial t} + \omega_0^2 P = \varepsilon_0 \Delta\varepsilon \cdot \omega_0^2 E$$

This is just the driven harmonic oscillator equation!

Now instead of one big convolution, you have three coupled equations:
1. Maxwell's equation for E (with a P term)
2. Maxwell's equation for H  
3. The oscillator equation for P

Each one is local in time — no memory integral needed.

---

## Part 7: The FDTD Algorithm

Standard FDTD discretizes space and time on a staggered grid:

```
Time:    n-½       n        n+½       n+1
         |         |         |         |
    -----H---------E---------H---------E-----
         |         |         |         |
Space:   k       k+½         k       k+½
```

E and H are computed at alternating half-steps in both space and time. This is called the **Yee scheme**.

**Update equations (vacuum):**

$$H^{n+1/2}_k = H^{n-1/2}_k - \frac{\Delta t}{\mu_0 \Delta z}(E^n_{k+1/2} - E^n_{k-1/2})$$

$$E^{n+1}_{k+1/2} = E^n_{k+1/2} - \frac{\Delta t}{\varepsilon_0 \Delta z}(H^{n+1/2}_{k+1} - H^{n+1/2}_k)$$

**With dispersion**, the E update gets modified to include the polarization:

$$E^{n+1} = \text{(complicated expression involving } P^n, P^{n-1}, E^n\text{)}$$

And you add an update equation for P.

---

## Part 8: What You'd Actually Validate

The project asks you to compute **reflection and transmission coefficients** and compare to theory.

**Setup**: A plane wave hits an interface between vacuum and dispersive material.

```
    Vacuum          |     Dispersive Material
                    |
    →→→ Incident    |     →→→ Transmitted
    ←←← Reflected   |
                    |
               Interface
```

**Analytical solution**: The Fresnel equations give reflection coefficient:

$$r(\omega) = \frac{1 - n(\omega)}{1 + n(\omega)}$$

where $n(\omega) = \sqrt{\varepsilon(\omega)}$.

**Your simulation**: 
1. Send in a pulse (contains many frequencies)
2. Record reflected and transmitted pulses
3. Fourier transform both
4. Compare to Fresnel prediction

If they match, your dispersive FDTD is validated.

---

## What You Should Read (In Order)

**1. For the physics of dispersion:**
- Griffiths, *Electrodynamics*, Chapter 9.4 (Absorption and Dispersion)
- This gives you the Lorentz model derivation from first principles

**2. For basic FDTD:**
- Taflove & Hagness, *Computational Electrodynamics*, Chapters 1-3
- Or the free resource: Gedney's *Introduction to FDTD* (the reference in your PDF)

**3. For dispersive FDTD specifically:**
- Taflove & Hagness, Chapter 9 (Dispersive Materials)
- The ADE method is in section 9.4

**4. Quick practical intro:**
- Search "FDTD 101" by EMPossible on YouTube — there's a free lecture series that builds everything from scratch

---

## Your Rough Draft Outline for Zygkiridis

Here's how I'd structure a 1-page proposal:

> **Objective**: Implement 1D FDTD with Lorentz dispersion model and validate against analytical Fresnel coefficients.
>
> **Background**: [2-3 sentences on why dispersive modeling matters — plasmonics, metamaterials, optical materials]
>
> **Approach**:
> 1. Implement standard 1D FDTD for vacuum (verify wave speed = c)
> 2. Add simple dielectric (verify wave speed = c/n)
> 3. Implement ADE formulation for Lorentz dispersion
> 4. Test: Gaussian pulse incident on dispersive half-space
> 5. Validate: Compare computed R(ω), T(ω) to analytical Fresnel equations
>
> **Deliverables**: Python/MATLAB code, validation plots, brief report on method and results
>
> **Timeline**: [rough weekly breakdown over semester]
>
> **References**: [Gedney, Taflove, Griffiths]

---

## One Concrete Starting Exercise

Before meeting him, implement basic 1D FDTD in vacuum. Just 50 lines of Python:

```python
import numpy as np
import matplotlib.pyplot as plt

# Grid
nz = 200
nt = 500
dz = 1e-9  # 1 nm cells
dt = dz / (3e8 * 2)  # CFL condition: dt < dz/c

E = np.zeros(nz)
H = np.zeros(nz)

# Time stepping
for n in range(nt):
    # Source: Gaussian pulse at center
    E[nz//2] += np.exp(-((n - 30)/10)**2)
    
    # Update H
    H[:-1] += (dt / (4e-7 * np.pi * dz)) * (E[1:] - E[:-1])
    
    # Update E
    E[1:] += (dt / (8.85e-12 * dz)) * (H[1:] - H[:-1])
```

If you can show him this working (wave propagates at c, reflects off boundaries), he'll know you're serious.

Want me to expand on any part — the physics of a specific dispersion model, the numerical stability analysis, or the validation methodology?
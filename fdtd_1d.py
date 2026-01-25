"""
1D FDTD Simulation - Foundation
================================
Simulates electromagnetic wave propagation in 1D vacuum.

The algorithm:
1. E and H fields are staggered in space (Yee grid)
2. E and H are updated alternately in time (leap-frog)
3. A Gaussian pulse is injected as a source

Run this to see a wave propagate and reflect off boundaries.
"""

import numpy as np

# Try to import matplotlib - visualization is optional
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not installed. Skipping visualization.")

# =============================================================================
# Physical Constants
# =============================================================================
c = 2.998e8          # Speed of light (m/s)
eps0 = 8.854e-12     # Permittivity of free space (F/m)
mu0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)

# =============================================================================
# Grid Setup
# =============================================================================
nz = 200             # Number of spatial cells
dz = 1e-9            # Cell size: 1 nanometer

# Time step from CFL condition: dt < dz/c
# We use a Courant number of 0.5 for stability
courant = 0.5
dt = courant * dz / c

nt = 500             # Number of time steps

print(f"Grid: {nz} cells, cell size = {dz*1e9:.1f} nm")
print(f"Time step: dt = {dt*1e18:.2f} attoseconds")
print(f"Total simulation time: {nt*dt*1e15:.2f} femtoseconds")

# =============================================================================
# Field Arrays
# =============================================================================
# E[k] is at position k*dz
# H[k] is at position (k + 0.5)*dz (staggered by half a cell)
E = np.zeros(nz)
H = np.zeros(nz)

# =============================================================================
# Update Coefficients
# =============================================================================
# From the update equations:
#   H_new = H_old - (dt / mu0 / dz) * (E[k+1] - E[k])
#   E_new = E_old - (dt / eps0 / dz) * (H[k] - H[k-1])

coef_H = dt / (mu0 * dz)  # Coefficient for H update
coef_E = dt / (eps0 * dz)  # Coefficient for E update

# =============================================================================
# Source Parameters
# =============================================================================
# Gaussian pulse centered at time step n0 with width spread
source_pos = nz // 4      # Source location (1/4 into the grid)
n0 = 50                   # Center of pulse in time
spread = 15               # Width of Gaussian

def gaussian_source(n):
    """Gaussian pulse at time step n."""
    return np.exp(-((n - n0) / spread) ** 2)

# =============================================================================
# Main FDTD Loop
# =============================================================================
# Storage for visualization
E_history = []
snapshot_interval = 10

print(f"\nRunning {nt} time steps...")

for n in range(nt):
    # --------------------------------------------------
    # Step 1: Update H field (uses current E)
    # --------------------------------------------------
    # H[k] sits between E[k] and E[k+1]
    # We update H[0] through H[nz-2] (H[nz-1] would need E[nz] which doesn't exist)
    H[:-1] = H[:-1] - coef_H * (E[1:] - E[:-1])

    # --------------------------------------------------
    # Step 2: Inject source into E field
    # --------------------------------------------------
    # "Soft source" - adds to existing field rather than replacing it
    E[source_pos] += gaussian_source(n)

    # --------------------------------------------------
    # Step 3: Update E field (uses new H)
    # --------------------------------------------------
    # E[k] sits between H[k-1] and H[k]
    # We update E[1] through E[nz-1] (E[0] would need H[-1] which doesn't exist)
    E[1:] = E[1:] - coef_E * (H[1:] - H[:-1])

    # --------------------------------------------------
    # Boundary conditions: simple reflection (PEC)
    # --------------------------------------------------
    # E = 0 at boundaries (perfect electric conductor)
    E[0] = 0
    E[-1] = 0

    # Save snapshots for plotting
    if n % snapshot_interval == 0:
        E_history.append(E.copy())

print("Done!")

# =============================================================================
# Visualization (optional - requires matplotlib)
# =============================================================================
E_history = np.array(E_history)

# Print some results even without plotting
print(f"\nFinal E field max: {np.max(np.abs(E)):.4f}")
print(f"Wave traveled approximately {c * nt * dt * 1e9:.1f} nm")

if HAS_MATPLOTLIB:
    # Plot 1: Waterfall plot showing wave evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Space axis in nanometers
    z = np.arange(nz) * dz * 1e9

    # Plot several snapshots
    n_plots = min(10, len(E_history))
    for i in range(n_plots):
        idx = i * len(E_history) // n_plots
        time_fs = idx * snapshot_interval * dt * 1e15
        ax1.plot(z, E_history[idx] + i * 0.3, label=f't = {time_fs:.1f} fs')

    ax1.set_xlabel('Position (nm)')
    ax1.set_ylabel('E field (offset for clarity)')
    ax1.set_title('Wave Propagation Snapshots')
    ax1.axvline(x=source_pos * dz * 1e9, color='red', linestyle='--', alpha=0.5, label='Source')
    ax1.legend(fontsize=8)

    # Plot 2: Space-time diagram (2D colormap)
    time_axis = np.arange(len(E_history)) * snapshot_interval * dt * 1e15
    im = ax2.imshow(E_history, aspect='auto', origin='lower',
                    extent=[0, z[-1], 0, time_axis[-1]],
                    cmap='RdBu', vmin=-1, vmax=1)
    ax2.set_xlabel('Position (nm)')
    ax2.set_ylabel('Time (fs)')
    ax2.set_title('Space-Time Diagram')
    ax2.axvline(x=source_pos * dz * 1e9, color='green', linestyle='--', alpha=0.7)
    plt.colorbar(im, ax=ax2, label='E field')

    plt.tight_layout()
    plt.savefig('fdtd_result.png', dpi=150)
    plt.show()
    print("\nFigure saved to fdtd_result.png")
else:
    print("\nInstall matplotlib to see visualization: pip install matplotlib")

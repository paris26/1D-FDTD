# /// script
# dependencies = ["torch"]
# ///

import torch
import math
import time

# --- Configuration ---
NX, NY, NZ = 64, 64, 64
MAX_TIME = 300

# Physical constants
C0 = 3.0e8              # Speed of light (m/s)
EPS0 = 8.854e-12        # Permittivity of free space
MU0 = 1.257e-6          # Permeability of free space

# Grid parameters
DX = 1e-3               # Grid spacing (1 mm)
DY = DX
DZ = DX
SC = 0.5                # Courant number (stability factor)
DT = SC * DX / C0       # Time step

# Source parameters (Gaussian pulse)
SRC_POS = (NX // 2, NY // 2, NZ // 2)
PULSE_WIDTH = 20.0
PULSE_DELAY = 40.0
PROBE_POS = (NX // 2 + 10, NY // 2, NZ // 2)

# --- Check for GPU ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("No GPU found, using CPU")

# --- Initialize Fields (Zero everywhere) ---
# Shape: (1, NX, NY, NZ) for easier broadcasting if needed
# We use float32 for performance, similar to the wgpu solution
zeros = lambda: torch.zeros((NX, NY, NZ), device=device, dtype=torch.float32)

Ex = zeros()
Ey = zeros()
Ez = zeros()
Hx = zeros()
Hy = zeros()
Hz = zeros()

# --- Material Coefficients (Free Space) ---
# In free space: conductivity sigma = 0, magnetic loss sigma_m = 0.
# The equations simplify to:
#   CA = 1, CP = 1
#   CB = DT / EPS0
#   CQ = DT / MU0

CA = torch.ones((NX, NY, NZ), device=device)
CB = torch.full((NX, NY, NZ), DT / EPS0, device=device)
CP = torch.ones((NX, NY, NZ), device=device)
CQ = torch.full((NX, NY, NZ), DT / MU0, device=device)

# Precompute inverse spatial steps for speed
inv_dx = 1.0 / DX
inv_dy = 1.0 / DY
inv_dz = 1.0 / DZ

# --- The "Shift & Add" Operations (Finite Differences) ---
# We use torch.roll to implement spatial shifts.
#   f[i+1] is roll(f, -1)   (shift left brings next element to current pos)
#   f[i-1] is roll(f, +1)   (shift right brings prev element to current pos)

def update_h():
    """Update magnetic fields using Faraday's Law."""
    global Hx, Hy, Hz
    
    # 1. Shift & Add Layer (Spatial Derivatives of E)
    # curl_E_x = dEz/dy - dEy/dz
    dEz_dy = (torch.roll(Ez, -1, dims=1) - Ez) * inv_dy
    dEy_dz = (torch.roll(Ey, -1, dims=2) - Ey) * inv_dz
    
    # curl_E_y = dEx/dz - dEz/dx
    dEx_dz = (torch.roll(Ex, -1, dims=2) - Ex) * inv_dz
    dEz_dx = (torch.roll(Ez, -1, dims=0) - Ez) * inv_dx
    
    # curl_E_z = dEy/dx - dEx/dy
    dEy_dx = (torch.roll(Ey, -1, dims=0) - Ey) * inv_dx
    dEx_dy = (torch.roll(Ex, -1, dims=1) - Ex) * inv_dy

    # 2. Hadamard Product & Summation Layer
    # Update H fields: H_new = CP * H_old - CQ * curl_E
    # (Note: In the paper, the signs depend on the exact grid definition. Standard Yee: H = H - curlE)
    Hx = CP * Hx - CQ * (dEz_dy - dEy_dz)
    Hy = CP * Hy - CQ * (dEx_dz - dEz_dx)
    Hz = CP * Hz - CQ * (dEy_dx - dEx_dy)

def update_e():
    """Update electric fields using Ampere's Law."""
    global Ex, Ey, Ez
    
    # 1. Shift & Add Layer (Spatial Derivatives of H)
    # curl_H_x = dHz/dy - dHy/dz
    # Note: For E-update, we need H[i] - H[i-1], so we roll +1
    dHz_dy = (Hz - torch.roll(Hz, 1, dims=1)) * inv_dy
    dHy_dz = (Hy - torch.roll(Hy, 1, dims=2)) * inv_dz
    
    # curl_H_y = dHx/dz - dHz/dx
    dHx_dz = (Hx - torch.roll(Hx, 1, dims=2)) * inv_dz
    dHz_dx = (Hz - torch.roll(Hz, 1, dims=0)) * inv_dx
    
    # curl_H_z = dHy/dx - dHx/dy
    dHy_dx = (Hy - torch.roll(Hy, 1, dims=0)) * inv_dx
    dHx_dy = (Hx - torch.roll(Hx, 1, dims=1)) * inv_dy
    
    # 2. Hadamard Product & Summation Layer
    # Update E fields: E_new = CA * E_old + CB * curl_H
    Ex = CA * Ex + CB * (dHz_dy - dHy_dz)
    Ey = CA * Ey + CB * (dHx_dz - dHz_dx)
    Ez = CA * Ez + CB * (dHy_dx - dHx_dy)

# --- Main Simulation Loop ---
print("Starting 3D FDTD Simulation (PyTorch)...")
t0 = time.time()

for n in range(MAX_TIME):
    # Update H fields
    update_h()
    
    # Update E fields
    update_e()
    
    # Source Injection (Gaussian Pulse)
    t = float(n) - PULSE_DELAY
    pulse = math.exp(-(t**2) / (PULSE_WIDTH**2))
    Ez[SRC_POS] += pulse  # Soft source addition
    
    # Probe Reading
    val = Ez[PROBE_POS].item()
    print(f"t={n:3d}  Ez[probe] = {val: .6e}")

t1 = time.time()
print(f"\nSimulation complete in {t1 - t0:.2f} seconds.")

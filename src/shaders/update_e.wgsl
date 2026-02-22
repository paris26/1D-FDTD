// ------------------------------------------------------------------
// update_e.wgsl  –  Electric field update (Shift & Add → Hadamard → Sum)
//
// Ex[i,j,k] = CA * Ex  +  CB * ( dHz/dy - dHy/dz )
// Ey[i,j,k] = CA * Ey  +  CB * ( dHx/dz - dHz/dx )
// Ez[i,j,k] = CA * Ez  +  CB * ( dHy/dx - dHx/dy )
// ------------------------------------------------------------------

struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    _pad: u32,
    inv_dx: f32,
    inv_dy: f32,
    inv_dz: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> p: Params;

// Magnetic fields (read-only this pass)
@group(0) @binding(1) var<storage, read>       hx: array<f32>;
@group(0) @binding(2) var<storage, read>       hy: array<f32>;
@group(0) @binding(3) var<storage, read>       hz: array<f32>;

// Electric fields (read-write)
@group(0) @binding(4) var<storage, read_write> ex: array<f32>;
@group(0) @binding(5) var<storage, read_write> ey: array<f32>;
@group(0) @binding(6) var<storage, read_write> ez: array<f32>;

// Material coefficients
@group(0) @binding(7) var<storage, read>       ca: array<f32>;
@group(0) @binding(8) var<storage, read>       cb: array<f32>;

fn idx(i: u32, j: u32, k: u32) -> u32 {
    return i + p.nx * (j + p.ny * k);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;

    // Guard: skip index 0 on each axis (need i-1, j-1, k-1)
    if (i == 0u || j == 0u || k == 0u || i >= p.nx || j >= p.ny || k >= p.nz) {
        return;
    }

    let id  = idx(i, j, k);
    let ca_v = ca[id];
    let cb_v = cb[id];

    // --- Shift & Add  (finite differences of H) -----------------------

    // Ex:  dHz/dy - dHy/dz
    let dHz_dy = (hz[id] - hz[idx(i, j - 1u, k)]) * p.inv_dy;
    let dHy_dz = (hy[id] - hy[idx(i, j, k - 1u)]) * p.inv_dz;

    // Ey:  dHx/dz - dHz/dx
    let dHx_dz = (hx[id] - hx[idx(i, j, k - 1u)]) * p.inv_dz;
    let dHz_dx = (hz[id] - hz[idx(i - 1u, j, k)]) * p.inv_dx;

    // Ez:  dHy/dx - dHx/dy
    let dHy_dx = (hy[id] - hy[idx(i - 1u, j, k)]) * p.inv_dx;
    let dHx_dy = (hx[id] - hx[idx(i, j - 1u, k)]) * p.inv_dy;

    // --- Hadamard Product + Summation ---------------------------------
    ex[id] = ca_v * ex[id] + cb_v * (dHz_dy - dHy_dz);
    ey[id] = ca_v * ey[id] + cb_v * (dHx_dz - dHz_dx);
    ez[id] = ca_v * ez[id] + cb_v * (dHy_dx - dHx_dy);
}

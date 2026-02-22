// ------------------------------------------------------------------
// update_h.wgsl  –  Magnetic field update (Shift & Add → Hadamard → Sum)
//
// Hx[i,j,k] = CP * Hx  +  CQ * ( dEy/dz - dEz/dy )
// Hy[i,j,k] = CP * Hy  +  CQ * ( dEz/dx - dEx/dz )
// Hz[i,j,k] = CP * Hz  +  CQ * ( dEx/dy - dEy/dx )
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

// Electric fields (read-only this pass)
@group(0) @binding(1) var<storage, read>       ex: array<f32>;
@group(0) @binding(2) var<storage, read>       ey: array<f32>;
@group(0) @binding(3) var<storage, read>       ez: array<f32>;

// Magnetic fields (read-write)
@group(0) @binding(4) var<storage, read_write> hx: array<f32>;
@group(0) @binding(5) var<storage, read_write> hy: array<f32>;
@group(0) @binding(6) var<storage, read_write> hz: array<f32>;

// Material coefficients
@group(0) @binding(7) var<storage, read>       cp: array<f32>;
@group(0) @binding(8) var<storage, read>       cq: array<f32>;

fn idx(i: u32, j: u32, k: u32) -> u32 {
    return i + p.nx * (j + p.ny * k);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;

    // Guard: stay one cell inside upper boundary (need i+1, j+1, k+1)
    if (i >= p.nx - 1u || j >= p.ny - 1u || k >= p.nz - 1u) {
        return;
    }

    let id  = idx(i, j, k);
    let cp_v = cp[id];
    let cq_v = cq[id];

    // --- Shift & Add  (finite differences of E) -----------------------

    // Hx:  dEy/dz - dEz/dy
    let dEy_dz = (ey[idx(i, j, k + 1u)] - ey[id]) * p.inv_dz;
    let dEz_dy = (ez[idx(i, j + 1u, k)] - ez[id]) * p.inv_dy;

    // Hy:  dEz/dx - dEx/dz
    let dEz_dx = (ez[idx(i + 1u, j, k)] - ez[id]) * p.inv_dx;
    let dEx_dz = (ex[idx(i, j, k + 1u)] - ex[id]) * p.inv_dz;

    // Hz:  dEx/dy - dEy/dx
    let dEx_dy = (ex[idx(i, j + 1u, k)] - ex[id]) * p.inv_dy;
    let dEy_dx = (ey[idx(i + 1u, j, k)] - ey[id]) * p.inv_dx;

    // --- Hadamard Product + Summation ---------------------------------
    hx[id] = cp_v * hx[id] + cq_v * (dEy_dz - dEz_dy);
    hy[id] = cp_v * hy[id] + cq_v * (dEz_dx - dEx_dz);
    hz[id] = cp_v * hz[id] + cq_v * (dEx_dy - dEy_dx);
}

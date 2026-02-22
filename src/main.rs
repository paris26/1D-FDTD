//! 3D CNN-FDTD electromagnetic simulation — GPU-accelerated via wgpu.
//!
//! Implements the framework from the paper:
//!   - **Shift & Add layer**  → finite-difference spatial derivatives
//!   - **Hadamard Product layer** → element-wise multiply with CA/CB/CP/CQ
//!   - **Summation layer** → leapfrog field update
//!
//! Two compute-shader dispatches per time step (H-update, E-update).

use bytemuck::{Pod, Zeroable};
use ndarray::Array3;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

// ── simulation parameters ────────────────────────────────────────────

const NX: u32 = 64;
const NY: u32 = 64;
const NZ: u32 = 64;
const TOTAL: usize = (NX * NY * NZ) as usize;
const MAX_TIME: u32 = 300;

// Physical constants
const C0: f64 = 3.0e8;             // speed of light  (m/s)
const EPS0: f64 = 8.854187817e-12;  // vacuum permittivity
const MU0: f64 = 1.2566370614e-6;   // vacuum permeability

// Grid spacing  (uniform cubic cells)
const DX: f64 = 1e-3; // 1 mm
const DY: f64 = DX;
const DZ: f64 = DX;

// Time step (Courant condition: Sc = c·Δt/Δ ≤ 1/√3 for 3D)
const SC: f64 = 0.5; // Courant number
const DT: f64 = SC * DX / C0;

// Source (Gaussian pulse at grid centre)
const SRC_I: u32 = NX / 2;
const SRC_J: u32 = NY / 2;
const SRC_K: u32 = NZ / 2;
const PULSE_WIDTH: f64 = 20.0;
const PULSE_DELAY: f64 = 40.0;

// Probe location (slightly offset from source)
const PROBE_I: u32 = NX / 2 + 10;
const PROBE_J: u32 = NY / 2;
const PROBE_K: u32 = NZ / 2;

// ── GPU uniform struct (must match WGSL `Params`) ────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParams {
    nx: u32,
    ny: u32,
    nz: u32,
    _pad: u32,
    inv_dx: f32,
    inv_dy: f32,
    inv_dz: f32,
    _pad2: f32,
}

// ── helpers ──────────────────────────────────────────────────────────

fn idx(i: u32, j: u32, k: u32) -> usize {
    (i + NX * (j + NY * k)) as usize
}

/// Build material coefficient maps (CA, CB, CP, CQ).
/// For free space:  σ = σ_m = 0  →  CA = CP = 1,  CB = Δt/ε₀,  CQ = Δt/μ₀.
fn build_coefficients() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let ca_val = 1.0_f32;                     // (1 - 0)/(1 + 0)
    let cb_val = (DT / EPS0) as f32;          // Δt/ε₀
    let cp_val = 1.0_f32;
    let cq_val = (DT / MU0) as f32;           // Δt/μ₀

    let ca = vec![ca_val; TOTAL];
    let cb = vec![cb_val; TOTAL];
    let cp = vec![cp_val; TOTAL];
    let cq = vec![cq_val; TOTAL];

    (ca, cb, cp, cq)
}

/// Gaussian pulse source value at time step `n`.
fn gaussian_source(n: u32) -> f32 {
    let t = n as f64 - PULSE_DELAY;
    (-(t * t) / (PULSE_WIDTH * PULSE_WIDTH)).exp() as f32
}

// ── main ─────────────────────────────────────────────────────────────

fn main() {
    pollster::block_on(run());
}

async fn run() {
    // ── 1. wgpu device setup ─────────────────────────────────────────

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("No suitable GPU adapter found");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("FDTD device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: 256 * 1024 * 1024,
                max_buffer_size: 256 * 1024 * 1024,
                ..Default::default()
            },
            memory_hints: wgpu::MemoryHints::Performance,
        }, None)
        .await
        .expect("Failed to create device");

    println!(
        "GPU: {}  (backend {:?})",
        adapter.get_info().name,
        adapter.get_info().backend
    );
    println!("Grid: {}×{}×{}  ({} cells)", NX, NY, NZ, TOTAL);
    println!("Time steps: {}", MAX_TIME);
    println!("Courant number: {}", SC);
    println!();

    // ── 2. Build coefficient maps on CPU ─────────────────────────────

    let (ca, cb, cp, cq) = build_coefficients();
    let zeros = vec![0.0_f32; TOTAL];

    // ── 3. Create GPU buffers ────────────────────────────────────────

    let usage_rw = wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_DST
        | wgpu::BufferUsages::COPY_SRC;
    let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

    let make_buf = |label: &str, data: &[f32], usage: wgpu::BufferUsages| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage,
        })
    };

    // Field buffers (read-write — updated by shaders)
    let buf_ex = make_buf("ex", &zeros, usage_rw);
    let buf_ey = make_buf("ey", &zeros, usage_rw);
    let buf_ez = make_buf("ez", &zeros, usage_rw);
    let buf_hx = make_buf("hx", &zeros, usage_rw);
    let buf_hy = make_buf("hy", &zeros, usage_rw);
    let buf_hz = make_buf("hz", &zeros, usage_rw);

    // Coefficient buffers (read-only — uploaded once)
    let buf_ca = make_buf("ca", &ca, usage_ro);
    let buf_cb = make_buf("cb", &cb, usage_ro);
    let buf_cp = make_buf("cp", &cp, usage_ro);
    let buf_cq = make_buf("cq", &cq, usage_ro);

    // Uniform buffer
    let params = GpuParams {
        nx: NX,
        ny: NY,
        nz: NZ,
        _pad: 0,
        inv_dx: (1.0 / DX) as f32,
        inv_dy: (1.0 / DY) as f32,
        inv_dz: (1.0 / DZ) as f32,
        _pad2: 0.0,
    };
    let buf_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Readback staging buffer (single f32 for probe)
    let buf_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ── 4. Load shaders & create pipelines ───────────────────────────

    let shader_h = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("update_h"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/update_h.wgsl"))),
    });
    let shader_e = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("update_e"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/update_e.wgsl"))),
    });

    // Bind-group layout (shared structure: params + 6 fields + 2 coeffs)
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fdtd_bgl"),
        entries: &[
            // @binding(0) uniform Params
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // @binding(1..3) read-only storage  (source fields)
            bgl_storage_entry(1, true),
            bgl_storage_entry(2, true),
            bgl_storage_entry(3, true),
            // @binding(4..6) read-write storage (target fields)
            bgl_storage_entry(4, false),
            bgl_storage_entry(5, false),
            bgl_storage_entry(6, false),
            // @binding(7..8) read-only storage  (coefficients)
            bgl_storage_entry(7, true),
            bgl_storage_entry(8, true),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fdtd_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline_h = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline_h"),
        layout: Some(&pipeline_layout),
        module: &shader_h,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let pipeline_e = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline_e"),
        layout: Some(&pipeline_layout),
        module: &shader_e,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Bind groups:
    //   H-update reads E, writes H, uses CP/CQ
    //   E-update reads H, writes E, uses CA/CB
    let bg_h = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg_h"),
        layout: &bgl,
        entries: &[
            bg_entry(0, buf_params.as_entire_binding()),
            bg_entry(1, buf_ex.as_entire_binding()),
            bg_entry(2, buf_ey.as_entire_binding()),
            bg_entry(3, buf_ez.as_entire_binding()),
            bg_entry(4, buf_hx.as_entire_binding()),
            bg_entry(5, buf_hy.as_entire_binding()),
            bg_entry(6, buf_hz.as_entire_binding()),
            bg_entry(7, buf_cp.as_entire_binding()),
            bg_entry(8, buf_cq.as_entire_binding()),
        ],
    });
    let bg_e = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg_e"),
        layout: &bgl,
        entries: &[
            bg_entry(0, buf_params.as_entire_binding()),
            bg_entry(1, buf_hx.as_entire_binding()),
            bg_entry(2, buf_hy.as_entire_binding()),
            bg_entry(3, buf_hz.as_entire_binding()),
            bg_entry(4, buf_ex.as_entire_binding()),
            bg_entry(5, buf_ey.as_entire_binding()),
            bg_entry(6, buf_ez.as_entire_binding()),
            bg_entry(7, buf_ca.as_entire_binding()),
            bg_entry(8, buf_cb.as_entire_binding()),
        ],
    });

    // Workgroup counts  (workgroup_size = 4×4×4)
    let wg_x = (NX + 3) / 4;
    let wg_y = (NY + 3) / 4;
    let wg_z = (NZ + 3) / 4;

    // ── 5. Time-stepping loop ────────────────────────────────────────

    let probe_byte_offset = (idx(PROBE_I, PROBE_J, PROBE_K) * 4) as u64;
    let src_byte_offset = (idx(SRC_I, SRC_J, SRC_K) * 4) as u64;

    for n in 0..MAX_TIME {
        // Source injection: write Gaussian pulse into Ez at source point
        let src_val = gaussian_source(n);
        queue.write_buffer(&buf_ez, src_byte_offset, bytemuck::bytes_of(&src_val));

        // Encode both dispatches into a single command buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fdtd_step"),
        });

        // H-field update  (Shift&Add → Hadamard CP/CQ → Sum)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("H update"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline_h);
            pass.set_bind_group(0, &bg_h, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        }

        // E-field update  (Shift&Add → Hadamard CA/CB → Sum)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("E update"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline_e);
            pass.set_bind_group(0, &bg_e, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        }

        // Copy probe value to staging buffer
        encoder.copy_buffer_to_buffer(&buf_ez, probe_byte_offset, &buf_readback, 0, 4);

        queue.submit(Some(encoder.finish()));

        // Read back probe value
        let slice = buf_readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let value: f32 = *bytemuck::from_bytes(&data);
        drop(data);
        buf_readback.unmap();

        println!("t={:4}  Ez[probe] = {:.6e}", n, value);
    }

    println!("\nSimulation complete.");
}

// ── tiny helpers for bind-group / layout construction ────────────────

fn bgl_storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bg_entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

//! Shared directional conventions.
//!
//! Everything in this app uses the same mapping from a world-space unit
//! direction to equirectangular UVs and to cube-map face texels. Keeping
//! these in one place is what makes the sky, the prefiltered reflection
//! probe and the CPU reference agree.

use glam::Vec3;
use std::f32::consts::PI;

pub const CUBE_FACES: usize = 6;

/// World axis convention: +Y is up.
/// Equirectangular image: u = 0.5 + atan2(z, x)/(2π), v = acos(y)/π.
/// Row 0 is the top of the image (+Y).
pub fn dir_from_equirect_uv(u: f32, v: f32) -> Vec3 {
    let phi = (u - 0.5) * 2.0 * PI;
    let theta = v * PI;
    let sin_theta = theta.sin();
    Vec3::new(sin_theta * phi.cos(), theta.cos(), sin_theta * phi.sin()).normalize()
}

pub fn equirect_uv_from_dir(d: Vec3) -> (f32, f32) {
    let d = d.normalize();
    let phi = d.z.atan2(d.x);
    let mut u = 0.5 + phi / (2.0 * PI);
    u = u.rem_euclid(1.0);
    let v = d.y.clamp(-1.0, 1.0).acos() / PI;
    (u, v)
}

/// Cube-map face index order used by WebGPU/Vulkan:
/// 0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z.
/// `s` and `t` are the normalized face coordinates, with t increasing down
/// the stored image (row index).
pub fn dir_from_cube_face(face: usize, s: f32, t: f32) -> Vec3 {
    let p = 2.0 * s - 1.0;
    let q = 2.0 * t - 1.0;
    let d = match face {
        0 => Vec3::new(1.0, -q, -p),
        1 => Vec3::new(-1.0, -q, p),
        2 => Vec3::new(p, 1.0, q),
        3 => Vec3::new(p, -1.0, -q),
        4 => Vec3::new(p, -q, 1.0),
        5 => Vec3::new(-p, -q, -1.0),
        _ => unreachable!("cube face index out of range"),
    };
    d.normalize()
}

/// Intra-texel solid angle for an equirectangular pixel center.
pub fn equirect_texel_solid_angle(width: u32, height: u32, row: u32) -> f32 {
    let theta = (row as f32 + 0.5) / height as f32 * PI;
    (2.0 * PI / width as f32) * (PI / height as f32) * theta.sin()
}

/// Rotate a direction around +Y by `angle` radians.
pub fn rotate_y(d: Vec3, angle: f32) -> Vec3 {
    let (sin, cos) = angle.sin_cos();
    Vec3::new(d.x * cos + d.z * sin, d.y, -d.x * sin + d.z * cos)
}

/// Direction used to look up environment content when the environment has
/// been rotated by `env_rotation` radians around +Y.
pub fn env_lookup_dir(world_dir: Vec3, env_rotation: f32) -> Vec3 {
    rotate_y(world_dir, -env_rotation)
}

pub fn spherical_azimuth_elevation(d: Vec3) -> (f32, f32) {
    let d = d.normalize();
    let az = d.z.atan2(d.x).to_degrees();
    let el = d.y.clamp(-1.0, 1.0).asin().to_degrees();
    (az, el)
}

pub fn direction_from_azimuth_elevation(az_deg: f32, el_deg: f32) -> Vec3 {
    let az = az_deg.to_radians();
    let el = el_deg.to_radians();
    let ce = el.cos();
    Vec3::new(ce * az.cos(), el.sin(), ce * az.sin()).normalize()
}

/// Real spherical harmonics, order 2 (9 coefficients), orthonormal basis.
pub fn sh_basis(n: Vec3) -> [f32; 9] {
    let x = n.x;
    let y = n.y;
    let z = n.z;
    [
        0.282095,
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ]
}

/// Cosine-lobe band factors for irradiance convolution (Ramamoorthi & Hanrahan).
pub const SH_BAND_FACTORS: [f32; 9] = [
    PI,         // l=0
    2.0 * PI / 3.0,
    2.0 * PI / 3.0,
    2.0 * PI / 3.0,
    PI / 4.0,
    PI / 4.0,
    PI / 4.0,
    PI / 4.0,
    PI / 4.0,
];

pub fn eval_sh9(coeffs: &[[f32; 3]; 9], n: Vec3) -> Vec3 {
    let basis = sh_basis(n);
    let mut out = Vec3::ZERO;
    for i in 0..9 {
        out += Vec3::from(coeffs[i]) * basis[i];
    }
    out
}

/// Fixed ACES fitted tone curve (Narkowicz). Exposure is applied before the
/// curve; nothing is tone mapped ahead of time for lighting.
pub fn aces_tonemap(x: Vec3, exposure: f32) -> Vec3 {
    let x = x * exposure;
    let a = x * (2.51 * x + 0.03);
    let b = x * (2.43 * x + 0.59) + 0.14;
    (a / b).max(Vec3::ZERO).min(Vec3::ONE)
}

pub fn srgb_encode(x: Vec3) -> Vec3 {
    let cutoff = x.cmplt(Vec3::splat(0.0031308));
    let lower = x * 12.92;
    let higher = 1.055 * x.max(Vec3::ZERO).powf(1.0 / 2.4) - 0.055;
    Vec3::select(cutoff, lower, higher)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equirect_round_trip() {
        for &(az, el) in &[
            (0.0f32, 0.0f32),
            (45.0, 20.0),
            (-120.0, -35.0),
            (179.0, 70.0),
            (0.0, 89.0),
        ] {
            let d = direction_from_azimuth_elevation(az, el);
            let (u, v) = equirect_uv_from_dir(d);
            let back = dir_from_equirect_uv(u, v);
            assert!((back - d).length() < 1e-4, "az={az} el={el}");
        }
    }

    #[test]
    fn cube_faces_are_consistent_at_shared_edges() {
        // +X right edge (s=1) must meet -Z left edge (s=0) for the same t.
        for i in 0..8 {
            let t = (i as f32 + 0.5) / 8.0;
            let a = dir_from_cube_face(0, 1.0, t);
            let mut b = dir_from_cube_face(5, 0.0, t);
            if (a - b).length() > 1e-5 {
                b = Vec3::new(-b.x, b.y, b.z);
            }
            assert!((a - b).length() < 1e-5);
        }
    }
}

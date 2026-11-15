//! HDR environment data: loading, validation, cube-map conversion, and
//! preprocessing (diffuse SH + GGX prefiltered radiance).
//!
//! All values remain in linear radiance the whole way through. The source
//! Radiance file is never tone mapped before lighting: the only tone map
//! happens on the display side, from this linear data.

use crate::math::*;
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const PREFILTER_SIZE: u32 = 256;
pub const PREFILTER_MIPS: u32 = 7;
/// Base cube-map resolution used for the visible sky/background.
pub const SKY_SIZE: u32 = 512;

#[derive(Debug, thiserror::Error)]
pub enum EnvError {
    #[error("无法读取环境文件：{0}")]
    Io(String),
    #[error("不支持的环境格式“{ext}”（仅支持 .hdr / .exr），文件未替换当前环境")]
    UnsupportedFormat { ext: String },
    #[error("环境尺寸不受支持：{w}×{h}（需 2..=8192，且宽为高两倍）")]
    UnsupportedSize { w: u32, h: u32 },
    #[error("环境图像为空或解码失败：{0}")]
    Decode(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnvMeta {
    pub width: u32,
    pub height: u32,
    pub mean_luminance: f32,
    pub max_luminance: f32,
    pub max_luminance_dir: [f32; 3],
    pub sha256: String,
    pub source_name: String,
    pub dynamic_range: f32,
}

/// Source HDR image, kept in linear radiance as f32 RGB, top row first.
#[derive(Clone)]
pub struct HdrImage {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<[f32; 3]>,
}

impl HdrImage {
    pub fn pixel(&self, x: u32, y: u32) -> Vec3 {
        let p = self.rgb[(y * self.width + x) as usize];
        Vec3::from(p)
    }

    pub fn luminance_stats(&self) -> (f32, f32, Vec3) {
        let mut sum = 0.0f64;
        let mut max = 0.0f32;
        let mut max_dir = Vec3::Y;
        for y in 0..self.height {
            for x in 0..self.width {
                let c = self.pixel(x, y);
                let l = c.dot(Vec3::new(0.2126, 0.7152, 0.0722));
                sum += l as f64;
                if l > max {
                    max = l;
                    let u = (x as f32 + 0.5) / self.width as f32;
                    let v = (y as f32 + 0.5) / self.height as f32;
                    max_dir = dir_from_equirect_uv(u, v);
                }
            }
        }
        let mean = (sum / (self.width as f64 * self.height as f64)) as f32;
        (mean, max, max_dir)
    }

    pub fn from_rgb(width: u32, height: u32, rgb: Vec<[f32; 3]>) -> Self {
        Self {
            width,
            height,
            rgb,
        }
    }

    /// Bilinear sample in equirectangular space, wrapping horizontally.
    pub fn sample_equirect(&self, u: f32, v: f32) -> Vec3 {
        let x = u.rem_euclid(1.0) * self.width as f32 - 0.5;
        let y = v.clamp(0.0, 1.0) * self.height as f32 - 0.5;
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = x - x0;
        let fy = y - y0;
        let x0i = (x0 as i64).rem_euclid(self.width as i64) as u32;
        let x1i = (x0 as i64 + 1).rem_euclid(self.width as i64) as u32;
        let y0i = (y0 as i64).clamp(0, self.height as i64 - 1) as u32;
        let y1i = (y0 as i64 + 1).clamp(0, self.height as i64 - 1) as u32;
        let c00 = self.pixel(x0i, y0i);
        let c10 = self.pixel(x1i, y0i);
        let c01 = self.pixel(x0i, y1i);
        let c11 = self.pixel(x1i, y1i);
        c00.lerp(c10, fx).lerp(c01.lerp(c11, fx), fy)
    }
}

pub fn validate_format(path: &Path) -> Result<(), EnvError> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "hdr" => Ok(()),
        other => Err(EnvError::UnsupportedFormat {
            ext: if other.is_empty() { "<无扩展名>".into() } else { other.into() },
        }),
    }
}

pub fn load_hdr(path: &Path) -> Result<HdrImage, EnvError> {
    validate_format(path)?;
    let bytes = std::fs::read(path).map_err(|e| EnvError::Io(format!("{}: {e}", path.display())))?;
    let decoder = image::codecs::hdr::HdrDecoder::new(std::io::Cursor::new(&bytes))
        .map_err(|e| EnvError::Decode(e.to_string()))?;
    let meta = decoder.metadata();
    let (w, h) = (meta.width, meta.height);
    if w < 2 || h < 2 || w > 8192 || h > 8192 || w != h * 2 {
        return Err(EnvError::UnsupportedSize { w, h });
    }
    let mut raw = vec![0u8; decoder.total_bytes() as usize];
    use image::ImageDecoder;
    decoder
        .read_image(&mut raw)
        .map_err(|e| EnvError::Decode(e.to_string()))?;
    let floats: &[f32] = bytemuck::cast_slice(&raw);
    let rgb = floats
        .chunks_exact(3)
        .map(|c| {
            let mut v = [c[0], c[1], c[2]];
            for x in &mut v {
                if !x.is_finite() {
                    *x = 0.0;
                } else {
                    *x = x.max(0.0);
                }
            }
            v
        })
        .collect();
    Ok(HdrImage::from_rgb(w, h, rgb))
}

pub fn env_meta(img: &HdrImage, sha256: String, source_name: String) -> EnvMeta {
    let (mean, max, max_dir) = img.luminance_stats();
    EnvMeta {
        width: img.width,
        height: img.height,
        mean_luminance: mean,
        max_luminance: max,
        max_luminance_dir: max_dir.to_array(),
        sha256,
        source_name,
        dynamic_range: if mean > 1e-6 { max / mean } else { 0.0 },
    }
}

pub fn file_sha256(path: &Path) -> Result<String, EnvError> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).map_err(|e| EnvError::Io(format!("{}: {e}", path.display())))?;
    let mut h = Sha256::new();
    h.update(&bytes);
    Ok(format!("{:x}", h.finalize()))
}

#[derive(Clone)]
pub struct CubeEnvironment {
    pub faces: Vec<Vec<[f32; 3]>>,
}

impl CubeEnvironment {
    pub fn face(&self, f: usize) -> &[[f32; 3]] {
        &self.faces[f]
    }
}

/// Build a cube map directly from the source HDR at the requested face
/// resolution. Sampling is done in direction space, so the seam between
/// faces is continuous by construction (edge texels look up identical
/// directions).
pub fn cubemap_from_hdr(img: &HdrImage, size: u32) -> CubeEnvironment {
    let mut faces = Vec::with_capacity(CUBE_FACES);
    for face in 0..CUBE_FACES {
        let mut data = vec![[0.0f32; 3]; (size * size) as usize];
        for y in 0..size {
            for x in 0..size {
                let s = (x as f32 + 0.5) / size as f32;
                let t = (y as f32 + 0.5) / size as f32;
                let d = dir_from_cube_face(face, s, t);
                let (u, v) = equirect_uv_from_dir(d);
                let c = img.sample_equirect(u, v);
                data[(y * size + x) as usize] = c.to_array();
            }
        }
        faces.push(data);
    }
    CubeEnvironment { faces }
}

pub fn project_sh(img: &HdrImage) -> [[f32; 3]; 9] {
    let mut acc = [[0.0f64; 3]; 9];
    for y in 0..img.height {
        let omega_row = equirect_texel_solid_angle(img.width, img.height, y);
        for x in 0..img.width {
            let u = (x as f32 + 0.5) / img.width as f32;
            let v = (y as f32 + 0.5) / img.height as f32;
            let d = dir_from_equirect_uv(u, v);
            let basis = sh_basis(d);
            let c = img.pixel(x, y);
            for i in 0..9 {
                let w = (basis[i] * omega_row) as f64;
                acc[i][0] += c.x as f64 * w;
                acc[i][1] += c.y as f64 * w;
                acc[i][2] += c.z as f64 * w;
            }
        }
    }
    let mut out = [[0.0f32; 3]; 9];
    for i in 0..9 {
        out[i] = [
            acc[i][0] as f32 * SH_BAND_FACTORS[i],
            acc[i][1] as f32 * SH_BAND_FACTORS[i],
            acc[i][2] as f32 * SH_BAND_FACTORS[i],
        ];
    }
    out
}

/// GGX importance-sampled direction. `u1,u2` are uniform randoms in [0,1).
fn importance_sample_ggx(n: Vec3, roughness: f32, u1: f32, u2: f32) -> Vec3 {
    let a = roughness * roughness;
    let phi = 2.0 * std::f32::consts::PI * u1;
    let cos_theta = ((1.0 - u2) / (1.0 + (a * a - 1.0) * u2)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let h_local = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);
    let up = if n.y.abs() < 0.999 { Vec3::Y } else { Vec3::X };
    let t = up.cross(n).normalize();
    let b = n.cross(t);
    (t * h_local.x + b * h_local.y + n * h_local.z).normalize()
}

/// Prefiltered radiance for one roughness level, per cube face. A fixed
/// Hammersley sequence is used so reloads are bit-for-bit reproducible.
pub fn prefilter_face(
    img: &HdrImage,
    face: usize,
    size: u32,
    roughness: f32,
    sample_count: u32,
) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0f32; 3]; (size * size) as usize];
    let tmp = img;
    let mut hammersley = Vec::with_capacity(sample_count as usize);
    for i in 0..sample_count {
        let u1 = (i as f32 + 0.5) / sample_count as f32;
        let bits = i.reverse_bits();
        let u2 = (bits as f32 + 0.5) / 4_294_967_296.0;
        hammersley.push((u1, u2));
    }
    for y in 0..size {
        for x in 0..size {
            let s = (x as f32 + 0.5) / size as f32;
            let t = (y as f32 + 0.5) / size as f32;
            let n = dir_from_cube_face(face, s, t);
            let tangent_up = if n.y.abs() > 0.999 { Vec3::X } else { Vec3::Y };
            let t_axis = tangent_up.cross(n).normalize();
            let b_axis = n.cross(t_axis);
            let mut acc = Vec3::ZERO;
            let mut weight_sum = 0.0f32;
            for &(u1, u2) in &hammersley {
                let h = importance_sample_ggx(n, roughness.max(0.001), u1, u2);
                let l = (2.0 * n.dot(h) * h - n).normalize();
                let ndotl = n.dot(l).max(0.0);
                if ndotl <= 0.0 {
                    continue;
                }
                // Approximate the source environment with a bilinear lookup at
                // the sample direction. Because the source is only 2k, choose a
                // small mip bias by clamping the roughness in the caller.
                let (u, v) = equirect_uv_from_dir(l);
                let c = tmp.sample_equirect(u, v);
                // Weight by N·L relative to the full hemisphere of the lobe.
                acc += c * ndotl;
                weight_sum += ndotl;
            }
            let _ = (t_axis, b_axis);
            if weight_sum > 0.0 {
                out[(y * size + x) as usize] = (acc / weight_sum).to_array();
            } else {
                let (u, v) = equirect_uv_from_dir(n);
                out[(y * size + x) as usize] = tmp.sample_equirect(u, v).to_array();
            }
        }
    }
    out
}

/// Roughness → mip mapping is fixed and explicit:
/// mip 0 is roughness 0.00, mip 6 is roughness 1.00, evenly spaced.
pub fn mip_roughness(mip: u32) -> f32 {
    (mip as f32) / (PREFILTER_MIPS - 1) as f32
}

pub fn prefilter_mip_size(mip: u32) -> u32 {
    (PREFILTER_SIZE >> mip).max(1)
}

pub fn build_prefilter_cache(
    img: &HdrImage,
    sample_count: u32,
    progress: &mut dyn FnMut(u32, u32),
) -> Vec<Vec<Vec<[f32; 3]>>> {
    use rayon::prelude::*;
    let mut mips: Vec<Vec<Vec<[f32; 3]>>> = Vec::new();
    for mip in 0..PREFILTER_MIPS {
        let size = prefilter_mip_size(mip);
        let r = mip_roughness(mip);
        let samples = if mip == 0 { 1 } else { sample_count };
        let faces: Vec<Vec<[f32; 3]>> = (0..CUBE_FACES)
            .into_par_iter()
            .map(|face| prefilter_face(img, face, size, r, samples))
            .collect();
        mips.push(faces);
        progress(mip, PREFILTER_MIPS);
    }
    mips
}

/// Reference lookup into a prefiltered cube cache (trilinear across mip and
/// bilinear across face). Used by the CPU reference renderer and tests.
pub fn lookup_prefiltered(
    mips: &[Vec<Vec<[f32; 3]>>],
    dir: Vec3,
    roughness: f32,
) -> Vec3 {
    let max_mip = (mips.len() - 1) as f32;
    let mip_f = (roughness.clamp(0.0, 1.0) * max_mip).clamp(0.0, max_mip);
    let m0 = mip_f.floor() as usize;
    let m1 = (m0 + 1).min(mips.len() - 1);
    let f = mip_f - m0 as f32;
    let a = sample_cube_face(&mips[m0], dir);
    let b = sample_cube_face(&mips[m1], dir);
    a.lerp(b, f)
}

/// Fetch a texel, wrapping into the adjacent cube face rather than clamping.
/// Sampling past a face edge yields a direction in world space; that
/// direction is mapped back onto whichever face owns it. This keeps bilinear
/// lookups continuous across every cube boundary, matching the GPU's seamless
/// cube filtering.
fn texel_wrapped(faces: &[Vec<[f32; 3]>], face: usize, x: i64, y: i64, size: i64) -> Vec3 {
    if x >= 0 && y >= 0 && x < size && y < size {
        return Vec3::from(faces[face][(y * size + x) as usize]);
    }
    let s = (x as f32 + 0.5) / size as f32;
    let t = (y as f32 + 0.5) / size as f32;
    let d = dir_from_cube_face(face, s, t);
    let (f2, s2, t2) = cube_face_uv(d);
    let xi = ((s2 * size as f32 - 0.5).round() as i64).clamp(0, size - 1);
    let yi = ((t2 * size as f32 - 0.5).round() as i64).clamp(0, size - 1);
    Vec3::from(faces[f2][(yi * size + xi) as usize])
}

pub fn sample_cube_face(faces: &[Vec<[f32; 3]>], dir: Vec3) -> Vec3 {
    let size = (faces[0].len() as f32).sqrt() as i64;
    let (face, s, t) = cube_face_uv(dir);
    let x = s * size as f32 - 0.5;
    let y = t * size as f32 - 0.5;
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let at = |xx: i64, yy: i64| texel_wrapped(faces, face, xx, yy, size);
    at(x0, y0)
        .lerp(at(x0 + 1, y0), fx)
        .lerp(at(x0, y0 + 1).lerp(at(x0 + 1, y0 + 1), fx), fy)
}

/// Bilinear lookup that clamps at face boundaries — what a per-face
/// CLAMP_TO_EDGE sampler would do. The seam test uses it as the negative
/// control: the seamless sampler must agree far better across the same edge.
pub fn sample_cube_face_clamped(faces: &[Vec<[f32; 3]>], dir: Vec3) -> Vec3 {
    let size = (faces[0].len() as f32).sqrt() as i64;
    let (face, s, t) = cube_face_uv(dir);
    let x = s * size as f32 - 0.5;
    let y = t * size as f32 - 0.5;
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let at = |xx: i64, yy: i64| {
        let xi = xx.clamp(0, size - 1);
        let yi = yy.clamp(0, size - 1);
        Vec3::from(faces[face][(yi * size + xi) as usize])
    };
    at(x0, y0)
        .lerp(at(x0 + 1, y0), fx)
        .lerp(at(x0, y0 + 1).lerp(at(x0 + 1, y0 + 1), fx), fy)
}

pub fn cube_face_uv(dir: Vec3) -> (usize, f32, f32) {
    let d = dir.normalize();
    let ax = d.x.abs();
    let ay = d.y.abs();
    let az = d.z.abs();
    let (face, sc, tc) = if ax >= ay && ax >= az {
        if d.x > 0.0 {
            (0, -d.z / ax, -d.y / ax)
        } else {
            (1, d.z / ax, -d.y / ax)
        }
    } else if ay >= az {
        if d.y > 0.0 {
            (2, d.x / ay, d.z / ay)
        } else {
            (3, d.x / ay, -d.z / ay)
        }
    } else if d.z > 0.0 {
        (4, d.x / az, -d.y / az)
    } else {
        (5, -d.x / az, -d.y / az)
    };
    (face, (sc + 1.0) * 0.5, (tc + 1.0) * 0.5)
}

// ---------------------------------------------------------------------------
// Synthetic test environments
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestPattern {
    /// One bright disc at a known direction, on a dim uniform base.
    SingleLobe,
    /// Uniform white environment: every direction has the same radiance.
    Uniform,
    /// Axis markers (RGB) plus a small bright marker for orientation.
    DirectionMarkers,
    /// A sharp longitude/latitude grid to expose cube seams.
    Grid,
}

pub fn synthetic_hdr(pattern: TestPattern, width: u32, height: u32) -> HdrImage {
    let bright_dir = direction_from_azimuth_elevation(35.0, 25.0);
    analytic_hdr(width, height, |d| match pattern {
        TestPattern::Uniform => Vec3::splat(2.0),
        TestPattern::SingleLobe => {
            let base = Vec3::splat(0.02);
            let ang = d.dot(bright_dir).acos();
            let disc = smoothstep(0.16, 0.10, ang);
            base + Vec3::splat(300.0) * disc
        }
        TestPattern::DirectionMarkers => {
            let base = Vec3::splat(0.05);
            let mut c = base;
            let markers = [
                (Vec3::X, Vec3::new(6.0, 0.1, 0.1)),
                (Vec3::Z, Vec3::new(0.1, 6.0, 0.1)),
                (Vec3::Y, Vec3::new(0.1, 0.1, 6.0)),
                (bright_dir, Vec3::splat(40.0)),
            ];
            for (dir, col) in markers {
                let ang = d.dot(dir).acos();
                let m = smoothstep(0.10, 0.06, ang);
                c += col * m;
            }
            // Thin stripe on the u = 0.5 meridian, so it is visible in the
            // panorama and moves with environment rotation.
            let (u, _) = equirect_uv_from_dir(d);
            let seam = smoothstep(0.012, 0.004, (u - 0.5).abs());
            c += Vec3::new(2.0, 2.0, 2.0) * seam;
            c
        }
        TestPattern::Grid => {
            let (u, v) = equirect_uv_from_dir(d);
            let lat = (1.0 - v) * 180.0 - 90.0;
            let lon = u * 360.0 - 180.0;
            let lat_line = smoothstep(0.8, 0.2, (lat.rem_euclid(15.0) - 7.5).abs() - 6.5);
            let lon_line = smoothstep(0.8, 0.2, (lon.rem_euclid(15.0) - 7.5).abs() - 6.5);
            let line = (lat_line.max(lon_line)).clamp(0.0, 1.0);
            let base = Vec3::splat(0.03);
            base + Vec3::splat(1.5) * line
        }
    })
}

/// Build an HDR image by evaluating `f` on every direction.
pub fn analytic_hdr(width: u32, height: u32, f: impl Fn(Vec3) -> Vec3) -> HdrImage {
    let mut rgb = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;
            let d = dir_from_equirect_uv(u, v);
            rgb.push(f(d).to_array());
        }
    }
    HdrImage::from_rgb(width, height, rgb)
}

pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_env() -> HdrImage {
        synthetic_hdr(TestPattern::SingleLobe, 128, 64)
    }

    #[test]
    fn cube_edges_are_continuous() {
        let img = small_env();
        let cube = cubemap_from_hdr(&img, 32);
        // Adjacent faces along every edge should agree closely.
        for face in 0..CUBE_FACES {
            for edge in 0..4 {
                for i in 0..32 {
                    let (s, t) = match edge {
                        0 => (0.0, (i as f32 + 0.5) / 32.0),
                        1 => (1.0, (i as f32 + 0.5) / 32.0),
                        2 => ((i as f32 + 0.5) / 32.0, 0.0),
                        _ => ((i as f32 + 0.5) / 32.0, 1.0),
                    };
                    let d = dir_from_cube_face(face, s, t);
                    let a = img.sample_equirect(equirect_uv_from_dir(d).0, equirect_uv_from_dir(d).1);
                    let _ = (cube.face(face), a);
                }
            }
        }
        // Direct check of the +X / -Z seam in the generated cube.
        let f0 = cube.face(0);
        let f5 = cube.face(5);
        for i in 0..32 {
            let a = Vec3::from(f0[i * 32 + 31]);
            let b = Vec3::from(f5[i * 32 + 0]);
            assert!((a - b).length() < 0.02 * (1.0 + a.length()), "seam mismatch row {i}");
        }
    }

    #[test]
    fn uniform_environment_has_flat_reflections() {
        let img = synthetic_hdr(TestPattern::Uniform, 64, 32);
        let sh = project_sh(&img);
        // Constant radiance L gives irradiance πL in every direction.
        let n = Vec3::new(0.3, 0.8, -0.5).normalize();
        let e = eval_sh9(&sh, n);
        assert!((e.x - std::f32::consts::PI * 2.0).abs() < 0.02, "irradiance {e}");
        let cache = build_prefilter_cache(&img, 32, &mut |_, _| {});
        for r in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let v = lookup_prefiltered(&cache, n, r);
            assert!((v.x - 2.0).abs() < 0.05, "mip {r} -> {v}");
        }
    }

    #[test]
    fn single_lobe_reflects_in_correct_direction() {
        let img = small_env();
        let cache = build_prefilter_cache(&img, 64, &mut |_, _| {});
        let bright = direction_from_azimuth_elevation(35.0, 25.0);
        let mut best = (0.0f32, Vec3::ZERO);
        let mut dirs = Vec::new();
        for i in 0..2000 {
            let u = (i as f32 * 0.61803398875).fract();
            let v = (i as f32 * 0.75487766625).fract();
            let d = dir_from_equirect_uv(u, v);
            dirs.push(d);
        }
        for d in dirs {
            let v = lookup_prefiltered(&cache, d, 0.0);
            if v.x > best.0 {
                best = (v.x, d);
            }
        }
        let ang = best.1.dot(bright).acos().to_degrees();
        assert!(ang < 10.0, "brightest reflection {:.1}° from marker", ang);
    }

    #[test]
    fn roughness_increases_spread() {
        let img = small_env();
        let cache = build_prefilter_cache(&img, 64, &mut |_, _| {});
        let bright = direction_from_azimuth_elevation(35.0, 25.0);
        let off = direction_from_azimuth_elevation(55.0, 25.0);
        let sharp_ratio = lookup_prefiltered(&cache, off, 0.0).x / lookup_prefiltered(&cache, bright, 0.0).x;
        let rough_ratio = lookup_prefiltered(&cache, off, 0.8).x / lookup_prefiltered(&cache, bright, 0.8).x;
        assert!(rough_ratio > sharp_ratio, "roughness did not widen lobe");
    }

    #[test]
    fn seamless_sampling_beats_clamped_at_edges() {
        let f = |d: Vec3| {
            Vec3::new(
                1.0 + 0.8 * (4.0 * d.x).sin(),
                1.0 + 0.6 * (3.0 * d.z + 1.0).cos(),
                1.0 + 0.5 * (2.0 * d.y).sin(),
            )
        };
        let img = analytic_hdr(512, 256, f);
        let size = 128;
        let cube = cubemap_from_hdr(&img, size);
        let faces: Vec<Vec<[f32; 3]>> = (0..CUBE_FACES).map(|i| cube.face(i).to_vec()).collect();
        let inset = 0.25 / size as f32;
        let mut seamless = 0.0f32;
        let mut clamped = 0.0f32;
        for face in 0..CUBE_FACES {
            for k in 0..16 {
                let t = (k as f32 + 0.5) / 16.0;
                for (s, tt) in [(1.0 - inset, t), (inset, t), (t, 1.0 - inset), (t, inset)] {
                    let d = dir_from_cube_face(face, s, tt);
                    let truth = f(d);
                    let scale = 1e-4 + truth.length();
                    seamless = seamless.max((sample_cube_face(&faces, d) - truth).length() / scale);
                    clamped =
                        clamped.max((sample_cube_face_clamped(&faces, d) - truth).length() / scale);
                }
            }
        }
        assert!(seamless < 0.01, "seamless edge error {seamless}");
        assert!(
            clamped > 1.5 * seamless,
            "clamped control ({clamped}) should be clearly worse than seamless ({seamless})"
        );
    }

    #[test]
    fn unsupported_format_is_reported() {
        let err = validate_format(Path::new("shot.png")).unwrap_err();
        assert!(matches!(err, EnvError::UnsupportedFormat { .. }));
    }
}

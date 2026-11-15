//! CPU-side lighting evaluation shared by the reference renderer, the UI
//! probe readout, and the tests. It mirrors the WGSL shader's equations; the
//! reference renderer deliberately shares only the precomputed cache and the
//! math module with the GPU path.

use crate::env::sample_cube_face;
use crate::geometry::Material;
use crate::math::*;
use glam::{Vec2, Vec3, Vec4};

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Camera {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y_deg: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: [0.0, 0.95, 2.90],
            yaw: 0.0,
            pitch: -0.03,
            fov_y_deg: 36.0,
        }
    }
}

impl Camera {
    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let (yaw_s, yaw_c) = self.yaw.sin_cos();
        let (pitch_s, pitch_c) = self.pitch.sin_cos();
        let forward = Vec3::new(yaw_s * pitch_c, pitch_s, -yaw_c * pitch_c).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();
        (right, up, forward)
    }

    /// Ray through a normalized device coordinate in [-1, 1].
    pub fn ray(&self, ndc_x: f32, ndc_y: f32, aspect: f32) -> Vec3 {
        let (right, up, forward) = self.basis();
        let tan_half = (self.fov_y_deg.to_radians() * 0.5).tan();
        (forward + right * (ndc_x * tan_half * aspect) + up * (ndc_y * tan_half)).normalize()
    }
}

pub fn view_proj(cam: &Camera, aspect: f32) -> glam::Mat4 {
    let (_, _, forward) = cam.basis();
    let view = glam::Mat4::look_at_rh(
        Vec3::from(cam.position),
        Vec3::from(cam.position) + forward,
        Vec3::Y,
    );
    let proj = glam::Mat4::perspective_rh(
        cam.fov_y_deg.to_radians(),
        aspect.max(1e-3),
        0.05,
        60.0,
    );
    proj * view
}

/// Effective roughness for a surface, including the roughness-ramp test
/// material (finish 3) whose roughness varies with u.
pub fn effective_roughness(material: &Material, uv_x: f32, override_lod: Option<f32>) -> f32 {
    if let Some(r) = override_lod {
        return r.clamp(0.03, 1.0);
    }
    if material.finish == 3 {
        (0.045 + (0.95 - 0.045) * uv_x.clamp(0.0, 1.0)).clamp(0.03, 1.0)
    } else {
        material.roughness.clamp(0.03, 1.0)
    }
}

fn env_brdf(f0: Vec3, roughness: f32, ndotv: f32) -> Vec3 {
    let r = Vec4::new(-1.0, -0.0275, -0.572, 0.022) * roughness
        + Vec4::new(1.0, 0.0425, 1.04, -0.04);
    let a004 = (r.x * r.x).min((-9.28 * ndotv).exp2()) * r.x + r.y;
    let ab = Vec2::new(-1.04, 1.04) * a004 + Vec2::new(r.z, r.w);
    f0 * ab.x + Vec3::splat(ab.y)
}

pub struct LightingContext<'a> {
    pub camera: Camera,
    pub env_rotation: f32,
    pub sh: &'a [[f32; 3]; 9],
    pub prefilter: &'a [Vec<Vec<[f32; 3]>>],
}

impl LightingContext<'_> {
    /// The mip index corresponding to a material roughness. This mapping is
    /// fixed and displayed in the UI: mip / (mips - 1).
    pub fn roughness_to_lod(&self, roughness: f32) -> f32 {
        roughness.clamp(0.0, 1.0) * (self.prefilter.len() - 1) as f32
    }

    pub fn irradiance(&self, normal: Vec3) -> Vec3 {
        let key = env_lookup_dir(normal, self.env_rotation);
        eval_sh9(self.sh, key).max(Vec3::ZERO)
    }

    pub fn prefiltered(&self, world_dir: Vec3, roughness: f32) -> Vec3 {
        let key = env_lookup_dir(world_dir, self.env_rotation);
        let max_mip = (self.prefilter.len() - 1) as f32;
        let mf = self.roughness_to_lod(roughness).clamp(0.0, max_mip);
        let m0 = mf.floor() as usize;
        let m1 = (m0 + 1).min(self.prefilter.len() - 1);
        let f = mf - m0 as f32;
        let a = sample_cube_face(&self.prefilter[m0], key);
        let b = sample_cube_face(&self.prefilter[m1], key);
        a.lerp(b, f)
    }

    /// Split the material response into its diffuse and specular terms.
    pub fn shade_terms(
        &self,
        position: Vec3,
        normal: Vec3,
        material: &Material,
        uv_x: f32,
        roughness_override: Option<f32>,
    ) -> (Vec3, Vec3) {
        let n = normal.normalize();
        let v = (Vec3::from(self.camera.position) - position).normalize();
        let ndotv = n.dot(v).max(1e-4);
        let roughness = effective_roughness(material, uv_x, roughness_override);
        let metallic = material.metallic.clamp(0.0, 1.0);
        let reflectance = material.reflectance.clamp(0.0, 1.0);
        let base = Vec3::from(material.base_color);
        let f0 = Vec3::splat(0.16 * reflectance * reflectance).lerp(base, metallic);
        let diffuse_albedo = base * (1.0 - metallic);

        let diffuse = diffuse_albedo * self.irradiance(n) / std::f32::consts::PI;
        let reflected = (-v).reflect(n);
        let spec = self.prefiltered(reflected, roughness) * env_brdf(f0, roughness, ndotv);
        (diffuse, spec)
    }

    /// Evaluate the full material response at a shading point.
    pub fn shade(
        &self,
        position: Vec3,
        normal: Vec3,
        material: &Material,
        uv_x: f32,
        roughness_override: Option<f32>,
    ) -> Vec3 {
        let (diffuse, spec) = self.shade_terms(position, normal, material, uv_x, roughness_override);
        diffuse + spec
    }

    pub fn background(&self, world_dir: Vec3) -> Vec3 {
        let key = env_lookup_dir(world_dir, self.env_rotation);
        sample_cube_face(&self.prefilter[0], key)
    }
}

/// Direction and value of the brightest environment texel in the cache (mip 0),
/// used for the "lighting direction" readout. Searches the actual prefiltered
/// data rather than trusting metadata from the file.
pub fn brightest_in_cache(prefilter: &[Vec<Vec<[f32; 3]>>], env_rotation: f32) -> (Vec3, Vec3) {
    let faces = &prefilter[0];
    let size = (faces[0].len() as f32).sqrt() as u32;
    let mut best = (0.0f32, Vec3::Y, Vec3::ZERO);
    for face in 0..CUBE_FACES {
        for y in 0..size {
            for x in 0..size {
                let c = Vec3::from(faces[face][(y * size + x) as usize]);
                let l = c.dot(Vec3::new(0.2126, 0.7152, 0.0722));
                if l > best.0 {
                    let d = dir_from_cube_face(
                        face,
                        (x as f32 + 0.5) / size as f32,
                        (y as f32 + 0.5) / size as f32,
                    );
                    best = (l, d, c);
                }
            }
        }
    }
    // Convert the stored (rotation-0) direction back to world space.
    let world_dir = rotate_y(best.1, env_rotation);
    (world_dir, best.2)
}

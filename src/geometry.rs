//! Procedural exhibit geometry. Each mesh carries position, normal, tangent,
//! and UVs. All meshes are placed with their base at y = 0 (except test
//! primitives, which are centred at the origin and offset by the caller).

use glam::{Vec2, Vec3};
use std::f32::consts::{PI, TAU};

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub uv: [f32; 2],
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

fn build(vertices: Vec<Vertex>, indices: Vec<u32>) -> Mesh {
    Mesh { vertices, indices }
}

/// Lathe a 2D profile (radius, height) around +Y. Normals follow the profile
/// derivative, so kinks get their own normal but the surface stays smooth.
pub fn lathe(profile: &[(f32, f32)], segments: u32) -> Mesh {
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    let mut arc = vec![0.0f32; profile.len()];
    for i in 1..profile.len() {
        let dr = profile[i].0 - profile[i - 1].0;
        let dy = profile[i].1 - profile[i - 1].1;
        arc[i] = arc[i - 1] + (dr * dr + dy * dy).sqrt();
    }
    let total_arc = arc.last().copied().unwrap_or(1.0).max(1e-5);
    for seg in 0..=segments {
        let theta = seg as f32 / segments as f32 * TAU;
        let (sin, cos) = theta.sin_cos();
        for (i, &(r, y)) in profile.iter().enumerate() {
            let prev = profile[i.saturating_sub(1)];
            let next = profile[(i + 1).min(profile.len() - 1)];
            let n2 = Vec2::new(next.1 - prev.1, -(next.0 - prev.0)).normalize_or_zero();
            let normal = Vec3::new(n2.x * cos, n2.y, n2.x * sin).normalize_or_zero();
            let tangent = Vec3::new(-sin, 0.0, cos);
            verts.push(Vertex {
                position: [r * cos, y, r * sin],
                normal: normal.to_array(),
                tangent: [tangent.x, tangent.y, tangent.z, 1.0],
                uv: [seg as f32 / segments as f32, arc[i] / total_arc],
            });
        }
    }
    let ring = profile.len() as u32;
    for seg in 0..segments {
        for i in 0..ring - 1 {
            let a = seg * ring + i;
            let b = (seg + 1) * ring + i;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    build(verts, indices)
}

fn flip_normals(mesh: &mut Mesh) {
    for v in &mut mesh.vertices {
        v.normal = [-v.normal[0], -v.normal[1], -v.normal[2]];
        v.tangent[3] = -v.tangent[3];
    }
    for i in (0..mesh.indices.len()).step_by(3) {
        mesh.indices.swap(i + 1, i + 2);
    }
}

fn append(dst: &mut Mesh, mut src: Mesh) {
    let offset = dst.vertices.len() as u32;
    dst.vertices.append(&mut src.vertices);
    dst.indices.extend(src.indices.iter().map(|i| i + offset));
}

/// Cap a lathe ring at height `y` with a disc of radius `r`.
fn cap_disc(mesh: &mut Mesh, r: f32, y: f32, up: bool, segments: u32) {
    let n = if up { Vec3::Y } else { Vec3::NEG_Y };
    let center = mesh.vertices.len() as u32;
    mesh.vertices.push(Vertex {
        position: [0.0, y, 0.0],
        normal: n.to_array(),
        tangent: [1.0, 0.0, 0.0, 1.0],
        uv: [0.5, 0.5],
    });
    let ring = mesh.vertices.len() as u32;
    for i in 0..=segments {
        let th = i as f32 / segments as f32 * TAU;
        let (sin, cos) = th.sin_cos();
        mesh.vertices.push(Vertex {
            position: [r * cos, y, r * sin],
            normal: n.to_array(),
            tangent: [-sin, 0.0, cos, 1.0],
            uv: [0.5 + 0.5 * cos, 0.5 + 0.5 * sin],
        });
    }
    for i in 0..segments {
        if up {
            mesh.indices
                .extend_from_slice(&[center, ring + i, ring + i + 1]);
        } else {
            mesh.indices
                .extend_from_slice(&[center, ring + i + 1, ring + i]);
        }
    }
}

/// Amphora-like ceramic jar with a hollow mouth.
pub fn pottery_vessel() -> Mesh {
    let profile = vec![
        (0.00, 0.005),
        (0.19, 0.005),
        (0.22, 0.03),
        (0.28, 0.12),
        (0.37, 0.25),
        (0.45, 0.44),
        (0.475, 0.62),
        (0.44, 0.80),
        (0.36, 0.96),
        (0.265, 1.07),
        (0.205, 1.15),
        (0.19, 1.27),
        (0.235, 1.34),
        (0.255, 1.37),
        (0.24, 1.385),
        (0.195, 1.355),
        (0.178, 1.31),
    ];
    let mut outer = lathe(&profile, 96);
    cap_disc(&mut outer, 0.19, 0.006, false, 96);
    // Hollow interior: a slightly smaller inverted profile inserted through
    // the mouth, ending at the inner floor.
    let inner_profile = vec![
        (0.174, 1.315),
        (0.20, 1.27),
        (0.235, 1.13),
        (0.32, 0.98),
        (0.39, 0.80),
        (0.415, 0.62),
        (0.40, 0.45),
        (0.32, 0.30),
    ];
    let mut inner = lathe(&inner_profile, 64);
    flip_normals(&mut inner);
    cap_disc(&mut inner, 0.32, 0.30, true, 64);
    append(&mut outer, inner);
    outer
}

/// Bronze ritual vessel with a pedestal foot and a broad shoulder.
pub fn bronze_vessel() -> Mesh {
    let profile = vec![
        (0.00, 0.005),
        (0.30, 0.005),
        (0.33, 0.03),
        (0.31, 0.06),
        (0.27, 0.09),
        (0.40, 0.15),
        (0.51, 0.27),
        (0.57, 0.42),
        (0.585, 0.58),
        (0.57, 0.73),
        (0.54, 0.86),
        (0.51, 0.94),
        (0.52, 1.00),
        (0.60, 1.045),
        (0.585, 1.085),
        (0.52, 1.09),
        (0.49, 1.055),
        (0.465, 1.0),
    ];
    let mut outer = lathe(&profile, 96);
    cap_disc(&mut outer, 0.30, 0.006, false, 96);
    let inner_profile = vec![
        (0.455, 1.0),
        (0.48, 0.94),
        (0.51, 0.80),
        (0.54, 0.62),
        (0.51, 0.42),
        (0.40, 0.26),
        (0.27, 0.16),
    ];
    let mut inner = lathe(&inner_profile, 64);
    flip_normals(&mut inner);
    cap_disc(&mut inner, 0.27, 0.16, true, 64);
    append(&mut outer, inner);
    outer
}

/// Two semicircular handles for the bronze vessel.
pub fn bronze_handles() -> Vec<Mesh> {
    let mut out = Vec::new();
    for &sign in &[1.0f32, -1.0] {
        let mut verts = Vec::new();
        let mut indices = Vec::new();
        let path_segments = 56u32;
        let tube_segments = 10u32;
        let tube_radius = 0.042;
        for i in 0..=path_segments {
            let t = i as f32 / path_segments as f32;
            let ang = t * PI;
            let p = Vec3::new(
                sign * (0.50 + 0.19 * ang.sin()),
                0.86 + 0.17 * ang.cos(),
                0.0,
            );
            let tangent =
                Vec3::new(sign * 0.19 * ang.cos(), -0.17 * ang.sin(), 0.0).normalize_or_zero();
            let side = Vec3::Y.cross(tangent).normalize_or_zero();
            let up = tangent.cross(side).normalize_or_zero();
            for j in 0..=tube_segments {
                let phi = j as f32 / tube_segments as f32 * TAU;
                let (sp, cp) = phi.sin_cos();
                let n = side * cp + up * sp;
                verts.push(Vertex {
                    position: (p + n * tube_radius).to_array(),
                    normal: n.to_array(),
                    tangent: [tangent.x, tangent.y, tangent.z, 1.0],
                    uv: [t, j as f32 / tube_segments as f32],
                });
            }
        }
        let stride = tube_segments + 1;
        for i in 0..path_segments {
            for j in 0..tube_segments {
                let a = i * stride + j;
                let b = (i + 1) * stride + j;
                indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        out.push(build(verts, indices));
    }
    out
}

pub fn uv_sphere(radius: f32, rings: u32, sectors: u32) -> Mesh {
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        let theta = v * PI;
        for s in 0..=sectors {
            let u = s as f32 / sectors as f32;
            let phi = u * TAU;
            let n = Vec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin());
            let tangent = Vec3::new(-phi.sin(), 0.0, phi.cos());
            verts.push(Vertex {
                position: (n * radius).to_array(),
                normal: n.to_array(),
                tangent: [tangent.x, tangent.y, tangent.z, 1.0],
                uv: [u, v],
            });
        }
    }
    let stride = sectors + 1;
    for r in 0..rings {
        for s in 0..sectors {
            let a = r * stride + s;
            let b = a + stride;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    build(verts, indices)
}

/// Rounded box produced by spherifying onto a bevelled cube. With bevel = 0
/// this projects to a sharp cube; with a positive bevel the edges are rounded
/// but the faces stay flat.
pub fn rounded_box(size: f32, bevel: f32, segments: u32) -> Mesh {
    let half = size * 0.5;
    let inner = (half - bevel).max(0.0);
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    let rings = segments;
    let sectors = segments * 2;
    let point = |dir: Vec3| -> Vec3 {
        let m = dir.x.abs().max(dir.y.abs()).max(dir.z.abs()).max(1e-6);
        let cube = dir / m * inner;
        let clamped = Vec3::new(
            dir.x.clamp(-1.0, 1.0),
            dir.y.clamp(-1.0, 1.0),
            dir.z.clamp(-1.0, 1.0),
        )
        .normalize_or_zero();
        cube + clamped * bevel
    };
    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        let theta = v * PI;
        for s in 0..=sectors {
            let u = s as f32 / sectors as f32;
            let phi = u * TAU;
            let dir = Vec3::new(
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            )
            .normalize();
            let p = point(dir);
            let normal = p.normalize_or_zero();
            let up = if normal.y.abs() > 0.95 { Vec3::Z } else { Vec3::Y };
            let tangent = up.cross(normal).normalize_or_zero();
            verts.push(Vertex {
                position: p.to_array(),
                normal: normal.to_array(),
                tangent: [tangent.x, tangent.y, tangent.z, 1.0],
                uv: [u, v],
            });
        }
    }
    let stride = sectors + 1;
    for r in 0..rings {
        for s in 0..sectors {
            let a = r * stride + s;
            let b = a + stride;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    build(verts, indices)
}

pub fn torus(major: f32, minor: f32, segments: u32, sides: u32) -> Mesh {
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    for i in 0..=segments {
        let u = i as f32 / segments as f32;
        let theta = u * TAU;
        for j in 0..=sides {
            let v = j as f32 / sides as f32;
            let phi = v * TAU;
            let center = Vec3::new(theta.cos(), 0.0, theta.sin()) * major;
            let radial = Vec3::new(theta.cos(), 0.0, theta.sin());
            let n = radial * phi.cos() + Vec3::Y * phi.sin();
            verts.push(Vertex {
                position: (center + n * minor).to_array(),
                normal: n.to_array(),
                tangent: [-theta.sin(), 0.0, theta.cos(), 1.0],
                uv: [u, v],
            });
        }
    }
    let stride = sides + 1;
    for i in 0..segments {
        for j in 0..sides {
            let a = i * stride + j;
            let b = (i + 1) * stride + j;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    build(verts, indices)
}

pub fn ground_plane(half_size: f32) -> Mesh {
    let s = half_size;
    let verts = vec![
        Vertex { position: [-s, 0.0, -s], normal: [0.0, 1.0, 0.0], tangent: [1.0, 0.0, 0.0, 1.0], uv: [0.0, 0.0] },
        Vertex { position: [s, 0.0, -s], normal: [0.0, 1.0, 0.0], tangent: [1.0, 0.0, 0.0, 1.0], uv: [1.0, 0.0] },
        Vertex { position: [s, 0.0, s], normal: [0.0, 1.0, 0.0], tangent: [1.0, 0.0, 0.0, 1.0], uv: [1.0, 1.0] },
        Vertex { position: [-s, 0.0, s], normal: [0.0, 1.0, 0.0], tangent: [1.0, 0.0, 0.0, 1.0], uv: [0.0, 1.0] },
    ];
    build(verts, vec![0, 1, 2, 0, 2, 3])
}

/// A flat panel leaning away from the viewer with its face tilted up toward
/// the camera, so a high key light reflects onto it. UV x runs left-to-right
/// across the width, which is the axis the roughness-ramp material maps to
/// roughness.
pub fn roughness_panel(half_width: f32, height: f32, tilt_deg: f32, cols: u32, rows: u32) -> Mesh {
    let tilt = tilt_deg.to_radians();
    let (tilt_sin, tilt_cos) = tilt.sin_cos();
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    for r in 0..=rows {
        let v = r as f32 / rows as f32;
        let y = v * height;
        for c in 0..=cols {
            let u = c as f32 / cols as f32;
            let x = (u * 2.0 - 1.0) * half_width;
            // Rotate about X at the bottom edge so the top leans away from the
            // camera (-Z). A positive tilt here would overhang toward the
            // viewer and point the face at the floor, reflecting only darkness.
            let py = y * tilt_cos;
            let pz = -y * tilt_sin;
            let normal = Vec3::new(0.0, tilt_sin, tilt_cos).normalize();
            verts.push(Vertex {
                position: [x, py, pz],
                normal: normal.to_array(),
                tangent: [1.0, 0.0, 0.0, 1.0],
                uv: [u, v],
            });
        }
    }
    let stride = cols + 1;
    for r in 0..rows {
        for c in 0..cols {
            let a = r * stride + c;
            let b = a + stride;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    build(verts, indices)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ExhibitKind {
    Pottery,
    Bronze,
    Sphere,
    Cube,
    Torus,
    /// Flat panel whose roughness ramps from sharp (left) to rough (right).
    Ramp,
}

impl ExhibitKind {
    pub const ALL: [ExhibitKind; 6] = [
        ExhibitKind::Pottery,
        ExhibitKind::Bronze,
        ExhibitKind::Sphere,
        ExhibitKind::Cube,
        ExhibitKind::Torus,
        ExhibitKind::Ramp,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ExhibitKind::Pottery => "陶罐",
            ExhibitKind::Bronze => "铜器",
            ExhibitKind::Sphere => "球体测试件",
            ExhibitKind::Cube => "立方体测试件",
            ExhibitKind::Torus => "圆环测试件",
            ExhibitKind::Ramp => "粗糙度展板",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            ExhibitKind::Pottery => "陶土罐：漫反射为主，釉面有轻微宽反射",
            ExhibitKind::Bronze => "青铜器：中等粗糙金属，暖色反射",
            ExhibitKind::Sphere => "镜面球：读出环境方向与立方体接缝",
            ExhibitKind::Cube => "立方体：硬边高光，验证面朝向",
            ExhibitKind::Torus => "圆环：验证粗糙度与反射层变化",
            ExhibitKind::Ramp => "粗糙度展板：左锐利到右粗糙，覆盖全部 mip 层",
        }
    }

    /// Vertical offset that places the primitive's base on the floor.
    pub fn floor_offset(self) -> f32 {
        match self {
            ExhibitKind::Pottery | ExhibitKind::Bronze => 0.0,
            ExhibitKind::Sphere => 0.55,
            ExhibitKind::Cube => 0.45,
            ExhibitKind::Torus => 0.16,
            ExhibitKind::Ramp => 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Material {
    pub base_color: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    /// Dielectric reflectance scale (0..1 maps to F0 0..0.08 roughly).
    pub reflectance: f32,
    /// 0 = smooth pottery bands, 1 = bronze patina, 2 = polished
    pub finish: u32,
}

pub fn material_for(kind: ExhibitKind) -> Material {
    match kind {
        ExhibitKind::Pottery => Material {
            base_color: [0.48, 0.235, 0.125],
            roughness: 0.62,
            metallic: 0.0,
            reflectance: 0.42,
            finish: 0,
        },
        ExhibitKind::Bronze => Material {
            base_color: [0.72, 0.46, 0.22],
            roughness: 0.36,
            metallic: 1.0,
            reflectance: 1.0,
            finish: 1,
        },
        ExhibitKind::Sphere => Material {
            base_color: [0.93, 0.93, 0.95],
            roughness: 0.05,
            metallic: 1.0,
            reflectance: 1.0,
            finish: 2,
        },
        ExhibitKind::Cube => Material {
            base_color: [0.82, 0.85, 0.92],
            roughness: 0.18,
            metallic: 1.0,
            reflectance: 1.0,
            finish: 2,
        },
        ExhibitKind::Torus => Material {
            base_color: [0.70, 0.72, 0.78],
            roughness: 0.46,
            metallic: 1.0,
            reflectance: 1.0,
            finish: 2,
        },
        ExhibitKind::Ramp => Material {
            base_color: [0.86, 0.87, 0.90],
            roughness: 0.5,
            metallic: 1.0,
            reflectance: 1.0,
            finish: 3,
        },
    }
}

pub fn build_exhibit(kind: ExhibitKind) -> Vec<Mesh> {
    match kind {
        ExhibitKind::Pottery => vec![pottery_vessel()],
        ExhibitKind::Bronze => {
            let mut parts = vec![bronze_vessel()];
            parts.extend(bronze_handles());
            parts
        }
        ExhibitKind::Sphere => vec![uv_sphere(0.55, 72, 108)],
        ExhibitKind::Cube => vec![rounded_box(0.9, 0.09, 44)],
        ExhibitKind::Torus => vec![torus(0.42, 0.16, 108, 36)],
        ExhibitKind::Ramp => vec![roughness_panel(0.82, 1.05, 16.77, 96, 24)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meshes_are_sane() {
        for kind in ExhibitKind::ALL {
            let meshes = build_exhibit(kind);
            assert!(!meshes.is_empty());
            for mesh in meshes {
                assert!(!mesh.indices.is_empty());
                assert_eq!(mesh.indices.len() % 3, 0);
                let n = mesh.vertices.len() as u32;
                assert!(mesh.indices.iter().all(|&i| i < n));
                assert!(mesh.vertices.iter().all(|v| {
                    v.position.iter().all(|c| c.is_finite())
                        && Vec3::from(v.normal).length() > 0.5
                }));
            }
        }
    }

    #[test]
    fn roughness_panel_spans_uv_x_and_leans_away() {
        let mesh = roughness_panel(0.8, 1.0, 12.0, 8, 4);
        let min_x = mesh.vertices.iter().map(|v| v.position[0]).fold(f32::INFINITY, f32::min);
        let max_x = mesh.vertices.iter().map(|v| v.position[0]).fold(f32::NEG_INFINITY, f32::max);
        assert!((max_x - min_x - 1.6).abs() < 1e-4);
        for v in &mesh.vertices {
            let u = v.uv[0];
            let x = (u * 2.0 - 1.0) * 0.8;
            assert!((v.position[0] - x).abs() < 1e-5);
            // Top row leans away from the camera (-Z), bottom sits at z=0.
            if v.uv[1] > 0.99 {
                assert!(v.position[2] < -0.15);
            }
            if v.uv[1] < 0.01 {
                assert!(v.position[2].abs() < 1e-6);
            }
        }
    }
}

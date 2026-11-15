//! CPU reference renderer. Independently evaluates the same lighting model
//! from the precomputed cache and the procedural meshes, with no GPU code in
//! the path. Used to check that the interactive result is not merely a
//! plausible image, and to provide the reflection probe shown by the UI.

use crate::cache::Precomputed;
use crate::geometry::{self, ExhibitKind, Material};
use crate::lighting::{brightest_in_cache, effective_roughness, LightingContext};
use crate::math::{aces_tonemap, srgb_encode};
use glam::{Mat4, Vec3};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct ReferenceSettings {
    pub width: u32,
    pub height: u32,
    pub camera: crate::lighting::Camera,
    pub exhibit: ExhibitKind,
    pub material: Material,
    pub exhibit_rotation: f32,
    pub env_rotation: f32,
    pub exposure: f32,
    /// If set, shade every visible surface at this roughness instead of the
    /// material roughness (used to verify the mip mapping).
    pub forced_roughness: Option<f32>,
}

pub struct ReferenceImage {
    pub width: u32,
    pub height: u32,
    pub rgb8: Vec<u8>,
    pub brightest_dir: Vec3,
    pub elapsed_ms: f64,
}

struct Tri {
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    n0: Vec3,
    n1: Vec3,
    n2: Vec3,
    uv0: f32,
    uv1: f32,
    uv2: f32,
    material: Material,
}

impl Tri {
    fn centroid(&self) -> Vec3 {
        (self.p0 + self.p1 + self.p2) / 3.0
    }
}

#[derive(Clone, Copy)]
struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }

    fn grow_point(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }
}

struct Node {
    aabb: Aabb,
    left: u32,
    right: u32,
    /// Leaves: first triangle index.
    first: u32,
    /// 0 for internal nodes, triangle count for leaves.
    count: u32,
}

struct Bvh {
    tris: Vec<Tri>,
    order: Vec<u32>,
    nodes: Vec<Node>,
}

impl Bvh {
    fn build(tris: Vec<Tri>) -> Self {
        let n = tris.len();
        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut nodes = Vec::with_capacity(n.max(1) * 2);
        if n == 0 {
            nodes.push(Node {
                aabb: Aabb::empty(),
                left: 0,
                right: 0,
                first: 0,
                count: 0,
            });
        } else {
            let mut bvh = Self {
                tris,
                order: Vec::new(),
                nodes: Vec::new(),
            };
            bvh.build_node(&mut order, 0, n, &mut nodes);
            bvh.order = order;
            bvh.nodes = nodes;
            return bvh;
        }
        Self {
            tris,
            order,
            nodes,
        }
    }

    fn build_node(
        &self,
        order: &mut [u32],
        start: usize,
        end: usize,
        nodes: &mut Vec<Node>,
    ) -> u32 {
        let mut bounds = Aabb::empty();
        let mut centroid_bounds = Aabb::empty();
        for &i in &order[start..end] {
            let t = &self.tris[i as usize];
            bounds.grow_point(t.p0);
            bounds.grow_point(t.p1);
            bounds.grow_point(t.p2);
            centroid_bounds.grow_point(t.centroid());
        }
        let node_index = nodes.len() as u32;
        nodes.push(Node {
            aabb: bounds,
            left: 0,
            right: 0,
            first: 0,
            count: 0,
        });
        let count = end - start;
        if count <= 6 {
            nodes[node_index as usize].first = start as u32;
            nodes[node_index as usize].count = count as u32;
            return node_index;
        }
        let extent = centroid_bounds.max - centroid_bounds.min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        let mid = start + count / 2;
        order[start..end].sort_by(|a, b| {
            let ca = self.tris[*a as usize].centroid()[axis];
            let cb = self.tris[*b as usize].centroid()[axis];
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let left = self.build_node(order, start, mid, nodes);
        let right = self.build_node(order, mid, end, nodes);
        let node = &mut nodes[node_index as usize];
        node.left = left;
        node.right = right;
        node.count = 0;
        node_index
    }

    fn hit(&self, o: Vec3, d: Vec3, t_max: f32) -> Option<(usize, f32, f32, f32)> {
        let inv = Vec3::new(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
        let mut best: Option<(usize, f32, f32, f32)> = None;
        let mut best_t = t_max;
        let mut stack = [0u32; 64];
        let mut stack_len = 1usize;
        stack[0] = 0;
        while stack_len > 0 {
            stack_len -= 1;
            let node = &self.nodes[stack[stack_len] as usize];
            if !slab_hit(&node.aabb, o, inv, best_t) {
                continue;
            }
            if node.count > 0 {
                let first = node.first as usize;
                for &ti in &self.order[first..first + node.count as usize] {
                    let tri = &self.tris[ti as usize];
                    if let Some((t, u, v)) = intersect(o, d, tri) {
                        if t < best_t {
                            best_t = t;
                            best = Some((ti as usize, t, u, v));
                        }
                    }
                }
            } else {
                stack[stack_len] = node.left;
                stack_len += 1;
                stack[stack_len] = node.right;
                stack_len += 1;
            }
        }
        best
    }
}

fn slab_hit(b: &Aabb, o: Vec3, inv: Vec3, t_max: f32) -> bool {
    let t0 = (b.min - o) * inv;
    let t1 = (b.max - o) * inv;
    let tmin = t0.min(t1).max_element().max(0.0);
    let tmax = t0.max(t1).min_element().min(t_max);
    tmin <= tmax
}

fn gather_triangles(kind: ExhibitKind, material: Material, rotation: f32, y_offset: f32) -> Vec<Tri> {
    let model = Mat4::from_rotation_y(rotation) * Mat4::from_translation(Vec3::new(0.0, y_offset, 0.0));
    let mut out = Vec::new();
    for mesh in geometry::build_exhibit(kind) {
        for idx in mesh.indices.chunks_exact(3) {
            let v = [
                &mesh.vertices[idx[0] as usize],
                &mesh.vertices[idx[1] as usize],
                &mesh.vertices[idx[2] as usize],
            ];
            let p = v.map(|x| model.transform_point3(Vec3::from(x.position)));
            let n = v.map(|x| model.transform_vector3(Vec3::from(x.normal)).normalize());
            out.push(Tri {
                p0: p[0],
                p1: p[1],
                p2: p[2],
                n0: n[0],
                n1: n[1],
                n2: n[2],
                uv0: v[0].uv[0],
                uv1: v[1].uv[0],
                uv2: v[2].uv[0],
                material,
            });
        }
    }
    out
}

fn ground_material() -> Material {
    Material {
        base_color: [0.24, 0.235, 0.225],
        roughness: 0.72,
        metallic: 0.0,
        reflectance: 0.35,
        finish: 2,
    }
}

/// Möller–Trumbore. Returns (t, u, v).
fn intersect(ray_o: Vec3, ray_d: Vec3, tri: &Tri) -> Option<(f32, f32, f32)> {
    let e1 = tri.p1 - tri.p0;
    let e2 = tri.p2 - tri.p0;
    let p = ray_d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-9 {
        return None;
    }
    let inv = 1.0 / det;
    let tv = ray_o - tri.p0;
    let u = tv.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = tv.cross(e1);
    let v = ray_d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    if t > 1e-4 {
        Some((t, u, v))
    } else {
        None
    }
}

fn ground_hit(o: Vec3, d: Vec3, t_max: f32) -> Option<f32> {
    if d.y.abs() < 1e-6 {
        return None;
    }
    let t = -o.y / d.y;
    if t <= 1e-4 || t >= t_max {
        return None;
    }
    let p = o + d * t;
    if p.x.abs() <= 6.0 && p.z.abs() <= 6.0 {
        Some(t)
    } else {
        None
    }
}

pub struct Probe {
    pub position: Vec3,
    pub normal: Vec3,
    pub reflected_dir: Vec3,
    pub roughness: f32,
    pub lod: f32,
    /// Linear radiance sampled from the prefiltered cube at `lod`.
    pub env_radiance: Vec3,
    /// Full shaded linear colour at the hit point.
    pub shaded: Vec3,
}

/// Trace one ray through the given pixel (NDC in [-1, 1], y up) and report the
/// reflection details. This is the number behind the UI's "reflection detail"
/// readout and is computed by the same CPU kernel as the reference renderer.
pub fn probe(
    precomputed: &Precomputed,
    settings: &ReferenceSettings,
    ndc_x: f32,
    ndc_y: f32,
) -> Option<Probe> {
    let aspect = settings.width.max(1) as f32 / settings.height.max(1) as f32;
    let ctx = LightingContext {
        camera: settings.camera,
        env_rotation: settings.env_rotation,
        sh: &precomputed.sh,
        prefilter: &precomputed.prefilter,
    };
    let tris = gather_triangles(
        settings.exhibit,
        settings.material,
        settings.exhibit_rotation,
        settings.exhibit.floor_offset(),
    );
    let bvh = Bvh::build(tris);
    let o = Vec3::from(settings.camera.position);
    let d = settings.camera.ray(ndc_x, ndc_y, aspect);
    let hit = bvh.hit(o, d, f32::INFINITY);
    let ground_t = ground_hit(o, d, hit.map(|h| h.1).unwrap_or(f32::INFINITY));
    if hit.is_none() && ground_t.is_none() {
        return None;
    }
    let (position, normal, material, uv_x) = match hit {
        Some((ti, t, u, v)) => {
            let tri = &bvh.tris[ti];
            let w0 = 1.0 - u - v;
            let normal = (tri.n0 * w0 + tri.n1 * u + tri.n2 * v).normalize();
            let uv_x = tri.uv0 * w0 + tri.uv1 * u + tri.uv2 * v;
            (o + d * t, normal, tri.material, uv_x)
        }
        None => (o + d * ground_t.unwrap(), Vec3::Y, ground_material(), 0.0),
    };
    let roughness = effective_roughness(&material, uv_x, settings.forced_roughness);
    let v = (o - position).normalize();
    let reflected = (-v).reflect(normal).normalize();
    let env_radiance = ctx.prefiltered(reflected, roughness);
    let shaded = ctx.shade(position, normal, &material, uv_x, settings.forced_roughness);
    Some(Probe {
        position,
        normal,
        reflected_dir: reflected,
        roughness,
        lod: ctx.roughness_to_lod(roughness),
        env_radiance,
        shaded,
    })
}

pub fn render(precomputed: &Precomputed, settings: &ReferenceSettings) -> ReferenceImage {
    let started = std::time::Instant::now();
    let w = settings.width.max(16);
    let h = settings.height.max(16);
    let aspect = w as f32 / h as f32;
    let ctx = LightingContext {
        camera: settings.camera,
        env_rotation: settings.env_rotation,
        sh: &precomputed.sh,
        prefilter: &precomputed.prefilter,
    };
    let tris = gather_triangles(
        settings.exhibit,
        settings.material,
        settings.exhibit_rotation,
        settings.exhibit.floor_offset(),
    );
    let bvh = Bvh::build(tris);
    let ground = ground_material();

    let mut rgb8 = vec![0u8; (w * h * 3) as usize];
    let cam_origin = Vec3::from(settings.camera.position);

    let row_stride = w as usize * 3;
    rgb8.par_chunks_mut(row_stride)
        .enumerate()
        .for_each(|(row, out_row)| {
            let y = 1.0 - 2.0 * (row as f32 + 0.5) / h as f32;
            for col in 0..w as usize {
                let x = 2.0 * (col as f32 + 0.5) / w as f32 - 1.0;
                let dir = settings.camera.ray(x, y, aspect);
                let tri_hit = bvh.hit(cam_origin, dir, f32::INFINITY);
                let ground_t = ground_hit(
                    cam_origin,
                    dir,
                    tri_hit.map(|h| h.1).unwrap_or(f32::INFINITY),
                );

                let color = match (tri_hit, ground_t) {
                    (Some((ti, t, u, v)), _) => {
                        let tri = &bvh.tris[ti];
                        let w0 = 1.0 - u - v;
                        let pos = cam_origin + dir * t;
                        let n = (tri.n0 * w0 + tri.n1 * u + tri.n2 * v).normalize();
                        let uv_x = tri.uv0 * w0 + tri.uv1 * u + tri.uv2 * v;
                        ctx.shade(pos, n, &tri.material, uv_x, settings.forced_roughness)
                    }
                    (_, Some(gt)) => {
                        let pos = cam_origin + dir * gt;
                        ctx.shade(pos, Vec3::Y, &ground, 0.0, settings.forced_roughness)
                    }
                    _ => ctx.background(dir),
                };
                let mapped = aces_tonemap(color, settings.exposure);
                let enc = srgb_encode(mapped);
                out_row[col * 3] = (enc.x.clamp(0.0, 1.0) * 255.0).round() as u8;
                out_row[col * 3 + 1] = (enc.y.clamp(0.0, 1.0) * 255.0).round() as u8;
                out_row[col * 3 + 2] = (enc.z.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        });

    ReferenceImage {
        width: w,
        height: h,
        rgb8,
        brightest_dir: brightest_in_cache(&precomputed.prefilter, settings.env_rotation).0,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

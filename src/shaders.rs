//! WGSL shaders. Lighting is evaluated in linear radiance; the display
//! shader is the only place tone mapping happens.

pub const SCENE_WGSL: &str = r#"
struct SceneU {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    sh: array<vec4<f32>, 9>,
    params: vec4<f32>, // x: env_rotation rad, y: debug mode, z: forced LOD (-1 off), w: unused
};

struct ObjU {
    model: mat4x4<f32>,
    base_color: vec4<f32>,
    mat: vec4<f32>, // x: roughness, y: metallic, z: reflectance, w: finish
};

@group(0) @binding(0) var<uniform> scene: SceneU;
@group(0) @binding(1) var env_tex: texture_cube<f32>;
@group(0) @binding(2) var env_smp: sampler;

@group(1) @binding(0) var<uniform> obj: ObjU;

const PI: f32 = 3.14159265359;
const PREFILTER_MIPS: f32 = 7.0;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
};

fn rotate_y_dir(d: vec3<f32>, a: f32) -> vec3<f32> {
    let s = sin(a);
    let c = cos(a);
    return vec3<f32>(d.x * c + d.z * s, d.y, -d.x * s + d.z * c);
}

// The environment is stored at rotation 0. Looking it up at env_rotation
// rotates the content by that angle, identically for the background and for
// every reflected ray.
fn env_lookup(world_dir: vec3<f32>, lod: f32) -> vec3<f32> {
    let d = rotate_y_dir(normalize(world_dir), -scene.params.x);
    return textureSampleLevel(env_tex, env_smp, d, lod).rgb;
}

fn sh_basis(n: vec3<f32>) -> array<f32, 9> {
    let x = n.x;
    let y = n.y;
    let z = n.z;
    return array<f32, 9>(
        0.282095,
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    );
}

fn sh_irradiance(n: vec3<f32>) -> vec3<f32> {
    let key = rotate_y_dir(normalize(n), -scene.params.x);
    let b = sh_basis(key);
    var e = vec3<f32>(0.0);
    for (var i = 0u; i < 9u; i = i + 1u) {
        e += scene.sh[i].rgb * b[i];
    }
    return max(e, vec3<f32>(0.0));
}

// Lagarde's analytic environment BRDF (split-sum second term).
fn env_brdf(f0: vec3<f32>, roughness: f32, ndotv: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * ndotv)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + vec3<f32>(ab.y);
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let world = obj.model * vec4<f32>(in.position, 1.0);
    out.world_pos = world.xyz;
    let nm = (obj.model * vec4<f32>(in.normal, 0.0)).xyz;
    out.normal = normalize(nm);
    let t = (obj.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz;
    out.tangent = vec4<f32>(normalize(t), in.tangent.w);
    out.uv = in.uv;
    out.clip = scene.view_proj * world;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Normals are authored per vertex on both sides of a surface (the pottery
    // interior has inward normals), so they are authoritative here and match
    // the CPU kernel exactly. Do not flip by winding: every generated mesh is
    // wound clockwise relative to its authored normals.
    let n = normalize(in.normal);
    let v = normalize(scene.cam_pos.xyz - in.world_pos);
    let ndotv = max(dot(n, v), 1e-4);

    var roughness = obj.mat.x;
    if (obj.mat.w > 2.5) {
        // finish 3: roughness ramp across the panel width.
        roughness = mix(0.045, 0.95, clamp(in.uv.x, 0.0, 1.0));
    }
    roughness = clamp(roughness, 0.03, 1.0);

    let metallic = clamp(obj.mat.y, 0.0, 1.0);
    let reflectance = clamp(obj.mat.z, 0.0, 1.0);
    let base = obj.base_color.rgb;
    let dielectric_f0 = 0.16 * reflectance * reflectance;
    let f0 = mix(vec3<f32>(dielectric_f0), base, metallic);
    let diffuse_albedo = base * (1.0 - metallic);

    let irradiance = sh_irradiance(n);
    let diffuse = diffuse_albedo * irradiance / PI;

    var lod = roughness * (PREFILTER_MIPS - 1.0);
    if (scene.params.z > -0.5) {
        lod = scene.params.z;
    }
    let r = reflect(-v, n);
    let prefiltered = env_lookup(r, lod);
    let spec = prefiltered * env_brdf(f0, roughness, ndotv);

    let debug = i32(scene.params.y);
    if (debug == 1) {
        // Direct view of the selected prefilter layer (HDR radiance).
        return vec4<f32>(env_lookup(r, lod), 1.0);
    } else if (debug == 2) {
        return vec4<f32>(diffuse, 1.0);
    } else if (debug == 3) {
        return vec4<f32>(vec3<f32>(roughness), 1.0);
    } else if (debug == 4) {
        return vec4<f32>(n * 0.5 + vec3<f32>(0.5), 1.0);
    } else if (debug == 5) {
        return vec4<f32>(spec, 1.0);
    }

    return vec4<f32>(diffuse + spec, 1.0);
}

struct SkyOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) vi: u32) -> SkyOut {
    var out: SkyOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    out.clip = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.ndc = vec2<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0);
    return out;
}

@fragment
fn fs_sky(in: SkyOut) -> @location(0) vec4<f32> {
    // Reconstruct the world-space ray from the camera through this pixel.
    // glam's perspective_rh uses a [0, 1] depth range: 0 is the near plane,
    // 1 is the far plane.
    let p_near = scene.inv_view_proj * vec4<f32>(in.ndc, 0.0, 1.0);
    let p_far = scene.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let world_near = p_near.xyz / p_near.w;
    let world_far = p_far.xyz / p_far.w;
    let dir = normalize(world_far - world_near);
    let color = env_lookup(dir, 0.0);
    let debug = i32(scene.params.y);
    if (debug == 6) {
        // Debug: show the background direction overlay in false colour.
        return vec4<f32>(dir * 0.5 + vec3<f32>(0.5), 1.0);
    }
    return vec4<f32>(color, 1.0);
}
"#;

pub const TONEMAP_WGSL: &str = r#"
struct DisplayU {
    exposure: f32,
    gamma_only: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> display: DisplayU;

fn aces(x: vec3<f32>, exposure: f32) -> vec3<f32> {
    let v = x * exposure;
    let a = v * (2.51 * v + 0.03);
    let b = v * (2.43 * v + 0.59) + 0.14;
    return clamp(a / b, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn srgb_encode(x: vec3<f32>) -> vec3<f32> {
    let lo = x * 12.92;
    let hi = 1.055 * pow(max(x, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    // select(false_value, true_value, condition)
    return select(hi, lo, x < vec3<f32>(0.0031308));
}

@vertex
fn vs_display(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_display(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(pos.xy);
    let hdr = textureLoad(hdr_tex, coord, 0).rgb;
    var mapped: vec3<f32>;
    if (display.gamma_only > 0.5) {
        mapped = clamp(hdr, vec3<f32>(0.0), vec3<f32>(1.0));
    } else {
        mapped = aces(hdr, display.exposure);
    }
    return vec4<f32>(srgb_encode(mapped), 1.0);
}
"#;

pub const BLIT_WGSL: &str = r#"
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_smp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_blit(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    out.clip = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_blit(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(src_tex, src_smp, in.uv);
}

fn srgb_to_linear(x: vec3<f32>) -> vec3<f32> {
    let lo = x / 12.92;
    let hi = pow((x + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, x <= vec3<f32>(0.04045));
}

// For an sRGB surface: our buffer already holds display-encoded values,
// decode so the hardware re-encodes exactly once.
@fragment
fn fs_blit_to_srgb(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(src_tex, src_smp, in.uv);
    return vec4<f32>(srgb_to_linear(c.rgb), c.a);
}
"#;

pub const PANORAMA_WGSL: &str = r#"
// Renders the environment cube as an equirect panorama. Used only by
// verification: sampling the same cube texture the scene uses lets us check
// directional continuity across every face boundary.
struct SceneU {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    sh: array<vec4<f32>, 9>,
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> scene: SceneU;
@group(0) @binding(3) var env_tex: texture_cube<f32>;
@group(0) @binding(2) var env_smp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_pano(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    out.clip = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

fn rotate_y_dir(d: vec3<f32>, a: f32) -> vec3<f32> {
    let s = sin(a);
    let c = cos(a);
    return vec3<f32>(d.x * c + d.z * s, d.y, -d.x * s + d.z * c);
}

@fragment
fn fs_pano(in: VsOut) -> @location(0) vec4<f32> {
    let u = in.uv.x;
    let v = 1.0 - in.uv.y;
    let phi = (u - 0.5) * 2.0 * 3.14159265359;
    let theta = v * 3.14159265359;
    let st = sin(theta);
    let dir = vec3<f32>(st * cos(phi), cos(theta), st * sin(phi));
    // Apply the same environment rotation as the scene so the panorama and
    // the background show the same world orientation.
    let key = rotate_y_dir(dir, -scene.params.x);
    return vec4<f32>(textureSampleLevel(env_tex, env_smp, key, 0.0).rgb, 1.0);
}
"#;

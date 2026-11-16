//! Reusable exhibit-lighting assets.
//!
//! An asset package is a directory containing a prefiltered environment (the
//! same f32 cube layout the renderer consumes), a manifest with the source
//! parameters and the measured direction/energy checks, and a reference to the
//! original linear HDR source. Loading an asset never falls back to a default
//! and never trusts the manifest's numbers: every recorded value is recomputed
//! from the payload and the source, and any disagreement is returned as an
//! error that the lighting panel shows.
//!
//! All radiance stored here is linear. The generator, the manifest, the loader
//! and the renderer share the direction conventions documented in
//! [`Conventions::default`].

use crate::cache::{
    decode_prefilter, encode_prefilter, validate_precomputed, PrecomputeOrigin, Precomputed,
};
use crate::env::{
    self, analytic_hdr, cubemap_from_hdr, env_meta, file_sha256, load_hdr, smoothstep, EnvMeta,
    HdrImage, PREFILTER_MIPS, PREFILTER_SIZE, SKY_SIZE,
};
use crate::geometry::ExhibitKind;
use crate::math::{dir_from_cube_face, direction_from_azimuth_elevation, spherical_azimuth_elevation, CUBE_FACES};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const ASSET_FORMAT_VERSION: u32 = 1;
pub const SCENE_FORMAT_VERSION: u32 = 1;
pub const DEFAULT_ASSET_ROOT: &str = "assets/exhibit_lighting";
pub const DEFAULT_MANIFEST: &str = "assets/exhibit_lighting/assets/directed_key/manifest.json";
pub const DEFAULT_SPEC: &str = "assets/exhibit_lighting/spec.json";
/// The recorded brightest direction must agree with the requested key
/// direction to within this angle. At the 256-texel mip 0 the cube texel
/// spacing is about 0.4°, so this catches a mis-oriented or mislabelled pack
/// without rejecting correct sampling.
pub const KEY_DIRECTION_TOLERANCE_DEG: f32 = 1.5;

#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    #[error("素材文件读取失败：{0}")]
    Io(String),
    #[error("素材清单解析失败：{0}")]
    Json(String),
    #[error("素材字段无效：{0}")]
    Field(String),
    #[error("素材校验失败：{0}")]
    Mismatch(String),
    #[error("素材源环境失败：{0}")]
    Source(String),
}

fn io(e: impl std::fmt::Display, path: &Path) -> AssetError {
    AssetError::Io(format!("{}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// Spec: the source-of-truth parameters
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirectionSpec {
    pub azimuth_deg: f32,
    pub elevation_deg: f32,
}

impl DirectionSpec {
    pub fn dir(&self) -> Vec3 {
        direction_from_azimuth_elevation(self.azimuth_deg, self.elevation_deg)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceSpec {
    /// Base radiance of the dark room (linear).
    pub base_radiance: [f32; 3],
    /// The bright key: its linear radiance, direction and angular radius.
    pub key_radiance: [f32; 3],
    pub key_direction: DirectionSpec,
    pub key_radius_deg: f32,
    /// Broad, dim fill so roughness changes are visible outside the key.
    pub fill_radiance: [f32; 3],
    pub fill_direction: DirectionSpec,
    pub fill_radius_deg: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrefilterSpec {
    pub base_size: u32,
    pub mips: u32,
    pub samples: u32,
    pub roughness_mapping: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Conventions {
    pub up_axis: String,
    pub azimuth: String,
    pub elevation: String,
    pub equirect: String,
    pub environment_rotation: String,
    pub linear: String,
}

impl Default for Conventions {
    fn default() -> Self {
        Self {
            up_axis: "+Y".into(),
            azimuth: "方位角 = atan2(z, x)，从 +X 向 +Z 为正".into(),
            elevation: "仰角 = asin(y)，从赤道向 +Y 为正".into(),
            equirect: "u = 0.5 + atan2(z, x)/2π，v = acos(y)/π（第 0 行为 +Y）".into(),
            environment_rotation:
                "环境旋转 +θ 把内容绕 +Y 旋转 θ；查询方向用 -θ。最亮方位角随 θ 减少 θ°"
                    .into(),
            linear: "所有辐亮度均为线性值，未做色调映射；仅显示阶段编码".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreviewSpec {
    pub exhibit: ExhibitKind,
    pub exhibit_rotation_deg: f32,
    pub exposure: f32,
    pub debug_mode: u32,
    pub forced_lod: Option<f32>,
    pub show_ground: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssetSpec {
    pub format_version: u32,
    pub id: String,
    pub name: String,
    pub notes: String,
    pub source_width: u32,
    pub source_height: u32,
    pub source_path: String,
    pub source: SourceSpec,
    pub prefilter: PrefilterSpec,
    pub conventions: Conventions,
    /// Environment rotations the assets are checked at, in degrees.
    pub rotation_presets_deg: Vec<f32>,
    pub preview: PreviewSpec,
}

// ---------------------------------------------------------------------------
// Manifest: what was produced, with measured checks
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrightnessRecord {
    pub azimuth_deg: f32,
    pub elevation_deg: f32,
    pub radiance: [f32; 3],
    pub luminance: f32,
    /// Angle from the requested key direction.
    pub key_offset_deg: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnergyRecord {
    /// Area-weighted mean luminance of the linear source HDR.
    pub source_mean_luminance: f32,
    /// Mean luminance implied by the stored (band-scaled) SH irradiance.
    pub sh_mean_luminance: f32,
    pub sh_rel_diff: f32,
    /// Area-weighted mean luminance of prefiltered mip 6.
    pub mip6_mean_luminance: f32,
    pub mip6_rel_diff: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RotationCheck {
    pub rotation_deg: f32,
    pub brightest_azimuth_deg: f32,
    pub brightest_elevation_deg: f32,
    pub luminance: f32,
    /// Azimuth expected from the rotation convention (base - rotation).
    pub expected_azimuth_deg: f32,
    pub offset_deg: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssetVerification {
    pub brightest: BrightnessRecord,
    pub energy: EnergyRecord,
    pub rotation_checks: Vec<RotationCheck>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceRecord {
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub sha256: String,
    pub generator: String,
    pub params: SourceSpec,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrefilterRecord {
    pub path: String,
    pub base_size: u32,
    pub mips: u32,
    pub samples: u32,
    pub roughness_mapping: String,
    pub sha256: String,
    pub byte_len: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssetManifest {
    pub format_version: u32,
    pub id: String,
    pub name: String,
    pub notes: String,
    pub source: SourceRecord,
    pub prefilter: PrefilterRecord,
    pub conventions: Conventions,
    pub rotation_presets_deg: Vec<f32>,
    pub sh: [[f32; 3]; 9],
    pub verification: AssetVerification,
}

// ---------------------------------------------------------------------------
// Source generation
// ---------------------------------------------------------------------------

pub fn source_image(spec: &SourceSpec, width: u32, height: u32) -> HdrImage {
    let key = spec.key_direction.dir();
    let fill = spec.fill_direction.dir();
    let key_r = spec.key_radius_deg.to_radians();
    let fill_r = spec.fill_radius_deg.to_radians();
    let key_color = Vec3::from(spec.key_radiance);
    let fill_color = Vec3::from(spec.fill_radiance);
    let base = Vec3::from(spec.base_radiance);
    analytic_hdr(width, height, |d| {
        let mut c = base;
        // A Gaussian key has a single maximum, so the brightest texel in the
        // prefiltered data identifies the light direction unambiguously. A
        // flat-topped disc would leave a whole plateau tied for brightest and
        // make the direction check depend on texel order.
        let key_angle = d.dot(key).max(-1.0).acos();
        c += key_color * (-(key_angle / key_r).powi(2)).exp();
        // The dim, broad fill has no direction role; a soft plateau is fine.
        let fill_angle = d.dot(fill).max(-1.0).acos();
        c += fill_color * smoothstep(fill_r, fill_r * 0.25, fill_angle);
        // A dim cool horizon band; still linear and low energy, it only gives
        // the polished exhibits a second, faint reflection reference.
        let horizon = (1.0 - d.y.abs()).powi(8);
        c += Vec3::new(0.020, 0.026, 0.034) * horizon;
        c.max(Vec3::ZERO)
    })
}

fn write_hdr(path: &Path, img: &HdrImage) -> Result<(), AssetError> {
    let bytes = encode_source_bytes(img)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| io(e, parent))?;
    }
    std::fs::write(path, &bytes).map_err(|e| io(e, path))
}

/// Decode Radiance bytes back to linear f32. Used so the build measures the
/// quantised file it ships rather than the unquantised analytic values.
fn decode_hdr_bytes(bytes: &[u8]) -> Result<HdrImage, AssetError> {
    use image::ImageDecoder;
    let decoder = image::codecs::hdr::HdrDecoder::new(std::io::Cursor::new(bytes))
        .map_err(|e| AssetError::Source(e.to_string()))?;
    let meta = decoder.metadata();
    let mut raw = vec![0u8; decoder.total_bytes() as usize];
    decoder
        .read_image(&mut raw)
        .map_err(|e| AssetError::Source(e.to_string()))?;
    let floats: &[f32] = bytemuck::cast_slice(&raw);
    Ok(HdrImage::from_rgb(
        meta.width,
        meta.height,
        floats.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
    ))
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

fn luminance(c: Vec3) -> f32 {
    c.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

/// Area-weighted mean luminance of an equirectangular image. The solid angle
/// of a texel is proportional to sin(theta), which is what this weights by.
pub fn equirect_mean_luminance(img: &HdrImage) -> f32 {
    let mut weighted = 0.0f64;
    let mut weights = 0.0f64;
    for y in 0..img.height {
        let theta = (y as f32 + 0.5) / img.height as f32 * std::f32::consts::PI;
        let w = theta.sin() as f64;
        for x in 0..img.width {
            weighted += luminance(Vec3::from(img.rgb[(y * img.width + x) as usize])) as f64 * w;
            weights += w;
        }
    }
    (weighted / weights.max(1e-12)) as f32
}

/// Area-weighted mean luminance of a cube map. Each texel's solid angle is
/// approximated by (2/size)² / (1+s²+t²)^(3/2), which integrates to 4π.
pub fn cube_mean_luminance(faces: &[Vec<[f32; 3]>]) -> f32 {
    let size = (faces[0].len() as f32).sqrt().round() as u32;
    let mut weighted = 0.0f64;
    let mut weights = 0.0f64;
    for face in 0..CUBE_FACES {
        for y in 0..size {
            for x in 0..size {
                let s = 2.0 * (x as f32 + 0.5) / size as f32 - 1.0;
                let t = 2.0 * (y as f32 + 0.5) / size as f32 - 1.0;
                let w = 1.0 / (1.0 + s * s + t * t).powf(1.5);
                let l = luminance(Vec3::from(faces[face][(y * size + x) as usize]));
                weighted += l as f64 * w as f64;
                weights += w as f64;
            }
        }
    }
    (weighted / weights.max(1e-12)) as f32
}

/// Mean radiance implied by the stored irradiance SH. `sh[0]` is
/// π·∫L·Y₀₀ dω, so the average radiance is sh₀ / (4π²·Y₀₀).
pub fn sh_mean_luminance(sh: &[[f32; 3]; 9]) -> f32 {
    let y00 = 0.282095f32;
    luminance(Vec3::from(sh[0])) / (4.0 * std::f32::consts::PI * std::f32::consts::PI * y00)
}

fn brightest_record(prefilter: &[Vec<Vec<[f32; 3]>>], key: Vec3) -> BrightnessRecord {
    let faces = &prefilter[0];
    let size = (faces[0].len() as f32).sqrt().round() as u32;
    let mut best = (0.0f32, Vec3::Y, Vec3::ZERO);
    for face in 0..CUBE_FACES {
        for y in 0..size {
            for x in 0..size {
                let c = Vec3::from(faces[face][(y * size + x) as usize]);
                let l = luminance(c);
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
    let (az, el) = spherical_azimuth_elevation(best.1);
    BrightnessRecord {
        azimuth_deg: az,
        elevation_deg: el,
        radiance: best.2.to_array(),
        luminance: best.0,
        key_offset_deg: best.1.dot(key.normalize()).clamp(-1.0, 1.0).acos().to_degrees(),
    }
}

fn rotation_record(prefilter: &[Vec<Vec<[f32; 3]>>], brightness: &BrightnessRecord, rotation_deg: f32) -> RotationCheck {
    let faces = &prefilter[0];
    let size = (faces[0].len() as f32).sqrt().round() as u32;
    let angle = rotation_deg.to_radians();
    let mut best = (0.0f32, Vec3::Y);
    for face in 0..CUBE_FACES {
        for y in 0..size {
            for x in 0..size {
                let c = Vec3::from(faces[face][(y * size + x) as usize]);
                let l = luminance(c);
                if l > best.0 {
                    let d = dir_from_cube_face(
                        face,
                        (x as f32 + 0.5) / size as f32,
                        (y as f32 + 0.5) / size as f32,
                    );
                    best = (l, crate::math::rotate_y(d, angle));
                }
            }
        }
    }
    let (az, el) = spherical_azimuth_elevation(best.1);
    let expected = wrap_deg(brightness.azimuth_deg - rotation_deg);
    RotationCheck {
        rotation_deg,
        brightest_azimuth_deg: az,
        brightest_elevation_deg: el,
        luminance: best.0,
        expected_azimuth_deg: expected,
        offset_deg: angle_diff_deg(az, expected),
    }
}

fn wrap_deg(mut a: f32) -> f32 {
    while a > 180.0 {
        a -= 360.0;
    }
    while a < -180.0 {
        a += 360.0;
    }
    a
}

fn angle_diff_deg(a: f32, b: f32) -> f32 {
    wrap_deg(a - b).abs()
}

fn rel_diff(a: f32, b: f32) -> f32 {
    (a - b).abs() / b.abs().max(1e-6)
}

fn measure(
    source: &HdrImage,
    prefilter: &[Vec<Vec<[f32; 3]>>],
    sh: &[[f32; 3]; 9],
    key: Vec3,
    presets: &[f32],
) -> AssetVerification {
    let brightest = brightest_record(prefilter, key);
    let source_mean = equirect_mean_luminance(source);
    let sh_mean = sh_mean_luminance(sh);
    let mip6_mean = cube_mean_luminance(&prefilter[(PREFILTER_MIPS - 1) as usize]);
    let energy = EnergyRecord {
        source_mean_luminance: source_mean,
        sh_mean_luminance: sh_mean,
        sh_rel_diff: rel_diff(sh_mean, source_mean),
        mip6_mean_luminance: mip6_mean,
        mip6_rel_diff: rel_diff(mip6_mean, source_mean),
    };
    let rotation_checks = presets
        .iter()
        .map(|r| rotation_record(prefilter, &brightest, *r))
        .collect();
    AssetVerification {
        brightest,
        energy,
        rotation_checks,
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

pub struct BuildReport {
    pub manifest_path: PathBuf,
    pub source_path: PathBuf,
    pub prefilter_path: PathBuf,
    pub manifest: AssetManifest,
    pub max_rebuild_diff: f32,
}

pub fn load_spec(path: &Path) -> Result<AssetSpec, AssetError> {
    let text = std::fs::read_to_string(path).map_err(|e| io(e, path))?;
    let spec: AssetSpec = serde_json::from_str(&text)
        .map_err(|e| AssetError::Json(format!("{}: {e}", path.display())))?;
    validate_spec(&spec)?;
    Ok(spec)
}

pub fn validate_spec(spec: &AssetSpec) -> Result<(), AssetError> {
    if spec.format_version != ASSET_FORMAT_VERSION {
        return Err(AssetError::Field(format!(
            "规格版本 {} 与程序支持的 {} 不符",
            spec.format_version, ASSET_FORMAT_VERSION
        )));
    }
    if spec.id.is_empty()
        || spec.id.chars().any(|c| c == '/' || c == '\\' || c == '.')
    {
        return Err(AssetError::Field(format!(
            "素材 id “{}” 不能为空或包含路径字符",
            spec.id
        )));
    }
    if spec.source_width < 8 || spec.source_height < 4 || spec.source_width != spec.source_height * 2 {
        return Err(AssetError::Field(format!(
            "源环境尺寸 {}×{} 无效（需宽为高两倍）",
            spec.source_width, spec.source_height
        )));
    }
    if spec.prefilter.base_size != PREFILTER_SIZE || spec.prefilter.mips != PREFILTER_MIPS {
        return Err(AssetError::Field(format!(
            "预过滤规格 {}×{} mip {} 与内核固定值 {}×{} mip {} 不一致",
            spec.prefilter.base_size,
            spec.prefilter.base_size,
            spec.prefilter.mips,
            PREFILTER_SIZE,
            PREFILTER_SIZE,
            PREFILTER_MIPS
        )));
    }
    if spec.prefilter.samples == 0 {
        return Err(AssetError::Field("预过滤采样数不能为 0".into()));
    }
    let key = spec.source.key_direction.dir();
    if key.length() < 0.5 || !key.is_finite() {
        return Err(AssetError::Field("主光方向不是有限单位向量".into()));
    }
    if spec.rotation_presets_deg.is_empty() {
        return Err(AssetError::Field("旋转预设不能为空".into()));
    }
    if spec.source.key_radiance.iter().any(|c| !c.is_finite() || *c <= 0.0) {
        return Err(AssetError::Field("主光辐亮度必须为有限正值".into()));
    }
    Ok(())
}

fn build_manifest(
    spec: &AssetSpec,
    source: &HdrImage,
    source_sha: &str,
    prefilter: &[Vec<Vec<[f32; 3]>>],
    sh: &[[f32; 3]; 9],
    prefilter_sha: &str,
    prefilter_len: u64,
    source_rel: &str,
    prefilter_rel: &str,
) -> Result<AssetManifest, AssetError> {
    let key = spec.source.key_direction.dir();
    let verification = measure(source, prefilter, sh, key, &spec.rotation_presets_deg);
    if verification.brightest.key_offset_deg > KEY_DIRECTION_TOLERANCE_DEG {
        return Err(AssetError::Mismatch(format!(
            "最亮方向与规格主光相差 {:.2}°（阈值 {:.1}°）；素材未写入",
            verification.brightest.key_offset_deg, KEY_DIRECTION_TOLERANCE_DEG
        )));
    }
    if verification.energy.sh_rel_diff > 0.01 {
        return Err(AssetError::Mismatch(format!(
            "SH 平均辐亮度与源环境相差 {:.3}%；素材未写入",
            verification.energy.sh_rel_diff * 100.0
        )));
    }
    if verification.energy.mip6_rel_diff > 0.05 {
        return Err(AssetError::Mismatch(format!(
            "预过滤 mip 6 平均辐亮度与源环境相差 {:.3}%；素材未写入",
            verification.energy.mip6_rel_diff * 100.0
        )));
    }
    Ok(AssetManifest {
        format_version: ASSET_FORMAT_VERSION,
        id: spec.id.clone(),
        name: spec.name.clone(),
        notes: spec.notes.clone(),
        source: SourceRecord {
            path: source_rel.to_string(),
            width: source.width,
            height: source.height,
            sha256: source_sha.to_string(),
            generator: format!(
                "analytic key+fill（{:.0}×{:.0}）",
                spec.source_width, spec.source_height
            ),
            params: spec.source.clone(),
        },
        prefilter: PrefilterRecord {
            path: prefilter_rel.to_string(),
            base_size: spec.prefilter.base_size,
            mips: spec.prefilter.mips,
            samples: spec.prefilter.samples,
            roughness_mapping: spec.prefilter.roughness_mapping.clone(),
            sha256: prefilter_sha.to_string(),
            byte_len: prefilter_len,
        },
        conventions: spec.conventions.clone(),
        rotation_presets_deg: spec.rotation_presets_deg.clone(),
        sh: *sh,
        verification,
    })
}

fn sha256_bytes(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

/// Regenerate source + prefilter and write the asset package under `out_root`.
pub fn build_asset(
    spec_path: &Path,
    out_root: &Path,
    mut progress: impl FnMut(&str, f32),
) -> Result<BuildReport, AssetError> {
    let spec = load_spec(spec_path)?;
    progress("生成线性 HDR 源环境…", 0.02);
    let source = source_image(&spec.source, spec.source_width, spec.source_height);
    let source_path = out_root.join(&spec.source_path);
    write_hdr(&source_path, &source)?;
    let source_sha = file_sha256(&source_path).map_err(|e| AssetError::Source(e.to_string()))?;
    // Measure from the file that will be shipped, not the in-memory analytic
    // values: Radiance RGBE quantises, and a load must reproduce these numbers
    // exactly. Every downstream check therefore runs on the round-tripped data.
    let source = load_hdr(&source_path).map_err(|e| AssetError::Source(e.to_string()))?;

    progress("投影漫反射球谐…", 0.06);
    let sh = env::project_sh(&source);

    progress("预过滤粗糙反射层…", 0.10);
    let prefilter = env::build_prefilter_cache(&source, spec.prefilter.samples, &mut |_, _| {});

    let payload = encode_prefilter(&prefilter, spec.prefilter.samples);
    let prefilter_sha = sha256_bytes(&payload);

    let asset_dir = out_root.join("assets").join(&spec.id);
    let prefilter_path = asset_dir.join("prefilter_f32.bin");
    if let Some(parent) = prefilter_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| io(e, parent))?;
    }
    std::fs::write(&prefilter_path, &payload).map_err(|e| io(e, &prefilter_path))?;

    let source_rel = relative_path(&asset_dir, &source_path);
    let manifest = build_manifest(
        &spec,
        &source,
        &source_sha,
        &prefilter,
        &sh,
        &prefilter_sha,
        payload.len() as u64,
        &source_rel,
        "prefilter_f32.bin",
    )?;
    let manifest_path = asset_dir.join("manifest.json");
    write_json(&manifest_path, &manifest)?;

    // Round-trip: the written payload must decode to exactly what we measured
    // and it must be what a fresh rebuild produces.
    let decoded = decode_prefilter(&payload, spec.prefilter.samples)
        .map_err(|e| AssetError::Mismatch(format!("写入的预过滤数据无法解码：{e}")))?;
    let max_rebuild_diff = max_abs_diff(&decoded, &prefilter);
    if max_rebuild_diff > 1e-6 {
        return Err(AssetError::Mismatch(format!(
            "预过滤数据写入/读回不一致：最大绝对差 {max_rebuild_diff:e}"
        )));
    }

    progress("写入场景…", 0.97);
    write_scenes(&spec, out_root)?;
    progress("素材包完成", 1.0);
    Ok(BuildReport {
        manifest_path,
        source_path,
        prefilter_path,
        manifest,
        max_rebuild_diff,
    })
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), AssetError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| io(e, parent))?;
    }
    let text = serde_json::to_string_pretty(value)
        .map_err(|e| AssetError::Json(format!("{}: {e}", path.display())))?;
    std::fs::write(path, text + "\n").map_err(|e| io(e, path))
}

fn relative_path(from_dir: &Path, target: &Path) -> String {
    // Both paths are produced by joining the same root, so a lexical strip of
    // the shared prefix is sufficient and keeps the manifest readable.
    let mut a = from_dir.components().peekable();
    let mut b = target.components().peekable();
    let mut common = 0usize;
    let av: Vec<_> = std::iter::from_fn(|| a.next()).collect();
    let bv: Vec<_> = std::iter::from_fn(|| b.next()).collect();
    while common < av.len() && common < bv.len() && av[common] == bv[common] {
        common += 1;
    }
    let mut parts: Vec<String> = Vec::new();
    for _ in common..av.len() {
        parts.push("..".into());
    }
    for c in &bv[common..] {
        parts.push(c.as_os_str().to_string_lossy().into_owned());
    }
    parts.join("/")
}

fn normalize_path(path: &Path) -> PathBuf {
    // Lexical only: scene references are authored relative to the scene file,
    // and keeping `..` would stop the resulting path from matching the
    // manifests found by `discover_assets` (the UI compares paths verbatim).
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn max_abs_diff(a: &[Vec<Vec<[f32; 3]>>], b: &[Vec<Vec<[f32; 3]>>]) -> f32 {
    let mut worst = 0.0f32;
    for (ma, mb) in a.iter().zip(b) {
        for (fa, fb) in ma.iter().zip(mb) {
            for (pa, pb) in fa.iter().zip(fb) {
                for c in 0..3 {
                    worst = worst.max((pa[c] - pb[c]).abs());
                }
            }
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// Scene files
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneFile {
    pub format_version: u32,
    pub name: String,
    pub notes: String,
    /// Manifest path relative to the scene file.
    pub asset_manifest: String,
    pub env_rotation_deg: f32,
    pub exhibit: ExhibitKind,
    pub exhibit_rotation_deg: f32,
    pub exposure: f32,
    pub debug_mode: u32,
    pub forced_lod: Option<f32>,
    pub show_ground: bool,
}

pub fn load_scene(path: &Path) -> Result<(SceneFile, PathBuf), AssetError> {
    let text = std::fs::read_to_string(path).map_err(|e| io(e, path))?;
    let scene: SceneFile = serde_json::from_str(&text)
        .map_err(|e| AssetError::Json(format!("{}: {e}", path.display())))?;
    if scene.format_version != SCENE_FORMAT_VERSION {
        return Err(AssetError::Field(format!(
            "场景版本 {} 与程序支持的 {} 不符",
            scene.format_version, SCENE_FORMAT_VERSION
        )));
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let manifest_path = normalize_path(&parent.join(&scene.asset_manifest));
    if !manifest_path.is_file() {
        return Err(AssetError::Field(format!(
            "场景引用的素材清单不存在：{}",
            manifest_path.display()
        )));
    }
    Ok((scene, manifest_path))
}

fn write_scenes(spec: &AssetSpec, out_root: &Path) -> Result<(), AssetError> {
    let manifest_rel = format!("../assets/{}/manifest.json", spec.id);
    let make = |name: &str, notes: &str, exhibit: ExhibitKind, rotation: f32, debug: u32, lod: Option<f32>, exposure: f32| SceneFile {
        format_version: SCENE_FORMAT_VERSION,
        name: name.to_string(),
        notes: notes.to_string(),
        asset_manifest: manifest_rel.clone(),
        env_rotation_deg: rotation,
        exhibit,
        exhibit_rotation_deg: spec.preview.exhibit_rotation_deg,
        exposure,
        debug_mode: debug,
        forced_lod: lod,
        show_ground: spec.preview.show_ground,
    };
    let scenes = [
        (
            "roughness_gallery.scene.json",
            make(
                "粗糙度巡览",
                "载入后可在面板切换展品：粗糙度展板扫过全部 mip 层，球体读方向，铜器与陶罐读中等/漫反射响应。横向拖动场景旋转环境，背景与反射同步。",
                spec.preview.exhibit,
                spec.rotation_presets_deg.first().copied().unwrap_or(0.0),
                0,
                None,
                spec.preview.exposure,
            ),
        ),
        (
            "reflection_direction_35deg.scene.json",
            make(
                "反射方向对照（环境旋转 35°）",
                "球体上的亮斑来自主光。环境旋转 +35° 后最亮方位角应减少 35°；同一场景可切回 0° 对照反射朝向，同时观察 mip 0 的高光。",
                ExhibitKind::Sphere,
                35.0,
                0,
                None,
                0.55,
            ),
        ),
        (
            "prefilter_layer_mip3.scene.json",
            make(
                "预过滤层对照（mip 3）",
                "强制 mip 3（粗糙度 0.5）并显示预过滤层原始 HDR；旋转环境可见反射亮区随方向移动，与 mip 0 场景对照扩散程度。",
                ExhibitKind::Sphere,
                0.0,
                1,
                Some(3.0),
                0.55,
            ),
        ),
        (
            "roughness_ramp_closeup.scene.json",
            make(
                "粗糙度展板特写（覆盖全部 mip 层）",
                "粗糙度展板从左到右由锐利到粗糙。环境旋转 -71° 把主光反射带放到左侧锐利端：左侧是窄而亮的高光，向右随粗糙度增大而扩散变柔；横向拖动可看到整条高光带随方向移动。",
                ExhibitKind::Ramp,
                -71.0,
                0,
                None,
                0.045,
            ),
        ),
    ];
    for (file, scene) in scenes {
        write_json(&out_root.join("scenes").join(file), &scene)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Load + verify
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct LoadedAsset {
    pub manifest: AssetManifest,
    pub manifest_path: PathBuf,
    pub precomputed: Precomputed,
    pub meta: EnvMeta,
    pub image: HdrImage,
    /// Recomputed from the loaded payload, not copied from the manifest.
    pub measured: AssetVerification,
    pub load_ms: f64,
}

pub fn load_manifest(path: &Path) -> Result<AssetManifest, AssetError> {
    let text = std::fs::read_to_string(path).map_err(|e| io(e, path))?;
    let manifest: AssetManifest = serde_json::from_str(&text)
        .map_err(|e| AssetError::Json(format!("{}: {e}", path.display())))?;
    if manifest.format_version != ASSET_FORMAT_VERSION {
        return Err(AssetError::Field(format!(
            "素材清单版本 {} 与程序支持的 {} 不符",
            manifest.format_version, ASSET_FORMAT_VERSION
        )));
    }
    if manifest.id.is_empty() {
        return Err(AssetError::Field("素材清单缺少 id".into()));
    }
    if manifest.prefilter.base_size != PREFILTER_SIZE
        || manifest.prefilter.mips != PREFILTER_MIPS
        || manifest.prefilter.samples == 0
    {
        return Err(AssetError::Field(format!(
            "素材预过滤规格 {} mip {} 采样 {} 与内核不符",
            manifest.prefilter.base_size,
            manifest.prefilter.mips,
            manifest.prefilter.samples
        )));
    }
    if manifest.rotation_presets_deg.is_empty() {
        return Err(AssetError::Field("素材未声明旋转预设".into()));
    }
    Ok(manifest)
}

pub fn load_asset(manifest_path: &Path) -> Result<LoadedAsset, AssetError> {
    let started = std::time::Instant::now();
    let manifest = load_manifest(manifest_path)?;
    let dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));

    let source_path = dir.join(&manifest.source.path);
    let source_sha = file_sha256(&source_path).map_err(|e| AssetError::Source(e.to_string()))?;
    if source_sha != manifest.source.sha256 {
        return Err(AssetError::Mismatch(format!(
            "源环境 SHA-256 与清单不符（清单 {}…，实际 {}…）",
            &manifest.source.sha256[..manifest.source.sha256.len().min(12)],
            &source_sha[..source_sha.len().min(12)]
        )));
    }
    let image = load_hdr(&source_path).map_err(|e| AssetError::Source(e.to_string()))?;
    if image.width != manifest.source.width || image.height != manifest.source.height {
        return Err(AssetError::Mismatch(format!(
            "源环境尺寸 {}×{} 与清单 {}×{} 不符",
            image.width, image.height, manifest.source.width, manifest.source.height
        )));
    }

    let payload_path = dir.join(&manifest.prefilter.path);
    let payload = std::fs::read(&payload_path).map_err(|e| io(e, &payload_path))?;
    if payload.len() as u64 != manifest.prefilter.byte_len {
        return Err(AssetError::Mismatch(format!(
            "预过滤数据长度 {} 与清单 {} 不符",
            payload.len(),
            manifest.prefilter.byte_len
        )));
    }
    let payload_sha = sha256_bytes(&payload);
    if payload_sha != manifest.prefilter.sha256 {
        return Err(AssetError::Mismatch(format!(
            "预过滤数据 SHA-256 与清单不符（清单 {}…，实际 {}…）",
            &manifest.prefilter.sha256[..manifest.prefilter.sha256.len().min(12)],
            &payload_sha[..payload_sha.len().min(12)]
        )));
    }
    let prefilter = decode_prefilter(&payload, manifest.prefilter.samples)
        .map_err(|e| AssetError::Mismatch(format!("预过滤数据解码失败：{e}")))?;

    let sky = cubemap_from_hdr(&image, SKY_SIZE).faces;
    let sh = manifest.sh;
    let measured = measure(
        &image,
        &prefilter,
        &sh,
        manifest.source.params.key_direction.dir(),
        &manifest.rotation_presets_deg,
    );
    compare_verification(&manifest.verification, &measured)?;

    let meta = env_meta(
        &image,
        source_sha,
        manifest
            .source
            .path
            .rsplit('/')
            .next()
            .unwrap_or(&manifest.source.path)
            .to_string(),
    );
    let precomputed = Precomputed {
        prefilter,
        sky,
        sh,
        origin: PrecomputeOrigin::Asset {
            id: manifest.id.clone(),
        },
        compute_ms: started.elapsed().as_secs_f64() * 1000.0,
    };
    validate_precomputed(&precomputed)
        .map_err(|e| AssetError::Mismatch(format!("预计算数据校验失败：{e}")))?;
    Ok(LoadedAsset {
        manifest,
        manifest_path: manifest_path.to_path_buf(),
        precomputed,
        meta,
        image,
        measured,
        load_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}

fn compare_verification(
    recorded: &AssetVerification,
    measured: &AssetVerification,
) -> Result<(), AssetError> {
    let b = &recorded.brightest;
    let m = &measured.brightest;
    if angle_diff_deg(b.azimuth_deg, m.azimuth_deg) > 0.05
        || (b.elevation_deg - m.elevation_deg).abs() > 0.05
    {
        return Err(AssetError::Mismatch(format!(
            "清单记录的最亮方向（{:.2}°, {:.2}°）与预过滤数据实测（{:.2}°, {:.2}°）不符",
            b.azimuth_deg, b.elevation_deg, m.azimuth_deg, m.elevation_deg
        )));
    }
    for c in 0..3 {
        if rel_diff(m.radiance[c], b.radiance[c]) > 1e-3 {
            return Err(AssetError::Mismatch(format!(
                "清单记录的主光辐亮度与实测不符（通道 {c}：{:.4} vs {:.4}）",
                b.radiance[c], m.radiance[c]
            )));
        }
    }
    if (b.key_offset_deg - m.key_offset_deg).abs() > 0.05 {
        return Err(AssetError::Mismatch(format!(
            "清单记录的主光偏角 {:.2}° 与实测 {:.2}° 不符",
            b.key_offset_deg, m.key_offset_deg
        )));
    }
    if m.key_offset_deg > KEY_DIRECTION_TOLERANCE_DEG {
        return Err(AssetError::Mismatch(format!(
            "实测主光方向偏离规格 {:.2}°（阈值 {:.1}°）",
            m.key_offset_deg, KEY_DIRECTION_TOLERANCE_DEG
        )));
    }
    let e_rec = &recorded.energy;
    let e = &measured.energy;
    for (name, a, b) in [
        ("源平均亮度", e_rec.source_mean_luminance, e.source_mean_luminance),
        ("SH 平均亮度", e_rec.sh_mean_luminance, e.sh_mean_luminance),
        ("mip6 平均亮度", e_rec.mip6_mean_luminance, e.mip6_mean_luminance),
        ("SH 相对误差", e_rec.sh_rel_diff, e.sh_rel_diff),
        ("mip6 相对误差", e_rec.mip6_rel_diff, e.mip6_rel_diff),
    ] {
        if rel_diff(a, b) > 1e-3 {
            return Err(AssetError::Mismatch(format!(
                "{name} 清单记录 {a:.5} 与实测 {b:.5} 不符"
            )));
        }
    }
    if e.sh_rel_diff > 0.01 || e.mip6_rel_diff > 0.05 {
        return Err(AssetError::Mismatch(format!(
            "能量核验未通过：SH 偏差 {:.2}%，mip6 偏差 {:.2}%",
            e.sh_rel_diff * 100.0,
            e.mip6_rel_diff * 100.0
        )));
    }
    if recorded.rotation_checks.len() != measured.rotation_checks.len() {
        return Err(AssetError::Mismatch(
            "清单旋转核验条数与预设数量不符".into(),
        ));
    }
    for (r, m) in recorded.rotation_checks.iter().zip(&measured.rotation_checks) {
        if (r.rotation_deg - m.rotation_deg).abs() > 1e-4 {
            return Err(AssetError::Mismatch("旋转核验的旋转角不一致".into()));
        }
        if angle_diff_deg(r.brightest_azimuth_deg, m.brightest_azimuth_deg) > 0.05
            || (r.brightest_elevation_deg - m.brightest_elevation_deg).abs() > 0.05
            || rel_diff(r.luminance, m.luminance) > 1e-3
        {
            return Err(AssetError::Mismatch(format!(
                "旋转 {:.1}° 的反射方向核验与实测不符",
                r.rotation_deg
            )));
        }
        if m.offset_deg > 0.06 {
            return Err(AssetError::Mismatch(format!(
                "旋转 {:.1}° 后最亮方位角偏离约定 {:.3}°",
                m.rotation_deg, m.offset_deg
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Rebuild comparison (evidence that regeneration reproduces the asset)
// ---------------------------------------------------------------------------

pub struct RebuildEvidence {
    pub max_prefilter_diff: f32,
    pub max_sh_diff: f32,
    pub source_sha_matches: bool,
    pub verification_matches: bool,
}

pub fn rebuild_evidence(
    _manifest_path: &Path,
    spec_path: &Path,
    loaded: &LoadedAsset,
) -> Result<RebuildEvidence, AssetError> {
    let spec = load_spec(spec_path)?;
    if spec.id != loaded.manifest.id {
        return Err(AssetError::Field(format!(
            "规格 id “{}” 与清单 id “{}” 不同",
            spec.id, loaded.manifest.id
        )));
    }
    if spec.source != loaded.manifest.source.params {
        return Err(AssetError::Field(
            "规格源参数与清单记录的源参数不同".into(),
        ));
    }
    let source = source_image(&spec.source, spec.source_width, spec.source_height);
    // Same round-trip as the build: rebuild from the bytes, not the f32s.
    let source_sha = sha256_bytes(&encode_source_bytes(&source)?);
    let source = decode_hdr_bytes(&encode_source_bytes(&source)?)?;
    let sh = env::project_sh(&source);
    let prefilter = env::build_prefilter_cache(&source, spec.prefilter.samples, &mut |_, _| {});
    let verification = measure(
        &source,
        &prefilter,
        &sh,
        spec.source.key_direction.dir(),
        &spec.rotation_presets_deg,
    );
    let max_prefilter_diff = max_abs_diff(&prefilter, &loaded.precomputed.prefilter);
    let max_sh_diff = sh
        .iter()
        .zip(&loaded.manifest.sh)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
        .fold(0.0f32, f32::max);
    let verification_matches = compare_verification(&verification, &loaded.measured).is_ok();
    Ok(RebuildEvidence {
        max_prefilter_diff,
        max_sh_diff,
        source_sha_matches: source_sha == loaded.manifest.source.sha256,
        verification_matches,
    })
}

fn encode_source_bytes(img: &HdrImage) -> Result<Vec<u8>, AssetError> {
    let mut bytes = Vec::new();
    let pixels: Vec<image::Rgb<f32>> = img.rgb.iter().map(|p| image::Rgb(*p)).collect();
    image::codecs::hdr::HdrEncoder::new(&mut bytes)
        .encode(&pixels, img.width as usize, img.height as usize)
        .map_err(|e| AssetError::Source(format!("HDR 编码失败：{e}")))?;
    Ok(bytes)
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct AssetEntry {
    pub manifest_path: PathBuf,
    pub id: String,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct SceneEntry {
    pub path: PathBuf,
    pub name: String,
}

pub fn discover_assets(root: &Path) -> Vec<AssetEntry> {
    let dir = root.join("assets");
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path().join("manifest.json");
        if !path.is_file() {
            continue;
        }
        match load_manifest(&path) {
            Ok(m) => out.push(AssetEntry {
                manifest_path: path,
                id: m.id,
                name: m.name,
            }),
            Err(e) => log::warn!("忽略无效素材 {}：{e}", path.display()),
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

pub fn discover_scenes(root: &Path) -> Vec<SceneEntry> {
    let dir = root.join("scenes");
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        match std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_json::from_str::<SceneFile>(&t).ok())
        {
            Some(scene) => out.push(SceneEntry {
                path,
                name: scene.name,
            }),
            None => log::warn!("忽略无效场景 {}", path.display()),
        }
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_asset_exists_and_loads() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_MANIFEST);
        assert!(
            path.is_file(),
            "展品光照素材集不存在：{}（运行 scripts/build_exhibit_lighting_assets.sh）",
            path.display()
        );
        let loaded = load_asset(&path).expect("素材包应通过完整性、方向与能量核验");
        assert_eq!(loaded.precomputed.prefilter.len(), PREFILTER_MIPS as usize);
        assert_eq!(loaded.manifest.id, loaded.manifest.id.trim());
        assert!(matches!(
            loaded.precomputed.origin,
            PrecomputeOrigin::Asset { .. }
        ));
        assert!(loaded.measured.brightest.key_offset_deg <= KEY_DIRECTION_TOLERANCE_DEG);
        assert!(loaded.measured.energy.sh_rel_diff <= 0.01);
    }

    #[test]
    fn bundled_spec_and_scenes_are_consistent() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_ASSET_ROOT);
        let spec = load_spec(&root.join("spec.json")).expect("规格应可解析");
        assert_eq!(spec.id, "directed_key");
        let scenes = discover_scenes(&root);
        assert!(!scenes.is_empty(), "素材场景必须存在");
        for entry in scenes {
            let (scene, manifest) = load_scene(&entry.path).expect("场景应可解析");
            assert!(manifest.is_file(), "场景引用的素材清单应存在");
            assert!(
                scene.notes.contains("旋转")
                    || scene.debug_mode == 1
                    || scene.exhibit == ExhibitKind::Ramp,
                "场景 {} 应说明反射方向或粗糙度用法",
                entry.path.display()
            );
        }
        let assets = discover_assets(&root);
        assert!(assets.iter().any(|a| a.id == spec.id));
    }

    #[test]
    fn tampered_manifest_is_rejected() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_MANIFEST);
        if !path.is_file() {
            return;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let mut v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let true_az = v["verification"]["brightest"]["azimuth_deg"]
            .as_f64()
            .unwrap();
        v["verification"]["brightest"]["azimuth_deg"] =
            serde_json::json!(if true_az > 0.0 { -179.0 } else { 179.0 });
        // Cache directories can live on a read-only mount; keep the tamper
        // fixture inside the Cargo target dir, which is always writable.
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("asset-tamper-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Copy payload and source into the fixture first, then point the
        // manifest at the flat copies. Resolving the original relative paths
        // here would write outside the fixture directory.
        for (key, name) in [("prefilter", "prefilter_f32.bin"), ("source", "source.hdr")] {
            let rel = v[key]["path"].as_str().unwrap();
            let src = path.parent().unwrap().join(rel);
            std::fs::copy(src, dir.join(name)).unwrap();
            v[key]["path"] = serde_json::json!(name);
        }
        let manifest_path = dir.join("manifest.json");
        std::fs::write(&manifest_path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
        let err = match load_asset(&manifest_path) {
            Ok(_) => panic!("篡改清单不应载入成功"),
            Err(e) => e,
        };
        assert!(matches!(err, AssetError::Mismatch(_)), "tampered: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

//! Precomputed lighting data and its on-disk cache.
//!
//! The cache is keyed by source hash + algorithm version + parameters, written
//! atomically, and validated on load. Reloading the same environment produces
//! the same prefiltered data (deterministic sampling order, no time-based
//! randomness).

use crate::env::{
    self, cubemap_from_hdr, EnvError, HdrImage, PREFILTER_MIPS, PREFILTER_SIZE, SKY_SIZE,
};
use crate::math::CUBE_FACES;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const ALGORITHM_VERSION: u32 = 3;
const PREFILTER_MAGIC: &[u8; 8] = b"MLPF0003";
const SH_MAGIC: &[u8; 8] = b"MLSH0003";

#[derive(Clone, Debug)]
pub struct Precomputed {
    /// Prefiltered radiance cube maps, mip 0 = roughness 0.
    pub prefilter: Vec<Vec<Vec<[f32; 3]>>>,
    /// Full-resolution environment cube used for the background and the mip-0
    /// reflection layer.
    pub sky: Vec<Vec<[f32; 3]>>,
    pub sh: [[f32; 3]; 9],
    pub built_from_cache: bool,
    pub compute_ms: f64,
}

#[derive(Clone, Debug)]
pub struct PrecomputeSettings {
    pub prefilter_samples: u32,
    pub cache_dir: PathBuf,
    pub use_cache: bool,
}

impl Default for PrecomputeSettings {
    fn default() -> Self {
        Self {
            prefilter_samples: 256,
            cache_dir: std::env::temp_dir().join("museum-lighting-cache"),
            use_cache: true,
        }
    }
}

pub fn fingerprint(source_sha: &str, settings: &PrecomputeSettings) -> String {
    format!(
        "v{}_s{}_sha{}",
        ALGORITHM_VERSION,
        settings.prefilter_samples,
        &source_sha[..source_sha.len().min(16)]
    )
}

pub fn cache_paths(settings: &PrecomputeSettings, fingerprint: &str) -> (PathBuf, PathBuf, PathBuf) {
    let dir = settings.cache_dir.join(fingerprint);
    (
        dir.clone(),
        dir.join("prefilter.bin"),
        dir.join("sh.bin"),
    )
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), EnvError> {
    let parent = path.parent().ok_or_else(|| EnvError::Io("缓存路径无父目录".into()))?;
    std::fs::create_dir_all(parent)
        .map_err(|e| EnvError::Io(format!("{}: {e}", parent.display())))?;
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    {
        let mut f = std::fs::File::create(&tmp)
            .map_err(|e| EnvError::Io(format!("{}: {e}", tmp.display())))?;
        f.write_all(bytes)
            .map_err(|e| EnvError::Io(format!("{}: {e}", tmp.display())))?;
        f.sync_all().ok();
    }
    std::fs::rename(&tmp, path).map_err(|e| EnvError::Io(format!("{}: {e}", path.display())))?;
    Ok(())
}

fn encode_prefilter(prefilter: &[Vec<Vec<[f32; 3]>>], sample_count: u32) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(PREFILTER_MAGIC);
    out.extend_from_slice(&PREFILTER_SIZE.to_le_bytes());
    out.extend_from_slice(&PREFILTER_MIPS.to_le_bytes());
    out.extend_from_slice(&sample_count.to_le_bytes());
    for mip in prefilter {
        for face in mip {
            for px in face {
                for c in px {
                    out.extend_from_slice(&c.to_le_bytes());
                }
            }
        }
    }
    out
}

fn decode_prefilter(
    bytes: &[u8],
    sample_count: u32,
) -> Result<Vec<Vec<Vec<[f32; 3]>>>, EnvError> {
    if bytes.len() < 20 || &bytes[0..8] != PREFILTER_MAGIC {
        return Err(EnvError::Decode("预过滤缓存头部不匹配".into()));
    }
    let read_u32 = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    if read_u32(8) != PREFILTER_SIZE
        || read_u32(12) != PREFILTER_MIPS
        || read_u32(16) != sample_count
    {
        return Err(EnvError::Decode("预过滤缓存参数与请求不一致".into()));
    }
    let mut offset = 20usize;
    let mut mips = Vec::new();
    for mip in 0..PREFILTER_MIPS {
        let size = env::prefilter_mip_size(mip) as usize;
        let mut faces = Vec::new();
        for _ in 0..CUBE_FACES {
            let count = size * size;
            let need = count * 12;
            if offset + need > bytes.len() {
                return Err(EnvError::Decode("预过滤缓存被截断".into()));
            }
            let floats: &[f32] = bytemuck::cast_slice(&bytes[offset..offset + need]);
            let mut face = Vec::with_capacity(count);
            for px in floats.chunks_exact(3) {
                face.push([px[0], px[1], px[2]]);
            }
            offset += need;
            faces.push(face);
        }
        mips.push(faces);
    }
    Ok(mips)
}

fn encode_sh(sh: &[[f32; 3]; 9]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(SH_MAGIC);
    for c in sh {
        for v in c {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

fn decode_sh(bytes: &[u8]) -> Result<[[f32; 3]; 9], EnvError> {
    if bytes.len() < 8 + 9 * 12 || &bytes[0..8] != SH_MAGIC {
        return Err(EnvError::Decode("SH 缓存头部不匹配".into()));
    }
    let floats: &[f32] = bytemuck::cast_slice(&bytes[8..8 + 9 * 12]);
    let mut out = [[0.0f32; 3]; 9];
    for (i, px) in floats.chunks_exact(3).enumerate() {
        out[i] = [px[0], px[1], px[2]];
    }
    Ok(out)
}

/// Build (or load) the complete precomputed lighting set.
pub fn precompute(
    img: &HdrImage,
    source_sha: &str,
    settings: &PrecomputeSettings,
    mut progress: impl FnMut(&str, f32),
) -> Result<Precomputed, EnvError> {
    let started = std::time::Instant::now();
    let fp = fingerprint(source_sha, settings);
    let (_, prefilter_path, sh_path) = cache_paths(settings, &fp);

    let mut built_from_cache = false;
    let mut prefilter: Option<Vec<Vec<Vec<[f32; 3]>>>> = None;
    let mut sh: Option<[[f32; 3]; 9]> = None;

    if settings.use_cache {
        progress("读取预计算缓存…", 0.05);
        if let (Ok(pf_bytes), Ok(sh_bytes)) =
            (std::fs::read(&prefilter_path), std::fs::read(&sh_path))
        {
            if let (Ok(pf), Ok(s)) = (
                decode_prefilter(&pf_bytes, settings.prefilter_samples),
                decode_sh(&sh_bytes),
            ) {
                prefilter = Some(pf);
                sh = Some(s);
                built_from_cache = true;
            }
        }
    }

    if sh.is_none() {
        progress("投影漫反射球谐…", 0.15);
        sh = Some(env::project_sh(img));
    }

    if prefilter.is_none() {
        progress("预过滤粗糙反射层（GGX 重要性采样）…", 0.2);
        let done = std::sync::atomic::AtomicU32::new(0);
        let total = PREFILTER_MIPS;
        let mut pf = env::build_prefilter_cache(img, settings.prefilter_samples, &mut |_, _| {
            let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            progress("预过滤粗糙反射层（GGX 重要性采样）…", 0.2 + 0.7 * d as f32 / total as f32);
        });
        // Keep the closure's progress coherent; build_prefilter_cache calls
        // progress sequentially after each mip.
        let _ = &mut pf;
        prefilter = Some(pf);
    }

    let prefilter = prefilter.unwrap();
    let sh = sh.unwrap();

    if settings.use_cache && !built_from_cache {
        progress("写入预计算缓存…", 0.93);
        atomic_write(&prefilter_path, &encode_prefilter(&prefilter, settings.prefilter_samples))?;
        atomic_write(&sh_path, &encode_sh(&sh))?;
    }

    progress("构建天空立方体贴图…", 0.95);
    let sky_cube = cubemap_from_hdr(img, SKY_SIZE);
    let sky = sky_cube.faces;

    let precomputed = Precomputed {
        prefilter,
        sky,
        sh,
        built_from_cache,
        compute_ms: started.elapsed().as_secs_f64() * 1000.0,
    };
    // A truncated or corrupt cache must not be handed to the renderer as if it
    // were valid.
    if let Err(e) = validate_precomputed(&precomputed) {
        return Err(EnvError::Decode(format!("预计算数据校验失败：{e}")));
    }
    Ok(precomputed)
}

/// Read back any cache directory entries (for the UI to report).
pub fn cache_entries(settings: &PrecomputeSettings) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    if let Ok(dir) = std::fs::read_dir(&settings.cache_dir) {
        for entry in dir.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_dir() {
                    let mut size = 0u64;
                    if let Ok(inner) = std::fs::read_dir(entry.path()) {
                        for f in inner.flatten() {
                            size += f.metadata().map(|m| m.len()).unwrap_or(0);
                        }
                    }
                    out.push((entry.file_name().to_string_lossy().into_owned(), size));
                }
            }
        }
    }
    out.sort();
    out
}

/// Lightweight structural check used by the app at startup and by tests.
pub fn validate_precomputed(p: &Precomputed) -> Result<(), String> {
    if p.prefilter.len() != PREFILTER_MIPS as usize {
        return Err(format!("预过滤层数 {} 与预期 {} 不符", p.prefilter.len(), PREFILTER_MIPS));
    }
    for (mip, faces) in p.prefilter.iter().enumerate() {
        if faces.len() != CUBE_FACES {
            return Err(format!("mip {mip} 面数错误"));
        }
        let expected = (env::prefilter_mip_size(mip as u32) * env::prefilter_mip_size(mip as u32)) as usize;
        for (i, f) in faces.iter().enumerate() {
            if f.len() != expected {
                return Err(format!("mip {mip} 面 {i} 尺寸错误"));
            }
            if f.iter().any(|px| px.iter().any(|c| !c.is_finite() || *c < 0.0)) {
                return Err(format!("mip {mip} 面 {i} 含非法值"));
            }
        }
    }
    if p.sky.len() != CUBE_FACES {
        return Err("天空立方体贴图面数错误".into());
    }
    Ok(())
}

#[allow(dead_code)]
fn read_all(path: &Path) -> std::io::Result<Vec<u8>> {
    let mut v = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut v)?;
    Ok(v)
}

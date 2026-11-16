//! Command-line front end for exhibit-lighting assets. This is an auxiliary
//! tool: the primary entry point for viewing lighting is the graphical
//! preview window (`museum-lighting` with no arguments). These subcommands
//! only build, verify, and list the asset packages that the GUI loads.

use crate::asset::{self, DEFAULT_ASSET_ROOT, DEFAULT_MANIFEST, DEFAULT_SPEC};
use crate::cache::PrecomputeSettings;
use std::path::{Path, PathBuf};

pub fn usage() -> &'static str {
    "博物馆展品环境光预览器 — 素材工具（图形预览请直接运行 museum-lighting）

用法：
  museum-lighting                          打开交互预览窗口（图形入口），默认载入内置素材包
  museum-lighting --env <hdr>              用指定 HDR 打开窗口
  museum-lighting --asset <manifest.json>  用素材包打开窗口
  museum-lighting --scene <scene.json>     载入场景文件打开窗口
  museum-lighting --verify                 无窗口校验默认环境
  museum-lighting asset build [--spec <spec.json>] [--root <目录>]
                                           从规格重建源 HDR、预过滤数据与场景
  museum-lighting asset verify [--manifest <manifest.json>] [--spec <spec.json>]
                                           载入素材并复核：源哈希、预过滤哈希、
                                           反射方向、能量、旋转预设、重建一致性
  museum-lighting asset list [--root <目录>]
                                           列出素材与场景

素材默认位置：
  规格   assets/exhibit_lighting/spec.json
  素材   assets/exhibit_lighting/assets/directed_key/manifest.json
  场景   assets/exhibit_lighting/scenes/*.scene.json
"
}

pub fn run(args: &[String]) -> anyhow::Result<()> {
    // `args` starts at the "asset" subcommand itself.
    let mut it = args.iter().skip(1);
    let sub = it
        .next()
        .map(|s| s.as_str())
        .unwrap_or("list");
    let mut spec = PathBuf::from(DEFAULT_SPEC);
    let mut manifest = PathBuf::from(DEFAULT_MANIFEST);
    let mut root = PathBuf::from(DEFAULT_ASSET_ROOT);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--spec" => {
                if let Some(p) = it.next() {
                    spec = PathBuf::from(p);
                }
            }
            "--manifest" => {
                if let Some(p) = it.next() {
                    manifest = PathBuf::from(p);
                }
            }
            "--root" => {
                if let Some(p) = it.next() {
                    root = PathBuf::from(p);
                }
            }
            other => anyhow::bail!("未知参数“{other}”（--help 查看用法）"),
        }
    }
    match sub {
        "build" => build(&spec, &root),
        "verify" => verify(&manifest, &spec),
        "list" => list(&root),
        other => anyhow::bail!("未知素材子命令“{other}”（支持 build / verify / list）"),
    }
}

fn build(spec_path: &Path, root: &Path) -> anyhow::Result<()> {
    let report = asset::build_asset(spec_path, root, |msg, p| {
        eprintln!("[{:3.0}%] {msg}", p * 100.0);
    })?;
    let m = &report.manifest;
    println!("素材 {}（{}）已写入", m.id, m.name);
    println!("  源环境：{}", report.source_path.display());
    println!("  预过滤：{}", report.prefilter_path.display());
    println!("  清单：{}", report.manifest_path.display());
    println!(
        "  最亮方向：方位角 {:.2}°，仰角 {:.2}°（与主光偏 {:.2}°），亮度 {:.2}",
        m.verification.brightest.azimuth_deg,
        m.verification.brightest.elevation_deg,
        m.verification.brightest.key_offset_deg,
        m.verification.brightest.luminance
    );
    println!(
        "  能量：源 {:.5} / SH {:.5}（偏差 {:.3}%）/ mip6 {:.5}（偏差 {:.3}%）",
        m.verification.energy.source_mean_luminance,
        m.verification.energy.sh_mean_luminance,
        m.verification.energy.sh_rel_diff * 100.0,
        m.verification.energy.mip6_mean_luminance,
        m.verification.energy.mip6_rel_diff * 100.0
    );
    for r in &m.verification.rotation_checks {
        println!(
            "  旋转 {:+7.1}° → 最亮方位角 {:+7.2}°（期望 {:+7.2}°）",
            r.rotation_deg, r.brightest_azimuth_deg, r.expected_azimuth_deg
        );
    }
    println!(
        "  写入/读回最大绝对差：{:.3e}",
        report.max_rebuild_diff
    );
    Ok(())
}

fn verify(manifest_path: &Path, spec_path: &Path) -> anyhow::Result<()> {
    let loaded = asset::load_asset(manifest_path)?;
    println!(
        "素材 {}（{}）载入 {:.1} ms",
        loaded.manifest.id, loaded.manifest.name, loaded.load_ms
    );
    println!(
        "  源环境 SHA-256 一致：{}（{}×{}）",
        &loaded.manifest.source.sha256[..16],
        loaded.manifest.source.width,
        loaded.manifest.source.height
    );
    println!(
        "  预过滤 SHA-256 一致：{}（{} 字节）",
        &loaded.manifest.prefilter.sha256[..16],
        loaded.manifest.prefilter.byte_len
    );
    let m = &loaded.measured;
    println!(
        "  实测最亮方向：方位角 {:.2}°，仰角 {:.2}°（与规格主光偏 {:.2}°）",
        m.brightest.azimuth_deg, m.brightest.elevation_deg, m.brightest.key_offset_deg
    );
    println!(
        "  实测辐亮度：{:.3}, {:.3}, {:.3}（亮度 {:.3}）",
        m.brightest.radiance[0],
        m.brightest.radiance[1],
        m.brightest.radiance[2],
        m.brightest.luminance
    );
    println!(
        "  能量：源 {:.5} / SH {:.5}（偏差 {:.3}%）/ mip6 {:.5}（偏差 {:.3}%）",
        m.energy.source_mean_luminance,
        m.energy.sh_mean_luminance,
        m.energy.sh_rel_diff * 100.0,
        m.energy.mip6_mean_luminance,
        m.energy.mip6_rel_diff * 100.0
    );
    for r in &m.rotation_checks {
        println!(
            "  旋转 {:+7.1}° → 最亮方位角 {:+7.2}°（期望 {:+7.2}°，偏差 {:.3}°）",
            r.rotation_deg, r.brightest_azimuth_deg, r.expected_azimuth_deg, r.offset_deg
        );
    }
    let evidence = asset::rebuild_evidence(manifest_path, spec_path, &loaded)?;
    println!(
        "  重建一致性：源哈希 {}，预过滤最大绝对差 {:.3e}，SH 最大绝对差 {:.3e}，核验 {}",
        if evidence.source_sha_matches { "相同" } else { "不同" },
        evidence.max_prefilter_diff,
        evidence.max_sh_diff,
        if evidence.verification_matches {
            "一致"
        } else {
            "不一致"
        }
    );
    if !evidence.source_sha_matches || evidence.max_prefilter_diff > 1e-6 {
        anyhow::bail!("重建结果与素材包不一致：素材包不是由当前规格生成的");
    }
    // Exercise the same loader the GUI uses for every scene, so a scene that
    // points at a missing or tampered asset fails here too.
    let root = manifest_path
        .ancestors()
        .nth(3)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_ASSET_ROOT));
    let mut scene_count = 0;
    for entry in asset::discover_scenes(&root) {
        let (scene, scene_manifest) = asset::load_scene(&entry.path)?;
        let scene_loaded = asset::load_asset(&scene_manifest)?;
        println!(
            "  场景 {}：{} → 素材 {}，环境旋转 {:+.1}°，展品 {}",
            entry.path.display(),
            scene.name,
            scene_loaded.manifest.id,
            scene.env_rotation_deg,
            scene.exhibit.label()
        );
        scene_count += 1;
    }
    if scene_count == 0 {
        anyhow::bail!("素材根目录没有可载入的场景：{}", root.display());
    }
    let settings = PrecomputeSettings::default();
    println!(
        "  预览内核：缓存目录 {}（仅 --verify 使用，图形预览优先载入素材包）",
        settings.cache_dir.display()
    );
    println!("素材与场景核验通过");
    Ok(())
}

fn list(root: &Path) -> anyhow::Result<()> {
    let assets = asset::discover_assets(root);
    if assets.is_empty() {
        println!("{} 下没有素材包", root.display());
    }
    for a in &assets {
        match asset::load_asset(&a.manifest_path) {
            Ok(loaded) => println!(
                "素材 {} — {}（{}，主光方位角 {:.2}°）",
                a.id,
                a.name,
                loaded.manifest_path.display(),
                loaded.measured.brightest.azimuth_deg
            ),
            Err(e) => println!("素材 {} — {}（无法载入：{e}）", a.id, a.name),
        }
    }
    for s in asset::discover_scenes(root) {
        println!("场景 {} — {}", s.path.display(), s.name);
    }
    Ok(())
}

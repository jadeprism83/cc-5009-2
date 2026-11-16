//! Museum exhibit ambient-lighting previewer.
//!
//! The interactive window is driven by the exhibit-lighting kernel
//! (`lighting` + `cache` + `reference`); the GPU renderer mirrors the same
//! equations. Environment-load errors are recoverable and surface in the
//! lighting panel, never leaving a stale environment labelled as a new one.
//!
//! `--verify` runs the same kernel headlessly, writes verification images and
//! a report, then exits.

mod asset;
mod asset_cli;
mod cache;
mod env;
mod geometry;
mod lighting;
mod math;
mod reference;
mod renderer;
mod shaders;

use cache::{PrecomputeSettings, Precomputed};
use env::{EnvError, EnvMeta, HdrImage, TestPattern};
use geometry::{ExhibitKind, Material};
use lighting::Camera;
use math::{spherical_azimuth_elevation, CUBE_FACES};
use renderer::{FrameParams, SceneRenderer};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

const APP_TITLE: &str = "博物馆展品环境光预览器";
const DEFAULT_ENV: &str = "assets/env/studio_small_09_2k.hdr";
const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/wqy-microhei.ttc");

// ---------------------------------------------------------------------------
// Background environment loading
// ---------------------------------------------------------------------------

enum EnvJob {
    Load(PathBuf),
    Generate(TestPattern),
    Asset(PathBuf),
}

struct EnvResult {
    source_name: String,
    image: Option<HdrImage>,
    precomputed: Option<Precomputed>,
    meta: Option<EnvMeta>,
    error: Option<String>,
    /// Present when the job loaded a bundled asset; the panel shows its
    /// recomputed checks and the loader's own verification.
    asset: Option<Box<asset::LoadedAsset>>,
}

struct EnvWorker {
    tx: Sender<EnvJob>,
    rx: Receiver<EnvResult>,
    progress: Arc<Mutex<(String, f32, bool)>>,
}

impl EnvWorker {
    fn new(settings: PrecomputeSettings) -> Self {
        let (job_tx, job_rx) = std::sync::mpsc::channel::<EnvJob>();
        let (res_tx, res_rx) = std::sync::mpsc::channel::<EnvResult>();
        let progress = Arc::new(Mutex::new((String::new(), 0.0f32, false)));
        let prog = progress.clone();
        std::thread::spawn(move || {
            while let Ok(job) = job_rx.recv() {
                let result = match job {
                    EnvJob::Load(path) => run_load(&path, &settings, &prog),
                    EnvJob::Generate(pattern) => run_generate(pattern, &settings, &prog),
                    EnvJob::Asset(path) => run_asset(&path),
                };
                let _ = res_tx.send(result);
            }
        });
        Self {
            tx: job_tx,
            rx: res_rx,
            progress,
        }
    }

    fn request(&self, job: EnvJob, label: String) -> Result<(), String> {
        self.tx
            .send(job)
            .map_err(|_| "环境工作线程已停止".to_string())?;
        *self.progress.lock().unwrap() = (format!("排队：{label}"), 0.0, true);
        Ok(())
    }

    fn poll(&self) -> Option<EnvResult> {
        match self.rx.try_recv() {
            Ok(r) => {
                if let Ok(mut p) = self.progress.lock() {
                    p.2 = false;
                }
                Some(r)
            }
            Err(_) => None,
        }
    }

    fn progress(&self) -> (String, f32, bool) {
        self.progress.lock().map(|p| p.clone()).unwrap_or_default()
    }
}

fn progress_fn(prog: &Arc<Mutex<(String, f32, bool)>>) -> impl FnMut(&str, f32) + '_ {
    move |msg, value| {
        if let Ok(mut p) = prog.lock() {
            p.0 = msg.to_string();
            p.1 = value;
        }
    }
}

fn run_load(
    path: &Path,
    settings: &PrecomputeSettings,
    prog: &Arc<Mutex<(String, f32, bool)>>,
) -> EnvResult {
    let fail = |error: EnvError, source: String| EnvResult {
        source_name: source,
        image: None,
        precomputed: None,
        meta: None,
        error: Some(error.to_string()),
        asset: None,
    };
    if let Err(e) = env::validate_format(path) {
        return fail(e, path.display().to_string());
    }
    let image = match env::load_hdr(path) {
        Ok(i) => i,
        Err(e) => return fail(e, path.display().to_string()),
    };
    let meta = env::env_meta(
        &image,
        env::file_sha256(path).unwrap_or_default(),
        path.file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string()),
    );
    let mut progress = progress_fn(prog);
    match cache::precompute(&image, &meta.sha256, settings, &mut progress) {
        Ok(p) => EnvResult {
            source_name: path.display().to_string(),
            image: Some(image),
            precomputed: Some(p),
            meta: Some(meta),
            error: None,
            asset: None,
        },
        Err(e) => fail(e, path.display().to_string()),
    }
}

fn run_generate(
    pattern: TestPattern,
    settings: &PrecomputeSettings,
    prog: &Arc<Mutex<(String, f32, bool)>>,
) -> EnvResult {
    let label = test_pattern_label(pattern);
    let image = env::synthetic_hdr(pattern, 512, 256);
    let meta = env::env_meta(&image, format!("synthetic:{pattern:?}"), label.to_string());
    let mut progress = progress_fn(prog);
    match cache::precompute(&image, &meta.sha256, settings, &mut progress) {
        Ok(p) => EnvResult {
            source_name: label.to_string(),
            image: Some(image),
            precomputed: Some(p),
            meta: Some(meta),
            error: None,
            asset: None,
        },
        Err(e) => EnvResult {
            source_name: label.to_string(),
            image: None,
            precomputed: None,
            meta: None,
            error: Some(e.to_string()),
            asset: None,
        },
    }
}

fn run_asset(manifest: &Path) -> EnvResult {
    match asset::load_asset(manifest) {
        Ok(loaded) => EnvResult {
            source_name: format!("素材 {}（{}）", loaded.manifest.id, loaded.manifest.name),
            image: Some(loaded.image.clone()),
            precomputed: Some(loaded.precomputed.clone()),
            meta: Some(loaded.meta.clone()),
            error: None,
            asset: Some(Box::new(loaded)),
        },
        Err(e) => EnvResult {
            source_name: manifest.display().to_string(),
            image: None,
            precomputed: None,
            meta: None,
            error: Some(format!("素材载入失败：{e}")),
            asset: None,
        },
    }
}

fn test_pattern_label(pattern: TestPattern) -> &'static str {
    match pattern {
        TestPattern::SingleLobe => "合成测试图：单个亮区",
        TestPattern::Uniform => "合成测试图：均匀亮度",
        TestPattern::DirectionMarkers => "合成测试图：方向标记",
        TestPattern::Grid => "合成测试图：经纬网格",
    }
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

struct ReferenceView {
    width: u32,
    height: u32,
    texture: Option<egui::TextureHandle>,
    elapsed_ms: f64,
    brightest_dir: glam::Vec3,
}

struct AppState {
    settings: PrecomputeSettings,
    env: Option<EnvMeta>,
    env_source: String,
    precomputed: Option<Precomputed>,
    image: Option<HdrImage>,

    exhibit: ExhibitKind,
    exhibit_rotation_deg: f32,
    env_rotation_deg: f32,
    exposure: f32,
    gamma_only: bool,
    debug_mode: u32,
    forced_lod: Option<f32>,
    show_ground: bool,
    show_panorama: bool,
    panorama_texture: Option<egui::TextureId>,

    camera: Camera,
    scene_rect: Option<egui::Rect>,
    pointer_in_scene: Option<egui::Pos2>,

    cpu_encode_ms: f64,
    gpu_scene_ms: Option<f64>,
    gpu_timing_note: String,
    precompute_ms: Option<f64>,
    precompute_from_cache: bool,
    source_elapsed_ms: f64,
    last_error: Option<String>,
    last_info: Option<String>,
    env_path_input: String,
    /// Path of the currently displayed asset manifest, if any. Asset loads are
    /// explicit: the panel shows which facts were recomputed at load time.
    asset_manifest: Option<PathBuf>,
    asset_status: Option<asset::LoadedAsset>,
    asset_choices: Vec<asset::AssetEntry>,
    asset_choice: Option<PathBuf>,
    scene_choices: Vec<asset::SceneEntry>,
    scene_choice: Option<PathBuf>,

    reference: Option<ReferenceView>,
    reference_auto: bool,
    reference_dirty: bool,
    probe_text: Option<String>,
    verification: Vec<String>,
}

impl AppState {
    fn new(settings: PrecomputeSettings) -> Self {
        Self {
            settings,
            env: None,
            env_source: String::new(),
            precomputed: None,
            image: None,
            exhibit: ExhibitKind::Pottery,
            exhibit_rotation_deg: 0.0,
            env_rotation_deg: 0.0,
            exposure: 0.55,
            gamma_only: false,
            debug_mode: 0,
            forced_lod: None,
            show_ground: true,
            show_panorama: false,
            panorama_texture: None,
            camera: Camera::default(),
            scene_rect: None,
            pointer_in_scene: None,
            cpu_encode_ms: 0.0,
            gpu_scene_ms: None,
            gpu_timing_note: String::new(),
            precompute_ms: None,
            precompute_from_cache: false,
            source_elapsed_ms: 0.0,
            last_error: None,
            last_info: None,
            env_path_input: DEFAULT_ENV.to_string(),
            asset_manifest: None,
            asset_status: None,
            asset_choices: asset::discover_assets(Path::new(asset::DEFAULT_ASSET_ROOT)),
            asset_choice: None,
            scene_choices: asset::discover_scenes(Path::new(asset::DEFAULT_ASSET_ROOT)),
            scene_choice: None,
            reference: None,
            reference_auto: true,
            reference_dirty: true,
            probe_text: None,
            verification: Vec::new(),
        }
    }

    fn material(&self) -> Material {
        geometry::material_for(self.exhibit)
    }

    fn env_rotation(&self) -> f32 {
        self.env_rotation_deg.to_radians()
    }

    fn frame_params(&self) -> FrameParams {
        FrameParams {
            camera: self.camera,
            exhibit: self.exhibit,
            exhibit_rotation: self.exhibit_rotation_deg.to_radians(),
            material: self.material(),
            env_rotation: self.env_rotation(),
            exposure: self.exposure,
            gamma_only: self.gamma_only,
            debug_mode: self.debug_mode,
            forced_lod: self.forced_lod,
            show_ground: self.show_ground,
        }
    }

    fn apply_env_result(&mut self, result: EnvResult) {
        if let Some(error) = result.error {
            // Recoverable: the previous environment stays, but it is explicitly
            // named as unchanged in the message. No stale data is relabelled.
            let current = if self.env_source.is_empty() {
                "无环境".to_string()
            } else {
                self.env_source.clone()
            };
            self.last_error = Some(format!("{error}（当前仍显示：{current}）"));
            self.last_info = None;
            return;
        }
        self.last_error = None;
        self.env = result.meta.clone();
        self.env_source = result.source_name.clone();
        self.source_elapsed_ms = 0.0;
        if let Some(p) = &result.precomputed {
            self.precompute_ms = Some(p.compute_ms);
            self.precompute_from_cache = matches!(p.origin, cache::PrecomputeOrigin::DiskCache);
        }
        self.precomputed = result.precomputed;
        self.image = result.image;
        self.asset_status = result.asset.map(|b| *b);
        self.asset_manifest = self
            .asset_status
            .as_ref()
            .map(|a| a.manifest_path.clone());
        if let Some(choice) = &self.asset_manifest {
            self.asset_choice = Some(choice.clone());
        }
        self.reference_dirty = true;
        self.verification.clear();
        self.last_info = Some(match &self.asset_status {
            Some(a) => format!(
                "已载入素材 {}：最亮方向方位角 {:.2}°、仰角 {:.2}°（与主光偏 {:.2}°）；能量偏差 SH {:.2}% / mip6 {:.2}%；载入耗时 {:.0} ms",
                a.manifest.id,
                a.measured.brightest.azimuth_deg,
                a.measured.brightest.elevation_deg,
                a.measured.brightest.key_offset_deg,
                a.measured.energy.sh_rel_diff * 100.0,
                a.measured.energy.mip6_rel_diff * 100.0,
                a.load_ms
            ),
            None => format!(
                "已载入 {}（预计算 {:.1} ms，{}）",
                self.env_source,
                self.precompute_ms.unwrap_or(0.0),
                self.precomputed
                    .as_ref()
                    .map(|p| p.origin.label())
                    .unwrap_or_default()
            ),
        });
    }

    fn brightest(&self) -> Option<(glam::Vec3, glam::Vec3)> {
        self.precomputed
            .as_ref()
            .map(|p| lighting::brightest_in_cache(&p.prefilter, self.env_rotation()))
    }

    fn reference_settings(&self, width: u32, height: u32) -> reference::ReferenceSettings {
        reference::ReferenceSettings {
            width,
            height,
            camera: self.camera,
            exhibit: self.exhibit,
            material: self.material(),
            exhibit_rotation: self.exhibit_rotation_deg.to_radians(),
            env_rotation: self.env_rotation(),
            exposure: self.exposure,
            forced_roughness: self.forced_lod,
        }
    }
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

fn setup_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    let data = egui::FontData::from_owned(FONT_BYTES.to_vec());
    fonts
        .font_data
        .insert("wqy-microhei".to_owned(), Arc::new(data));
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .insert(0, "wqy-microhei".to_owned());
    }
    ctx.set_fonts(fonts);
}

fn dir_label(dir: glam::Vec3) -> String {
    let (az, el) = spherical_azimuth_elevation(dir);
    format!("方位角 {az:+.1}°，仰角 {el:+.1}°")
}

enum UiAction {
    Load(PathBuf),
    LoadAsset(PathBuf),
    LoadScene(PathBuf),
    Generate(TestPattern),
    SaveScreenshot,
    Verify,
    VerifyAsset,
    RebuildReference,
}

fn lighting_panel(ctx: &egui::Context, state: &mut AppState, progress: (String, f32, bool)) -> Vec<UiAction> {
    let mut actions = Vec::new();
    let (progress_msg, progress_val, running) = progress;
    egui::SidePanel::left("lighting_panel")
        .resizable(true)
        .default_width(340.0)
        .show(ctx, |ui| {
            ui.heading(APP_TITLE);
            ui.small("所有控件均由展品光照内核驱动；失败信息回报到本面板");
            ui.separator();

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.strong("展品");
                ui.label(egui::RichText::new(state.exhibit.description()).small());
            });
            ui.horizontal_wrapped(|ui| {
                for kind in ExhibitKind::ALL {
                    ui.selectable_value(&mut state.exhibit, kind, kind.label());
                }
            });
            ui.horizontal(|ui| {
                ui.label("展品世界旋转");
                if ui
                    .add(
                        egui::DragValue::new(&mut state.exhibit_rotation_deg)
                            .speed(0.4)
                            .range(-180.0..=180.0)
                            .suffix("°"),
                    )
                    .changed()
                {
                    state.reference_dirty = true;
                }
                ui.small("只影响模型矩阵");
            });

            ui.separator();
            ui.heading("环境光");
            ui.horizontal(|ui| {
                ui.label("环境旋转");
                if ui
                    .add(
                        egui::DragValue::new(&mut state.env_rotation_deg)
                            .speed(0.4)
                            .range(-180.0..=180.0)
                            .suffix("°"),
                    )
                    .changed()
                {
                    state.reference_dirty = true;
                }
            });
            ui.small("在场景中横向拖动即可旋转环境；背景与反射同步变化");
            if ui
                .checkbox(&mut state.show_panorama, "显示环境全景（可拖动旋转）")
                .changed()
            {
                state.reference_dirty = true;
            }

            ui.separator();
            ui.heading("光照数据");
            if let Some(meta) = &state.env {
                egui::Grid::new("env_meta").num_columns(2).show(ui, |ui| {
                    ui.label("来源");
                    ui.label(&state.env_source);
                    ui.end_row();
                    ui.label("尺寸");
                    ui.label(format!("{}×{}", meta.width, meta.height));
                    ui.end_row();
                    ui.label("平均亮度（线性）");
                    ui.label(format!("{:.4}", meta.mean_luminance));
                    ui.end_row();
                    ui.label("最大亮度（线性）");
                    ui.label(format!("{:.1}", meta.max_luminance));
                    ui.end_row();
                    ui.label("动态范围");
                    ui.label(format!("{:.0}:1", meta.dynamic_range));
                    ui.end_row();
                });
            } else {
                ui.label("尚无环境数据");
            }
            if let Some((dir, radiance)) = state.brightest() {
                ui.label(format!("最亮方向：{}", dir_label(dir)));
                ui.label(format!(
                    "缓存辐亮度：{:.3}, {:.3}, {:.3}",
                    radiance.x, radiance.y, radiance.z
                ));
                ui.small("由预过滤缓存 mip 0 实际数据求得，随环境旋转实时更新");
            }
            if let Some(p) = &state.precomputed {
                ui.label(format!(
                    "预过滤 {} 层；mip→粗糙度 = mip / {}",
                    p.prefilter.len(),
                    p.prefilter.len() - 1
                ));
                ui.label(format!(
                    "预计算：{:.1} ms（{}）",
                    state.precompute_ms.unwrap_or(p.compute_ms),
                    if state.precompute_from_cache {
                        "缓存命中"
                    } else {
                        "本次重建并已写入缓存"
                    }
                ));
                for (name, size) in cache::cache_entries(&state.settings) {
                    ui.small(format!("缓存 {name} · {:.1} MB", size as f64 / 1e6));
                }
            }

            ui.separator();
            ui.heading("显示映射");
            ui.small("照明在线性空间计算；这里只决定线性 HDR 如何映射到屏幕");
            let changed = ui
                .add(
                    egui::Slider::new(&mut state.exposure, 0.01..=8.0)
                        .logarithmic(true)
                        .text("曝光"),
                )
                .changed();
            let changed = ui
                .checkbox(
                    &mut state.gamma_only,
                    "固定线性截断（仅 gamma 编码，不做色调映射）",
                )
                .changed()
                || changed;
            if changed {
                state.reference_dirty = true;
            }
            let mut forced_enabled = state.forced_lod.is_some();
            if ui.checkbox(&mut forced_enabled, "强制 mip 层").changed() {
                state.forced_lod = forced_enabled.then_some(0.0);
                state.reference_dirty = true;
            }
            if let Some(lod) = &mut state.forced_lod {
                if ui
                    .add(egui::Slider::new(lod, 0.0..=6.0).text("mip"))
                    .changed()
                {
                    state.reference_dirty = true;
                }
                ui.small(format!("对应粗糙度 {:.3}", *lod / 6.0));
            }
            ui.horizontal(|ui| {
                ui.label("调试通道");
                egui::ComboBox::from_id_salt("debug_mode")
                    .selected_text(DEBUG_MODES[state.debug_mode.min(6) as usize])
                    .show_ui(ui, |ui| {
                        for (i, name) in DEBUG_MODES.iter().enumerate() {
                            ui.selectable_value(&mut state.debug_mode, i as u32, *name);
                        }
                    });
            });
            ui.checkbox(&mut state.show_ground, "显示地面");

            ui.separator();
            ui.heading("耗时（分别统计）");
            ui.label(format!(
                "预计算：{}",
                match state.precompute_ms {
                    Some(ms) => format!("{ms:.1} ms"),
                    None => "—".to_string(),
                }
            ));
            ui.label(format!("每帧 CPU 编码：{:.2} ms", state.cpu_encode_ms));
            ui.label(match state.gpu_scene_ms {
                Some(ms) => format!("每帧 GPU 场景绘制：{ms:.2} ms"),
                None => "每帧 GPU 场景绘制：等待时间戳结果…".to_string(),
            });
            ui.small(&state.gpu_timing_note);

            ui.separator();
            ui.heading("参考渲染（CPU 独立实现）");
            ui.label(format!(
                "480×360，耗时 {:.0} ms",
                state
                    .reference
                    .as_ref()
                    .map(|r| r.elapsed_ms)
                    .unwrap_or(0.0)
            ));
            ui.checkbox(&mut state.reference_auto, "参数变化时自动重建");
            if ui.button("立即重建参考图").clicked() {
                actions.push(UiAction::RebuildReference);
            }
            if let Some(view) = &state.reference {
                if let Some(tex) = &view.texture {
                    let avail = ui.available_width().min(320.0);
                    let scale = avail / view.width as f32;
                    ui.image(egui::load::SizedTexture::new(
                        tex.id(),
                        egui::Vec2::new(view.width as f32 * scale, view.height as f32 * scale),
                    ));
                }
                ui.small(format!(
                    "参考图最亮方向：{}",
                    dir_label(view.brightest_dir)
                ));
            }
            if let Some(text) = &state.probe_text {
                ui.separator();
                ui.label("反射探针（鼠标指向场景读取）");
                ui.label(egui::RichText::new(text).monospace().small());
            }

            ui.separator();
            ui.heading("展品光照素材集");
            ui.small("素材包内含源 HDR、预过滤数据与核验清单；载入时重新计算并核对方向与能量");
            if state.asset_choices.is_empty() {
                ui.colored_label(
                    egui::Color32::from_rgb(240, 180, 100),
                    format!(
                        "未发现素材包（{}）。可运行：\n\
                         museum-lighting asset build",
                        asset::DEFAULT_ASSET_ROOT
                    ),
                );
            } else {
                let current = state
                    .asset_choice
                    .as_ref()
                    .and_then(|p| {
                        state
                            .asset_choices
                            .iter()
                            .find(|a| &a.manifest_path == p)
                    })
                    .map(|a| format!("{} — {}", a.id, a.name))
                    .unwrap_or_else(|| "选择素材…".to_string());
                egui::ComboBox::from_id_salt("asset_choice")
                    .selected_text(current)
                    .width(280.0)
                    .show_ui(ui, |ui| {
                        for a in &state.asset_choices {
                            let selected = state
                                .asset_choice
                                .as_ref()
                                .map(|p| p == &a.manifest_path)
                                .unwrap_or(false);
                            if ui
                                .selectable_label(
                                    selected,
                                    format!("{} — {}", a.id, a.name),
                                )
                                .clicked()
                            {
                                actions.push(UiAction::LoadAsset(a.manifest_path.clone()));
                            }
                        }
                    });
            }
            if let Some(loaded) = &state.asset_status {
                let m = &loaded.manifest;
                let v = &loaded.measured;
                egui::Grid::new("asset_meta").num_columns(2).show(ui, |ui| {
                    ui.label("素材");
                    ui.label(format!("{}（{}）", m.id, m.name));
                    ui.end_row();
                    ui.label("源环境");
                    ui.label(format!(
                        "{} · {}×{}",
                        m.source.path, m.source.width, m.source.height
                    ));
                    ui.end_row();
                    ui.label("源 SHA-256");
                    ui.label(
                        egui::RichText::new(&m.source.sha256[..16]).monospace().small(),
                    );
                    ui.end_row();
                    ui.label("预过滤");
                    ui.label(format!(
                        "{} mip · {} 面 · 采样 {} · {} MB",
                        m.prefilter.mips,
                        CUBE_FACES,
                        m.prefilter.samples,
                        m.prefilter.byte_len as f64 / 1e6
                    ));
                    ui.end_row();
                    ui.label("实测最亮方向");
                    ui.label(format!(
                        "方位角 {:+.2}°，仰角 {:+.2}°",
                        v.brightest.azimuth_deg, v.brightest.elevation_deg
                    ));
                    ui.end_row();
                    ui.label("与规格主光");
                    ui.label(format!("偏 {:.2}°（阈值 {:.1}°）", v.brightest.key_offset_deg, asset::KEY_DIRECTION_TOLERANCE_DEG));
                    ui.end_row();
                    ui.label("主光辐亮度");
                    ui.label(format!(
                        "{:.2}, {:.2}, {:.2}（亮度 {:.2}）",
                        v.brightest.radiance[0],
                        v.brightest.radiance[1],
                        v.brightest.radiance[2],
                        v.brightest.luminance
                    ));
                    ui.end_row();
                    ui.label("能量核验");
                    ui.label(format!(
                        "源 {:.4} / SH {:.4}（{:.2}%）/ mip6 {:.4}（{:.2}%）",
                        v.energy.source_mean_luminance,
                        v.energy.sh_mean_luminance,
                        v.energy.sh_rel_diff * 100.0,
                        v.energy.mip6_mean_luminance,
                        v.energy.mip6_rel_diff * 100.0
                    ));
                    ui.end_row();
                });
                ui.small("以上数值由载入时的预过滤数据与源环境重新计算，不读清单里的记录值");
                egui::CollapsingHeader::new("旋转核验（反射朝向，随环境旋转）")
                    .default_open(false)
                    .show(ui, |ui| {
                        for r in &v.rotation_checks {
                            ui.small(format!(
                                "旋转 {:+7.1}° → 实测最亮方位角 {:+7.2}°，期望 {:+7.2}°（偏差 {:.3}°）",
                                r.rotation_deg,
                                r.brightest_azimuth_deg,
                                r.expected_azimuth_deg,
                                r.offset_deg
                            ));
                        }
                    });
                egui::CollapsingHeader::new("来源与参数")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.small(format!("说明：{}", m.notes));
                        ui.small(format!("生成器：{}", m.source.generator));
                        ui.small(format!(
                            "主光：辐亮度 {:.2}, {:.2}, {:.2}，方位角 {:.1}°，仰角 {:.1}°，半径 {:.1}°",
                            m.source.params.key_radiance[0],
                            m.source.params.key_radiance[1],
                            m.source.params.key_radiance[2],
                            m.source.params.key_direction.azimuth_deg,
                            m.source.params.key_direction.elevation_deg,
                            m.source.params.key_radius_deg
                        ));
                        ui.small(format!(
                            "暗环境：基底 {:.4}, {:.4}, {:.4}；补光 {:.3}, {:.3}, {:.3}",
                            m.source.params.base_radiance[0],
                            m.source.params.base_radiance[1],
                            m.source.params.base_radiance[2],
                            m.source.params.fill_radiance[0],
                            m.source.params.fill_radiance[1],
                            m.source.params.fill_radiance[2]
                        ));
                        ui.small(format!("粗糙度约定：{}", m.prefilter.roughness_mapping));
                        ui.small(format!(
                            "方向约定：{}；{}；{}",
                            m.conventions.up_axis,
                            m.conventions.azimuth,
                            m.conventions.elevation
                        ));
                        ui.small(format!("环境旋转：{}", m.conventions.environment_rotation));
                    });
            }
            if !state.scene_choices.is_empty() {
                ui.add_space(4.0);
                ui.label("场景（图形入口，可直接打开）");
                let current = state
                    .scene_choice
                    .as_ref()
                    .and_then(|p| state.scene_choices.iter().find(|s| &s.path == p))
                    .map(|s| s.name.clone())
                    .unwrap_or_else(|| "选择场景…".to_string());
                egui::ComboBox::from_id_salt("scene_choice")
                    .selected_text(current)
                    .width(280.0)
                    .show_ui(ui, |ui| {
                        for s in &state.scene_choices {
                            let selected = state
                                .scene_choice
                                .as_ref()
                                .map(|p| p == &s.path)
                                .unwrap_or(false);
                            if ui.selectable_label(selected, &s.name).clicked() {
                                actions.push(UiAction::LoadScene(s.path.clone()));
                            }
                        }
                    });
                ui.small("场景同时设置环境旋转、展品与调试通道，便于对照反射方向");
            }

            ui.separator();
            ui.heading("环境文件");
            ui.horizontal(|ui| {
                if ui.button("重新载入默认环境").clicked() {
                    actions.push(UiAction::Load(PathBuf::from(DEFAULT_ENV)));
                }
            });
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut state.env_path_input)
                        .desired_width(220.0)
                        .hint_text("环境文件路径（.hdr）"),
                );
                if ui.button("载入").clicked() {
                    actions.push(UiAction::Load(PathBuf::from(
                        state.env_path_input.trim().to_string(),
                    )));
                }
            });
            ui.small("不支持的尺寸或格式会返回可恢复错误，不会冒用新文件名");
            ui.horizontal_wrapped(|ui| {
                for (pattern, label) in [
                    (TestPattern::SingleLobe, "单亮区"),
                    (TestPattern::DirectionMarkers, "方向标记"),
                    (TestPattern::Uniform, "均匀亮度"),
                    (TestPattern::Grid, "经纬网格"),
                ] {
                    if ui.button(label).clicked() {
                        actions.push(UiAction::Generate(pattern));
                    }
                }
            });
            ui.small("测试图用于验证反射朝向、立方体接缝与粗糙层变化");

            if running {
                ui.add(egui::ProgressBar::new(progress_val).text(progress_msg));
            }
            if let Some(err) = &state.last_error {
                ui.separator();
                ui.colored_label(egui::Color32::from_rgb(240, 120, 100), err);
            }
            if let Some(info) = &state.last_info {
                ui.small(info);
            }

            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("保存画面 PNG").clicked() {
                    actions.push(UiAction::SaveScreenshot);
                }
                if ui.button("运行校验").clicked() {
                    actions.push(UiAction::Verify);
                }
                if state.asset_status.is_some()
                    && ui
                        .button("复核素材（按规格重建比对）")
                        .on_hover_text(
                            "从规格重新生成源环境与预过滤数据，与当前素材包逐值比较",
                        )
                        .clicked()
                {
                    actions.push(UiAction::VerifyAsset);
                }
            });
            if !state.verification.is_empty() {
                ui.collapsing("校验结果", |ui| {
                    for line in &state.verification {
                        ui.small(line);
                    }
                });
            }

            ui.separator();
            ui.collapsing("如何启动界面", |ui| {
                ui.small(
                    "构建：cargo build --release\n\
                     启动：./target/release/museum-lighting\n\
                     指定环境：--env path/to/env.hdr\n\
                     无显示环境：Xvfb :99 & DISPLAY=:99 后启动\n\
                     软件 Vulkan：设置 VK_ICD_FILENAMES 指向 lvp.json，\
                     并让 LD_LIBRARY_PATH 包含其依赖库\n\n\
                     场景中：按住左键横向拖动旋转环境光；按住右键或中键拖动\
                     可环绕相机；左侧面板切换展品、曝光与调试通道。",
                );
            });
                });
        });
    actions
}

const DEBUG_MODES: [&str; 7] = [
    "完整着色",
    "预过滤层原始 HDR",
    "漫反射项",
    "粗糙度",
    "法线",
    "高光项",
    "背景方向",
];

// ---------------------------------------------------------------------------
// Window application
// ---------------------------------------------------------------------------

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    egui_renderer: egui_wgpu::Renderer,
    egui_state: egui_winit::State,
    egui_ctx: egui::Context,
}

struct App {
    window: Option<Arc<winit::window::Window>>,
    gpu: Option<GpuContext>,
    renderer: Option<SceneRenderer>,
    state: AppState,
    worker: EnvWorker,
    screenshot_pending: bool,
}

impl App {
    fn new(settings: PrecomputeSettings, initial_env: PathBuf) -> Self {
        let worker = EnvWorker::new(settings.clone());
        let _ = worker.request(
            EnvJob::Load(initial_env.clone()),
            initial_env.display().to_string(),
        );
        Self {
            window: None,
            gpu: None,
            renderer: None,
            state: AppState::new(settings),
            worker,
            screenshot_pending: false,
        }
    }

    /// Start with a bundled asset package. The default path is the packaged
    /// `directed_key` asset; when a scene is given its preview settings are
    /// applied as soon as the window and GPU are up.
    fn new_asset(
        settings: PrecomputeSettings,
        manifest: PathBuf,
        scene: Option<PathBuf>,
    ) -> Self {
        let worker = EnvWorker::new(settings.clone());
        let _ = worker.request(
            EnvJob::Asset(manifest.clone()),
            manifest.display().to_string(),
        );
        let mut state = AppState::new(settings);
        state.asset_manifest = Some(manifest.clone());
        state.asset_choice = Some(manifest.clone());
        if let Some(scene_path) = scene {
            if let Ok((scene, _)) = asset::load_scene(&scene_path) {
                state.exhibit = scene.exhibit;
                state.exhibit_rotation_deg = scene.exhibit_rotation_deg;
                state.env_rotation_deg = scene.env_rotation_deg;
                state.exposure = scene.exposure;
                state.debug_mode = scene.debug_mode.min(6);
                state.forced_lod = scene.forced_lod.map(|l| l.clamp(0.0, 6.0));
                state.show_ground = scene.show_ground;
                state.scene_choice = Some(scene_path);
            }
        }
        Self {
            window: None,
            gpu: None,
            renderer: None,
            state,
            worker,
            screenshot_pending: false,
        }
    }

    fn init_gpu(&mut self, window: Arc<winit::window::Window>) {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = match instance.create_surface(window.clone()) {
            Ok(s) => s,
            Err(e) => {
                self.state.last_error = Some(format!("无法创建窗口表面：{e}"));
                return;
            }
        };
        let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })) {
            Ok(a) => a,
            Err(e) => {
                self.state.last_error = Some(format!("无法获取 GPU 适配器：{e}"));
                return;
            }
        };
        let caps = surface.get_capabilities(&adapter);
        let present_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: present_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        let features = adapter.features()
            & (wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        let (device, queue) = match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("museum device"),
            required_features: features,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        })) {
            Ok(dq) => dq,
            Err(e) => {
                self.state.last_error = Some(format!("无法创建设备：{e}"));
                return;
            }
        };
        surface.configure(&device, &config);
        let egui_ctx = egui::Context::default();
        setup_fonts(&egui_ctx);
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            Some(2048),
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, present_format, None, 1, false);
        log::info!("GPU 适配器：{}", adapter.get_info().name);
        self.gpu = Some(GpuContext {
            device,
            queue,
            surface,
            config,
            egui_renderer,
            egui_state,
            egui_ctx,
        });
    }

    fn ensure_scene_renderer(&mut self) {
        if self.renderer.is_some() {
            return;
        }
        let (Some(gpu), Some(precomputed)) = (&mut self.gpu, &self.state.precomputed) else {
            return;
        };
        let mut renderer = SceneRenderer::new(
            &gpu.device,
            &gpu.queue,
            precomputed,
            gpu.config.format,
        );
        let size = self
            .window
            .as_ref()
            .map(|w| w.inner_size())
            .unwrap_or_default();
        renderer.ensure_target(&gpu.device, size.width.max(1), size.height.max(1));
        if let Some(view) = renderer.target_view_for_egui() {
            let id = gpu
                .egui_renderer
                .register_native_texture(&gpu.device, view, wgpu::FilterMode::Linear);
            renderer.set_target_egui_id(id);
        }
        let pano_view = renderer.panorama_display_view().clone();
        let pano_id = gpu
            .egui_renderer
            .register_native_texture(&gpu.device, &pano_view, wgpu::FilterMode::Linear);
        renderer.set_panorama_egui_id(pano_id);
        self.state.panorama_texture = Some(pano_id);
        self.state.gpu_timing_note = renderer.timestamp_note.clone();
        self.renderer = Some(renderer);
    }

    fn rebuild_reference(&mut self) {
        let (Some(precomputed), Some(gpu)) = (&self.state.precomputed, &self.gpu) else {
            return;
        };
        let settings = self.state.reference_settings(480, 360);
        let img = reference::render(precomputed, &settings);
        let color = egui::ColorImage::from_rgb([img.width as usize, img.height as usize], &img.rgb8);
        let handle = gpu
            .egui_ctx
            .load_texture("reference", color, egui::TextureOptions::LINEAR);
        self.state.reference = Some(ReferenceView {
            width: img.width,
            height: img.height,
            texture: Some(handle),
            elapsed_ms: img.elapsed_ms,
            brightest_dir: img.brightest_dir,
        });
        self.state.reference_dirty = false;
    }

    fn update_probe(&mut self) {
        let (Some(precomputed), Some(pos), Some(rect)) = (
            &self.state.precomputed,
            self.state.pointer_in_scene,
            self.state.scene_rect,
        ) else {
            return;
        };
        if rect.width() < 2.0 {
            return;
        }
        let ndc_x = ((pos.x - rect.left()) / rect.width()) * 2.0 - 1.0;
        let ndc_y = 1.0 - ((pos.y - rect.top()) / rect.height()) * 2.0;
        let settings = self.state.reference_settings(480, 360);
        self.state.probe_text = reference::probe(precomputed, &settings, ndc_x, ndc_y).map(|p| {
            let lum = p
                .env_radiance
                .dot(glam::Vec3::new(0.2126, 0.7152, 0.0722));
            format!(
                "位置 {:.2},{:.2},{:.2}\n法线 {:.2},{:.2},{:.2}\n反射方向 {}\n粗糙度 {:.3} → mip {:.2}\n缓存辐亮度 {:.3},{:.3},{:.3}（亮度 {:.3}）\n内核着色 {:.3},{:.3},{:.3}",
                p.position.x,
                p.position.y,
                p.position.z,
                p.normal.x,
                p.normal.y,
                p.normal.z,
                dir_label(p.reflected_dir),
                p.roughness,
                p.lod,
                p.env_radiance.x,
                p.env_radiance.y,
                p.env_radiance.z,
                lum,
                p.shaded.x,
                p.shaded.y,
                p.shaded.z
            )
        });
    }

    fn save_screenshot(&mut self) {
        let (Some(gpu), Some(renderer)) = (&self.gpu, &self.renderer) else {
            return;
        };
        let _ = std::fs::create_dir_all("artifacts");
        if let Some((w, h, rgba)) = renderer.read_display_rgba(&gpu.device, &gpu.queue) {
            if let Some(buf) = image::RgbaImage::from_raw(w, h, rgba) {
                match buf.save("artifacts/scene.png") {
                    Ok(()) => self.state.last_info = Some("已保存 artifacts/scene.png".into()),
                    Err(e) => self.state.last_error = Some(format!("保存画面失败：{e}")),
                }
            }
        }
        if let Some((w, h, rgba)) = renderer.read_panorama_rgba(&gpu.device, &gpu.queue) {
            if let Some(buf) = image::RgbaImage::from_raw(w, h, rgba) {
                let _ = buf.save("artifacts/panorama.png");
            }
        }
    }

    fn handle_actions(&mut self, actions: Vec<UiAction>) {
        for action in actions {
            match action {
                UiAction::Load(path) => {
                    self.state.env_path_input = path.display().to_string();
                    if let Err(e) = self
                        .worker
                        .request(EnvJob::Load(path.clone()), path.display().to_string())
                    {
                        self.state.last_error = Some(e);
                    } else {
                        self.state.last_info =
                            Some(format!("正在载入 {} …", path.display()));
                        self.state.last_error = None;
                    }
                }
                UiAction::LoadAsset(path) => {
                    if let Err(e) = self
                        .worker
                        .request(EnvJob::Asset(path.clone()), path.display().to_string())
                    {
                        self.state.last_error = Some(e);
                    } else {
                        self.state.asset_manifest = Some(path.clone());
                        self.state.asset_choice = Some(path.clone());
                        self.state.last_info =
                            Some(format!("正在载入素材 {} …", path.display()));
                        self.state.last_error = None;
                    }
                }
                UiAction::LoadScene(path) => {
                    match asset::load_scene(&path) {
                        Ok((scene, manifest)) => {
                            self.state.scene_choice = Some(path.clone());
                            self.state.asset_choice = Some(manifest.clone());
                            if let Err(e) = self
                                .worker
                                .request(EnvJob::Asset(manifest), scene.name.clone())
                            {
                                self.state.last_error = Some(e);
                            } else {
                                self.state.exhibit = scene.exhibit;
                                self.state.exhibit_rotation_deg = scene.exhibit_rotation_deg;
                                self.state.env_rotation_deg = scene.env_rotation_deg;
                                self.state.exposure = scene.exposure;
                                self.state.debug_mode = scene.debug_mode.min(6);
                                self.state.forced_lod =
                                    scene.forced_lod.map(|l| l.clamp(0.0, 6.0));
                                self.state.show_ground = scene.show_ground;
                                self.state.reference_dirty = true;
                                self.state.last_info =
                                    Some(format!("正在载入场景 {} …", scene.name));
                                self.state.last_error = None;
                            }
                        }
                        Err(e) => {
                            let current = if self.state.env_source.is_empty() {
                                "无环境".to_string()
                            } else {
                                self.state.env_source.clone()
                            };
                            self.state.last_error = Some(format!(
                                "场景载入失败：{e}（当前仍显示：{current}）"
                            ));
                        }
                    }
                }
                UiAction::Generate(pattern) => {
                    let label = test_pattern_label(pattern).to_string();
                    if let Err(e) = self.worker.request(EnvJob::Generate(pattern), label.clone()) {
                        self.state.last_error = Some(e);
                    } else {
                        self.state.last_info = Some(format!("正在生成 {label} …"));
                        self.state.last_error = None;
                    }
                }
                UiAction::SaveScreenshot => self.screenshot_pending = true,
                UiAction::Verify => {
                    if let Some(p) = &self.state.precomputed {
                        let mut lines =
                            verify_kernel(p, self.state.env_rotation(), &self.state.settings);
                        if let Some(a) = &self.state.asset_status {
                            let v = &a.measured;
                            lines.push(format!(
                                "素材 {} 实测：最亮方位角 {:+.2}°、仰角 {:+.2}°（偏主光 {:.2}°）；SH 偏差 {:.3}%，mip6 偏差 {:.3}%",
                                a.manifest.id,
                                v.brightest.azimuth_deg,
                                v.brightest.elevation_deg,
                                v.brightest.key_offset_deg,
                                v.energy.sh_rel_diff * 100.0,
                                v.energy.mip6_rel_diff * 100.0
                            ));
                        }
                        self.state.verification = lines;
                    }
                }
                UiAction::VerifyAsset => {
                    let Some(a) = self.state.asset_status.clone() else {
                        self.state.last_error =
                            Some("当前不是素材包；先在「展品光照素材集」中载入一个素材".into());
                        continue;
                    };
                    let spec = Path::new(asset::DEFAULT_SPEC);
                    match asset::rebuild_evidence(&a.manifest_path, spec, &a) {
                        Ok(ev) => {
                            let mut lines = vec![format!(
                                "素材 {} 重建复核：源哈希 {}，预过滤最大绝对差 {:.3e}，SH 最大绝对差 {:.3e}，核验记录{}",
                                a.manifest.id,
                                if ev.source_sha_matches { "相同" } else { "不同" },
                                ev.max_prefilter_diff,
                                ev.max_sh_diff,
                                if ev.verification_matches { "一致" } else { "不一致" }
                            )];
                            for r in &a.measured.rotation_checks {
                                lines.push(format!(
                                    "旋转 {:+7.1}° → 最亮方位角 {:+7.2}°，期望 {:+7.2}°，偏差 {:.3}°",
                                    r.rotation_deg,
                                    r.brightest_azimuth_deg,
                                    r.expected_azimuth_deg,
                                    r.offset_deg
                                ));
                            }
                            if !ev.source_sha_matches || ev.max_prefilter_diff > 1e-6 {
                                self.state.last_error = Some(format!(
                                    "重建结果与素材包不一致：素材包不是由 {} 生成的",
                                    spec.display()
                                ));
                            } else {
                                self.state.last_info =
                                    Some("素材重建复核通过：素材包与当前规格逐值一致".into());
                                self.state.last_error = None;
                            }
                            self.state.verification = lines;
                        }
                        Err(e) => {
                            self.state.last_error = Some(format!("素材重建复核失败：{e}"));
                        }
                    }
                }
                UiAction::RebuildReference => {
                    self.state.reference = None;
                    self.rebuild_reference();
                }
            }
        }
    }

    fn render(&mut self) {
        let Some(window) = self.window.clone() else { return };
        if self.gpu.is_none() {
            return;
        }
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }
        {
            let gpu = self.gpu.as_mut().unwrap();
            if gpu.config.width != size.width || gpu.config.height != size.height {
                gpu.config.width = size.width;
                gpu.config.height = size.height;
                gpu.surface.configure(&gpu.device, &gpu.config);
            }
        }

        // Apply finished background work before building the UI.
        while let Some(result) = self.worker.poll() {
            let loaded = result.error.is_none();
            self.state.apply_env_result(result);
            if loaded {
                // Upload the new cube maps into the live renderer. Without this
                // the UI would show the new file name while the scene still
                // used the previous environment.
                if let (Some(renderer), Some(gpu), Some(precomputed)) = (
                    self.renderer.as_mut(),
                    self.gpu.as_ref(),
                    self.state.precomputed.as_ref(),
                ) {
                    renderer.update_environment(&gpu.queue, precomputed);
                }
            }
            self.ensure_scene_renderer();
        }

        let raw_input = self
            .gpu
            .as_mut()
            .unwrap()
            .egui_state
            .take_egui_input(&window);
        let egui_ctx = self.gpu.as_ref().unwrap().egui_ctx.clone();
        let mut actions = Vec::new();
        let panorama_id = self.renderer.as_ref().and_then(|r| r.panorama_egui_id());
        let display_id = self.renderer.as_ref().and_then(|r| r.display_egui_id());
        let progress = self.worker.progress();
        let state = &mut self.state;
        let full_output = egui_ctx.run(raw_input, |ctx| {
            actions.extend(lighting_panel(ctx, state, progress.clone()));

            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ctx, |ui| {
                    let rect = ui.max_rect();
                    if let Some(id) = display_id {
                        let (rect, response) =
                            ui.allocate_exact_size(rect.size(), egui::Sense::click_and_drag());
                        ui.painter().image(
                            id,
                            rect,
                            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                            egui::Color32::WHITE,
                        );
                        state.scene_rect = Some(rect);
                        state.pointer_in_scene =
                            response.hovered().then(|| response.hover_pos()).flatten();
                        if response.dragged_by(egui::PointerButton::Primary) {
                            let delta = response.drag_delta();
                            if delta.x.abs() > 0.0 {
                                state.env_rotation_deg += delta.x * 0.35;
                                state.reference_dirty = true;
                            }
                        }
                        if response.dragged_by(egui::PointerButton::Secondary)
                            || response.dragged_by(egui::PointerButton::Middle)
                        {
                            let delta = response.drag_delta();
                            state.camera.yaw += delta.x * 0.004;
                            state.camera.pitch =
                                (state.camera.pitch - delta.y * 0.004).clamp(-1.3, 1.3);
                            state.reference_dirty = true;
                        }
                    } else {
                        ui.centered_and_justified(|ui| {
                            if state.last_error.is_none() {
                                ui.spinner();
                                ui.label("正在准备环境与预计算…");
                            }
                        });
                    }
                });

            if state.show_panorama {
                if let Some(id) = panorama_id {
                    egui::Window::new("环境全景（拖动旋转）")
                        .default_pos([400.0, 60.0])
                        .default_size([620.0, 340.0])
                        .show(ctx, |ui| {
                            let width = ui.available_width();
                            let response = ui.add(
                                egui::Image::new(egui::load::SizedTexture::new(
                                    id,
                                    egui::Vec2::new(width, width * 0.5),
                                ))
                                .sense(egui::Sense::drag()),
                            );
                            if response.dragged() {
                                state.env_rotation_deg += response.drag_delta().x * 0.35;
                                state.reference_dirty = true;
                            }
                        });
                }
            }
        });

        if self.screenshot_pending {
            self.screenshot_pending = false;
            self.save_screenshot();
        }
        if self.state.reference_auto
            && self.state.reference_dirty
            && self.state.precomputed.is_some()
        {
            self.rebuild_reference();
        }
        self.update_probe();

        let clipped = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let frame_params = self.state.frame_params();
        {
            let gpu = self.gpu.as_mut().unwrap();
            let screen = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [gpu.config.width, gpu.config.height],
                pixels_per_point: full_output.pixels_per_point,
            };
            // Size the offscreen scene to the central panel's actual pixel rect,
            // so the exhibit is not stretched to the full window while the UI
            // reserves part of it for the side panel.
            if let Some(rect) = self.state.scene_rect {
                if let Some(renderer) = &mut self.renderer {
                    let ppp = full_output.pixels_per_point.max(0.5);
                    let w = (rect.width() * ppp).round().max(1.0) as u32;
                    let h = (rect.height() * ppp).round().max(1.0) as u32;
                    if renderer.ensure_target(&gpu.device, w, h) {
                        if let Some(view) = renderer.target_view_for_egui() {
                            match renderer.display_egui_id() {
                                Some(id) => gpu
                                    .egui_renderer
                                    .update_egui_texture_from_wgpu_texture(
                                        &gpu.device,
                                        view,
                                        wgpu::FilterMode::Linear,
                                        id,
                                    ),
                                None => {
                                    let id = gpu.egui_renderer.register_native_texture(
                                        &gpu.device,
                                        view,
                                        wgpu::FilterMode::Linear,
                                    );
                                    renderer.set_target_egui_id(id);
                                }
                            }
                        }
                    }
                }
            }
            for (id, image_delta) in &full_output.textures_delta.set {
                gpu.egui_renderer
                    .update_texture(&gpu.device, &gpu.queue, *id, image_delta);
            }
            let mut encoder =
                gpu.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("frame encoder"),
                    });
            if let Some(renderer) = &mut self.renderer {
                let timing =
                    renderer.encode(&gpu.device, &gpu.queue, &mut encoder, &frame_params);
                self.state.cpu_encode_ms = timing.cpu_encode_ms;
                self.state.gpu_scene_ms = timing.gpu_scene_ms;
            }
            let surface_tex = match gpu.surface.get_current_texture() {
                Ok(t) => t,
                Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    return;
                }
                Err(e) => {
                    self.state.last_error = Some(format!("获取窗口纹理失败：{e}"));
                    return;
                }
            };
            let view = surface_tex
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            if let Some(renderer) = &self.renderer {
                renderer.blit_to_surface(&mut encoder, &view, gpu.config.format);
            }
            {
                let user_cmds = gpu.egui_renderer.update_buffers(
                    &gpu.device,
                    &gpu.queue,
                    &mut encoder,
                    &clipped,
                    &screen,
                );
                let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                let mut pass = pass.forget_lifetime();
                gpu.egui_renderer.render(&mut pass, &clipped, &screen);
                drop(pass);
                gpu.queue.submit(user_cmds);
            }
            gpu.queue.submit([encoder.finish()]);
            surface_tex.present();
            if let Some(renderer) = &mut self.renderer {
                renderer.collect_timestamps(&gpu.device);
            }
            for id in &full_output.textures_delta.free {
                gpu.egui_renderer.free_texture(id);
            }
        }
        self.handle_actions(actions);
        window.request_redraw();
    }

    fn refresh_target_texture(&mut self) {
        let (Some(gpu), Some(renderer)) = (&mut self.gpu, &mut self.renderer) else {
            return;
        };
        if let Some(view) = renderer.target_view_for_egui() {
            if let Some(id) = renderer.display_egui_id() {
                gpu.egui_renderer.update_egui_texture_from_wgpu_texture(
                    &gpu.device,
                    view,
                    wgpu::FilterMode::Linear,
                    id,
                );
            } else {
                let id = gpu.egui_renderer.register_native_texture(
                    &gpu.device,
                    view,
                    wgpu::FilterMode::Linear,
                );
                renderer.set_target_egui_id(id);
            }
        }
    }
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = winit::window::WindowAttributes::default()
            .with_title(APP_TITLE)
            .with_inner_size(winit::dpi::LogicalSize::new(1380.0, 860.0));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("创建窗口失败：{e}");
                event_loop.exit();
                return;
            }
        };
        self.window = Some(window.clone());
        self.init_gpu(window);
        self.ensure_scene_renderer();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(window) = self.window.clone() else { return };
        if let Some(gpu) = &mut self.gpu {
            let response = gpu.egui_state.on_window_event(&window, &event);
            if response.repaint {
                window.request_redraw();
            }
            if response.consumed {
                return;
            }
        }
        match event {
            winit::event::WindowEvent::CloseRequested => event_loop.exit(),
            winit::event::WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    if size.width > 0 && size.height > 0 {
                        gpu.config.width = size.width;
                        gpu.config.height = size.height;
                        gpu.surface.configure(&gpu.device, &gpu.config);
                        if let Some(renderer) = &mut self.renderer {
                            if renderer.ensure_target(&gpu.device, size.width, size.height) {
                                self.refresh_target_texture();
                            }
                        }
                    }
                }
            }
            winit::event::WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
        self.ensure_scene_renderer();
    }
}

// ---------------------------------------------------------------------------
// Kernel checks shared by the UI "校验" button and `--verify`
// ---------------------------------------------------------------------------

fn verify_kernel(
    precomputed: &Precomputed,
    env_rotation: f32,
    _settings: &PrecomputeSettings,
) -> Vec<String> {
    let mut log = Vec::new();
    log.push(format!(
        "缓存：{} 层 mip × {} 面；mip→粗糙度 = mip / {}",
        precomputed.prefilter.len(),
        CUBE_FACES,
        precomputed.prefilter.len() - 1
    ));
    log.push(format!(
        "预计算来源：{}",
        precomputed.origin.label()
    ));
    let probe_dir = glam::Vec3::new(0.4, 0.55, -0.3).normalize();
    for (i, mip) in precomputed.prefilter.iter().enumerate() {
        let v = env::sample_cube_face(mip, probe_dir);
        log.push(format!(
            "mip {i}（粗糙度 {:.3}）：{:.4}, {:.4}, {:.4}",
            env::mip_roughness(i as u32),
            v.x,
            v.y,
            v.z
        ));
    }

    // Uniform environment: a constant reflection is only possible if the
    // prefilter preserves energy.
    let uniform = env::synthetic_hdr(TestPattern::Uniform, 128, 64);
    let uniform_sh = env::project_sh(&uniform);
    let irr = math::eval_sh9(&uniform_sh, glam::Vec3::Y);
    log.push(format!(
        "均匀环境（辐亮度 2）SH 辐照度：{:.4}（理论 πL = {:.4}）",
        irr.x,
        std::f32::consts::PI * 2.0
    ));

    // Single lobe orientation.
    let lobe = env::synthetic_hdr(TestPattern::SingleLobe, 256, 128);
    let cache = env::build_prefilter_cache(&lobe, 96, &mut |_, _| {});
    let (bright, _) = lighting::brightest_in_cache(&cache, 0.0);
    let expected = math::direction_from_azimuth_elevation(35.0, 25.0);
    log.push(format!(
        "单亮区反射朝向偏差：{:.2}°（阈值 10°）",
        bright.dot(expected).acos().to_degrees()
    ));

    // Seam continuity, measured against an analytic environment. The test
    // environment is a smooth function of direction, so cube edge texels agree
    // by construction and any boundary error is a sampling artifact. Near an
    // edge the bilinear footprint reaches into the neighbouring face; the
    // wrapping sampler must cross over, while a per-face clamped sampler would
    // duplicate the edge texel.
    let seam_size = 128u32;
    let f_truth = |d: glam::Vec3| {
        glam::Vec3::new(
            1.0 + 0.8 * (4.0 * d.x).sin(),
            1.0 + 0.6 * (3.0 * d.z + 1.0).cos(),
            1.0 + 0.5 * (2.0 * d.y).sin(),
        )
    };
    let seam_env = env::analytic_hdr(512, 256, f_truth);
    let seam_cube = env::cubemap_from_hdr(&seam_env, seam_size);
    let face_data: Vec<Vec<[f32; 3]>> = (0..CUBE_FACES).map(|f| seam_cube.face(f).to_vec()).collect();
    let inset = 0.25 / seam_size as f32;
    let mut err_seamless = 0.0f32;
    let mut err_clamped = 0.0f32;
    let mut err_interior = 0.0f32;
    for face in 0..CUBE_FACES {
        for k in 0..16 {
            let t = (k as f32 + 0.5) / 16.0;
            for (s, tt) in [
                (1.0 - inset, t),
                (inset, t),
                (t, 1.0 - inset),
                (t, inset),
            ] {
                let d = math::dir_from_cube_face(face, s, tt);
                let truth = f_truth(d);
                let scale = 1e-4 + truth.length();
                let good = env::sample_cube_face(&face_data, d);
                let bad = env::sample_cube_face_clamped(&face_data, d);
                err_seamless = err_seamless.max((good - truth).length() / scale);
                err_clamped = err_clamped.max((bad - truth).length() / scale);
            }
        }
        // Face interior: same sampler, far from any edge. If the edge error is
        // no larger than this, the boundary adds no discontinuity of its own.
        for i in 0..8 {
            for j in 0..8 {
                let s = 0.25 + 0.5 * (i as f32 + 0.5) / 8.0;
                let tt = 0.25 + 0.5 * (j as f32 + 0.5) / 8.0;
                let d = math::dir_from_cube_face(face, s, tt);
                let truth = f_truth(d);
                let scale = 1e-4 + truth.length();
                let v = env::sample_cube_face(&face_data, d);
                err_interior = err_interior.max((v - truth).length() / scale);
            }
        }
    }
    log.push(format!(
        "立方体接缝相对误差：无缝 {:.5} / 逐面钳制 {:.5}；面内基准 {:.5}",
        err_seamless, err_clamped, err_interior
    ));

    // Roughness widens the lobe.
    let bright_dir = math::direction_from_azimuth_elevation(35.0, 25.0);
    let off = math::direction_from_azimuth_elevation(52.0, 25.0);
    let sharp = env::lookup_prefiltered(&cache, off, 0.0).x / env::lookup_prefiltered(&cache, bright_dir, 0.0).x;
    let rough = env::lookup_prefiltered(&cache, off, 0.85).x / env::lookup_prefiltered(&cache, bright_dir, 0.85).x;
    log.push(format!(
        "粗糙层扩散比：锐利 {:.4} → 粗糙 {:.4}（应增大）",
        sharp, rough
    ));

    let (dir, rad) = lighting::brightest_in_cache(&precomputed.prefilter, env_rotation);
    log.push(format!(
        "当前最亮方向：{}；辐亮度 {:.3}, {:.3}, {:.3}",
        dir_label(dir),
        rad.x,
        rad.y,
        rad.z
    ));
    log
}

// ---------------------------------------------------------------------------
// Headless verification (GPU readback + report)
// ---------------------------------------------------------------------------

fn run_headless(
    env_path: &Path,
    out_dir: &Path,
    settings: &PrecomputeSettings,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let image = env::load_hdr(env_path)
        .map_err(|e| anyhow::anyhow!("载入环境失败：{e}"))?;
    let meta = env::env_meta(
        &image,
        env::file_sha256(env_path).unwrap_or_default(),
        env_path.display().to_string(),
    );
    let precomputed = cache::precompute(&image, &meta.sha256, settings, |msg, p| {
        eprintln!("[{:3.0}%] {msg}", p * 100.0);
    })
    .map_err(|e| anyhow::anyhow!("预计算失败：{e}"))?;

    let mut log = verify_kernel(&precomputed, 0.0, settings);
    log.insert(
        0,
        format!(
            "环境 {}（{}×{}，平均亮度 {:.3}，动态范围 {:.0}:1，预计算 {:.1} ms）",
            env_path.display(),
            meta.width,
            meta.height,
            meta.mean_luminance,
            meta.dynamic_range,
            precomputed.compute_ms
        ),
    );

    // CPU reference at default camera.
    let settings_ref = reference::ReferenceSettings {
        width: 640,
        height: 480,
        camera: Camera::default(),
        exhibit: ExhibitKind::Pottery,
        material: geometry::material_for(ExhibitKind::Pottery),
        exhibit_rotation: 0.0,
        env_rotation: 0.0,
        exposure: 1.0,
        forced_roughness: None,
    };
    let reference_img = reference::render(&precomputed, &settings_ref);
    let path = out_dir.join("reference_0deg.png");
    image::save_buffer(
        &path,
        &reference_img.rgb8,
        reference_img.width,
        reference_img.height,
        image::ColorType::Rgb8,
    )?;
    log.push(format!(
        "CPU 参考图 {}（{:.0} ms），最亮方向 {}",
        path.display(),
        reference_img.elapsed_ms,
        dir_label(reference_img.brightest_dir)
    ));

    // Verify that reloading the same cached data is deterministic.
    if settings.use_cache {
        let reloaded = cache::precompute(&image, &meta.sha256, settings, |_, _| {})
            .map_err(|e| anyhow::anyhow!("缓存重载失败：{e}"))?;
        let same = reloaded.prefilter[0][0]
            .iter()
            .zip(&precomputed.prefilter[0][0])
            .all(|(a, b)| a == b);
        log.push(format!(
            "缓存重载一致性：{}（mip0 面0 逐像素{}）",
            if same { "一致" } else { "不一致" },
            if same { "相同" } else { "不同" }
        ));
    }

    // GPU images: panorama + scene at two rotations, using the same renderer
    // the window uses.
    if let Err(e) = render_gpu_views(&precomputed, out_dir, &mut log) {
        log.push(format!("GPU 渲染失败：{e}"));
    }

    let report = log.join("\n");
    std::fs::write(out_dir.join("verify_report.txt"), &report)?;
    println!("{report}");
    Ok(())
}

fn run_headless_asset(
    manifest_path: &Path,
    out_dir: &Path,
    _settings: &PrecomputeSettings,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let loaded = asset::load_asset(manifest_path)
        .map_err(|e| anyhow::anyhow!("素材载入失败：{e}"))?;
    let m = &loaded.manifest;
    let v = &loaded.measured;
    let mut log = Vec::new();
    log.push(format!(
        "素材 {}（{}）载入 {:.0} ms",
        m.id, m.name, loaded.load_ms
    ));
    log.push(format!(
        "源环境 {}（{}×{}，SHA-256 {}…）",
        m.source.path,
        m.source.width,
        m.source.height,
        &m.source.sha256[..16]
    ));
    log.push(format!(
        "预过滤 {}（{} mip，采样 {}，{} 字节，SHA-256 {}…）",
        m.prefilter.path,
        m.prefilter.mips,
        m.prefilter.samples,
        m.prefilter.byte_len,
        &m.prefilter.sha256[..16]
    ));
    log.push(format!(
        "实测最亮方向：方位角 {:+.2}°，仰角 {:+.2}°；主光偏角 {:.2}°（阈值 {:.1}°）",
        v.brightest.azimuth_deg,
        v.brightest.elevation_deg,
        v.brightest.key_offset_deg,
        asset::KEY_DIRECTION_TOLERANCE_DEG
    ));
    log.push(format!(
        "主光辐亮度：{:.3}, {:.3}, {:.3}（亮度 {:.3}）",
        v.brightest.radiance[0],
        v.brightest.radiance[1],
        v.brightest.radiance[2],
        v.brightest.luminance
    ));
    log.push(format!(
        "能量：源平均 {:.5} / SH 平均 {:.5}（偏差 {:.3}%）/ mip6 平均 {:.5}（偏差 {:.3}%）",
        v.energy.source_mean_luminance,
        v.energy.sh_mean_luminance,
        v.energy.sh_rel_diff * 100.0,
        v.energy.mip6_mean_luminance,
        v.energy.mip6_rel_diff * 100.0
    ));
    for r in &v.rotation_checks {
        log.push(format!(
            "旋转 {:+7.1}° → 实测最亮方位角 {:+7.2}°，期望 {:+7.2}°，偏差 {:.3}°",
            r.rotation_deg,
            r.brightest_azimuth_deg,
            r.expected_azimuth_deg,
            r.offset_deg
        ));
    }

    // CPU reference from the asset payload; proves the material preview can be
    // produced from the package alone.
    let ref_path = out_dir.join("asset_reference.png");
    let ref_img = reference::render(
        &loaded.precomputed,
        &reference::ReferenceSettings {
            width: 640,
            height: 480,
            camera: Camera::default(),
            exhibit: ExhibitKind::Pottery,
            material: geometry::material_for(ExhibitKind::Pottery),
            exhibit_rotation: 0.0,
            env_rotation: 0.0,
            exposure: 1.0,
            forced_roughness: None,
        },
    );
    image::save_buffer(
        &ref_path,
        &ref_img.rgb8,
        ref_img.width,
        ref_img.height,
        image::ColorType::Rgb8,
    )?;
    log.push(format!(
        "CPU 参考图 {}，最亮方向 {}",
        ref_path.display(),
        dir_label(ref_img.brightest_dir)
    ));

    // Instrumented kernel checks (same code the panel's 校验 button runs).
    log.extend(verify_kernel(
        &loaded.precomputed,
        0.0,
        &PrecomputeSettings::default(),
    ));

    if let Err(e) = render_gpu_views(&loaded.precomputed, out_dir, &mut log) {
        log.push(format!("GPU 渲染失败：{e}"));
    }

    let report = log.join("\n");
    std::fs::write(out_dir.join("asset_verify_report.txt"), &report)?;
    println!("{report}");
    Ok(())
}

fn render_gpu_views(
    precomputed: &Precomputed,
    out_dir: &Path,
    log: &mut Vec<String>,
) -> anyhow::Result<()> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    let features = adapter.features()
            & (wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("headless device"),
        required_features: features,
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
    }))?;
    let mut renderer = SceneRenderer::new(&device, &queue, precomputed, renderer::DISPLAY_FORMAT);
    renderer.ensure_target(&device, 960, 720);

    let mut cases: Vec<(String, f32)> = vec![("0deg".into(), 0.0), ("35deg".into(), 35.0)];
    for (name, rotation) in cases.drain(..) {
        let mut params = FrameParams::default();
        params.env_rotation = rotation.to_radians();
        params.exhibit = ExhibitKind::Bronze;
        params.material = geometry::material_for(ExhibitKind::Bronze);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("headless encoder"),
        });
        renderer.encode(&device, &queue, &mut encoder, &params);
        queue.submit([encoder.finish()]);
        if let Some((w, h, rgba)) = renderer.read_display_rgba(&device, &queue) {
            let path = out_dir.join(format!("scene_{name}.png"));
            if let Some(buf) = image::RgbaImage::from_raw(w, h, rgba) {
                buf.save(&path)?;
                log.push(format!("GPU 场景图 {}", path.display()));
            }
        }
        if let Some((w, h, rgba)) = renderer.read_panorama_rgba(&device, &queue) {
            let path = out_dir.join(format!("panorama_{name}.png"));
            if let Some(buf) = image::RgbaImage::from_raw(w, h, rgba) {
                buf.save(&path)?;
                log.push(format!("环境全景图 {}", path.display()));
            }
        }
    }

    // Match the CPU reference at the same resolution and camera so the two
    // images can be compared directly. The CPU path shares the precomputed
    // cache and the math conventions with the GPU path but evaluates lighting
    // itself, so disagreement beyond normal sampling noise means the GPU is
    // not reproducing the kernel.
    let cmp_w = 480u32;
    let cmp_h = 360u32;
    renderer.ensure_target(&device, cmp_w, cmp_h);
    {
        let mut params = FrameParams::default();
        params.exhibit = ExhibitKind::Pottery;
        params.material = geometry::material_for(ExhibitKind::Pottery);
        params.exposure = 1.0;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("agreement encoder"),
        });
        renderer.encode(&device, &queue, &mut encoder, &params);
        queue.submit([encoder.finish()]);
        let cpu = reference::render(
            precomputed,
            &reference::ReferenceSettings {
                width: cmp_w,
                height: cmp_h,
                camera: Camera::default(),
                exhibit: ExhibitKind::Pottery,
                material: geometry::material_for(ExhibitKind::Pottery),
                exhibit_rotation: 0.0,
                env_rotation: 0.0,
                exposure: 1.0,
                forced_roughness: None,
            },
        );
        if let Some((w, h, rgba)) = renderer.read_display_rgba(&device, &queue) {
            if (w, h) == (cmp_w, cmp_h) {
                let mut sum = 0u64;
                let mut count = 0u64;
                let mut worst = 0u8;
                for (i, px) in rgba.chunks_exact(4).enumerate() {
                    for c in 0..3 {
                        let gpu = px[c];
                        let cpu_v = cpu.rgb8[i * 3 + c];
                        let d = gpu.abs_diff(cpu_v);
                        sum += d as u64;
                        worst = worst.max(d);
                        count += 1;
                    }
                }
                let mean = sum as f64 / count.max(1) as f64;
                log.push(format!(
                    "GPU 与 CPU 参考一致度：平均绝对差 {:.2}/255（最大 {}），{}×{} 同机位同曝光",
                    mean, worst, cmp_w, cmp_h
                ));
                if let Some(buf) = image::RgbaImage::from_raw(w, h, rgba) {
                    buf.save(out_dir.join("agreement_gpu.png"))?;
                }
                image::save_buffer(
                    out_dir.join("agreement_cpu.png"),
                    &cpu.rgb8,
                    cmp_w,
                    cmp_h,
                    image::ColorType::Rgb8,
                )?;
            }
        }
    }

    // Confirm the scene target really is linear HDR and that the sky is
    // sampled from the same cube the reflections use.
    {
        let mut params = FrameParams::default();
        params.exhibit = ExhibitKind::Pottery;
        params.material = geometry::material_for(ExhibitKind::Pottery);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hdr encoder"),
        });
        renderer.encode(&device, &queue, &mut encoder, &params);
        queue.submit([encoder.finish()]);
        if let Some(hdr) = renderer.read_hdr_pixels(&device, &queue) {
            log.push(format!(
                "线性 HDR 抽样（4 个位置）：{:.3},{:.3},{:.3} · {:.3},{:.3},{:.3} · {:.3},{:.3},{:.3} · {:.3},{:.3},{:.3}",
                hdr[0][0], hdr[0][1], hdr[0][2],
                hdr[1][0], hdr[1][1], hdr[1][2],
                hdr[2][0], hdr[2][1], hdr[2][2],
                hdr[3][0], hdr[3][1], hdr[3][2],
            ));
        }
    }
    let mut params = FrameParams::default();
    params.exhibit = ExhibitKind::Sphere;
    params.material = geometry::material_for(ExhibitKind::Sphere);
    for (i, lod) in [0.0f32, 2.0, 4.0, 6.0].iter().enumerate() {
        params.forced_lod = Some(*lod);
        params.debug_mode = 1;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("layer encoder"),
        });
        renderer.encode(&device, &queue, &mut encoder, &params);
        queue.submit([encoder.finish()]);
        if let Some((w, h, rgba)) = renderer.read_display_rgba(&device, &queue) {
            let path = out_dir.join(format!("roughness_layer_{i}_lod{lod:.0}.png"));
            if let Some(buf) = image::RgbaImage::from_raw(w, h, rgba) {
                buf.save(&path)?;
            }
        }
    }
    log.push("粗糙层对比图：mip 0/2/4/6 已输出".to_string());
    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum Startup {
    /// Open the window and precompute the given HDR.
    Env(PathBuf),
    /// Open the window with a bundled asset package and optional scene file.
    Asset {
        manifest: PathBuf,
        scene: Option<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "asset" {
        return asset_cli::run(&args[1..]);
    }
    let mut env_path = PathBuf::from(DEFAULT_ENV);
    let mut env_explicit = false;
    let mut verify_only = false;
    let mut out_dir = PathBuf::from("artifacts");
    let mut no_cache = false;
    let mut scene_path: Option<PathBuf> = None;
    let mut asset_manifest: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--env" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    env_path = PathBuf::from(p);
                    env_explicit = true;
                }
            }
            "--asset" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    asset_manifest = Some(PathBuf::from(p));
                }
            }
            "--scene" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    scene_path = Some(PathBuf::from(p));
                }
            }
            "--verify" => verify_only = true,
            "--out" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    out_dir = PathBuf::from(p);
                }
            }
            "--no-cache" => no_cache = true,
            "--help" | "-h" => {
                println!("{}", asset_cli::usage());
                return Ok(());
            }
            other => eprintln!("未知参数：{other}（使用 --help 查看用法）"),
        }
        i += 1;
    }

    let startup = if let Some(scene) = scene_path {
        let (_scene, manifest) = asset::load_scene(&scene)
            .map_err(|e| anyhow::anyhow!("载入场景失败：{e}"))?;
        Startup::Asset {
            manifest,
            scene: Some(scene),
        }
    } else if let Some(manifest) = asset_manifest {
        Startup::Asset {
            manifest,
            scene: None,
        }
    } else if !env_explicit && !verify_only {
        // The graphical entry point opens the bundled exhibit-lighting asset so
        // viewers see the documented lighting out of the box. `--verify` keeps
        // using the original sample environment below.
        let manifest = PathBuf::from(asset::DEFAULT_MANIFEST);
        if manifest.exists() {
            Startup::Asset {
                manifest,
                scene: None,
            }
        } else {
            Startup::Env(env_path)
        }
    } else {
        Startup::Env(env_path)
    };
    let settings = PrecomputeSettings {
        use_cache: !no_cache,
        ..PrecomputeSettings::default()
    };
    if verify_only {
        return match startup {
            Startup::Env(p) => run_headless(&p, &out_dir, &settings),
            Startup::Asset { manifest, .. } => run_headless_asset(&manifest, &out_dir, &settings),
        };
    }

    let event_loop = winit::event_loop::EventLoop::new()?;
    let mut app = match startup {
        Startup::Env(p) => App::new(settings, p),
        Startup::Asset { manifest, scene } => App::new_asset(settings, manifest, scene),
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

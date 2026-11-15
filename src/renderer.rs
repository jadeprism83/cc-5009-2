//! GPU scene renderer: environment cube upload, lit exhibit pass, sky pass,
//! display tone map, and surface presentation. All lighting targets stay in
//! linear HDR; only the display copy is gamma encoded.

use crate::cache::Precomputed;
use crate::env::{PREFILTER_MIPS, PREFILTER_SIZE, SKY_SIZE};
use crate::geometry::{ExhibitKind, Material, Mesh, Vertex};
use crate::lighting::{view_proj, Camera};
use crate::math::CUBE_FACES;
use crate::shaders;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const DISPLAY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    sh: [[f32; 4]; 9],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ObjectUniform {
    model: [[f32; 4]; 4],
    base_color: [f32; 4],
    mat: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DisplayUniform {
    exposure: f32,
    gamma_only: f32,
    _pad: [f32; 2],
}

const OBJECT_SLOT: u64 = 256;

pub struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

impl GpuMesh {
    fn new(device: &wgpu::Device, mesh: &Mesh) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices.len() as u32,
        }
    }
}

pub struct ViewTarget {
    pub width: u32,
    pub height: u32,
    hdr_texture: wgpu::Texture,
    hdr_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    pub display_texture: wgpu::Texture,
    pub display_view: wgpu::TextureView,
    display_bind_group: wgpu::BindGroup,
    blit_bind_group: wgpu::BindGroup,
    pub egui_id: Option<egui::TextureId>,
}

#[derive(Clone, Debug)]
pub struct FrameParams {
    pub camera: Camera,
    pub exhibit: ExhibitKind,
    pub exhibit_rotation: f32,
    pub material: Material,
    pub env_rotation: f32,
    pub exposure: f32,
    pub gamma_only: bool,
    pub debug_mode: u32,
    pub forced_lod: Option<f32>,
    pub show_ground: bool,
}

impl Default for FrameParams {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            exhibit: ExhibitKind::Pottery,
            exhibit_rotation: 0.0,
            material: crate::geometry::material_for(ExhibitKind::Pottery),
            env_rotation: 0.0,
            exposure: 1.0,
            gamma_only: false,
            debug_mode: 0,
            forced_lod: None,
            show_ground: true,
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct FrameTiming {
    pub cpu_encode_ms: f64,
    pub gpu_scene_ms: Option<f64>,
}

struct TimestampSupport {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f32,
    in_flight: bool,
    map_requested: bool,
    mapped: Arc<AtomicBool>,
}

pub struct SceneRenderer {
    display_bgl: wgpu::BindGroupLayout,
    blit_bgl: wgpu::BindGroupLayout,
    scene_pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    tonemap_pipeline: wgpu::RenderPipeline,
    panorama_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    blit_srgb_pipeline: wgpu::RenderPipeline,
    scene_bind_group: wgpu::BindGroup,
    scene_uniform: wgpu::Buffer,
    object_uniform: wgpu::Buffer,
    object_bind_group: wgpu::BindGroup,
    display_uniform: wgpu::Buffer,
    panorama_uniform: wgpu::Buffer,
    prefilter_texture: wgpu::Texture,
    sky_texture: wgpu::Texture,
    sampler: wgpu::Sampler,
    blit_sampler: wgpu::Sampler,
    meshes: HashMap<ExhibitKind, Vec<GpuMesh>>,
    ground: GpuMesh,
    pub targets: Option<ViewTarget>,
    pub panorama: wgpu::TextureView,
    panorama_display_texture: wgpu::Texture,
    panorama_display_view: wgpu::TextureView,
    panorama_bind_group: wgpu::BindGroup,
    panorama_display_bind_group: wgpu::BindGroup,
    panorama_egui_id: Option<egui::TextureId>,
    timestamps: Option<TimestampSupport>,
    pub timestamp_note: String,
    pub gpu_scene_ms: Option<f64>,
    scene_sh: [[f32; 4]; 9],
}

fn cube_view() -> wgpu::TextureViewDescriptor<'static> {
    wgpu::TextureViewDescriptor {
        label: Some("cube view"),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    }
}

fn make_cube_texture(device: &wgpu::Device, label: &str, size: u32, mips: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 6,
        },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn upload_cube(queue: &wgpu::Queue, tex: &wgpu::Texture, mips: &[Vec<Vec<[f32; 3]>>]) {
    for (mip, faces) in mips.iter().enumerate() {
        let size = (faces[0].len() as f32).sqrt().round() as u32;
        for face in 0..CUBE_FACES {
            let mut bytes = Vec::with_capacity(faces[face].len() * 8);
            for px in &faces[face] {
                for c in px {
                    bytes.extend_from_slice(&half::f16::from_f32(*c).to_le_bytes());
                }
                bytes.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
            }
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: mip as u32,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: face as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size * 8),
                    rows_per_image: Some(size),
                },
                wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

impl SceneRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        precomputed: &Precomputed,
        present_format: wgpu::TextureFormat,
    ) -> Self {
        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let object_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("object bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<ObjectUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let display_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let scene_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene uniform"),
            size: std::mem::size_of::<SceneUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let display_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("display uniform"),
            size: std::mem::size_of::<DisplayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let panorama_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("panorama display uniform"),
            size: std::mem::size_of::<DisplayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let prefilter_texture = make_cube_texture(
            device,
            "prefiltered radiance",
            PREFILTER_SIZE,
            PREFILTER_MIPS,
        );
        let sky_texture = make_cube_texture(device, "sky radiance", SKY_SIZE, 1);
        upload_cube(queue, &prefilter_texture, &precomputed.prefilter);
        upload_cube(queue, &sky_texture, &[precomputed.sky.clone()]);
        let prefilter_view = prefilter_texture.create_view(&cube_view());
        let sky_view = sky_texture.create_view(&cube_view());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("env sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: PREFILTER_MIPS as f32 - 1.0,
            ..Default::default()
        });
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene bind group"),
            layout: &scene_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&prefilter_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&sky_view),
                },
            ],
        });

        let ground = GpuMesh::new(device, &crate::geometry::ground_plane(6.0));
        let object_count = 4u64;
        let object_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("object uniforms"),
            size: OBJECT_SLOT * object_count,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let object_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("object bind group"),
            layout: &object_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &object_uniform,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<ObjectUniform>() as u64),
                }),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::SCENE_WGSL.into()),
        });
        let tonemap_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tonemap shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::TONEMAP_WGSL.into()),
        });
        let panorama_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("panorama shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::PANORAMA_WGSL.into()),
        });
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::BLIT_WGSL.into()),
        });

        let scene_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene layout"),
            bind_group_layouts: &[&scene_bgl, &object_bgl],
            push_constant_ranges: &[],
        });
        let scene_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene pipeline"),
            layout: Some(&scene_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Float32x2],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let sky_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky layout"),
            bind_group_layouts: &[&scene_bgl],
            push_constant_ranges: &[],
        });
        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky pipeline"),
            layout: Some(&sky_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_sky"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_sky"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let fullscreen_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fullscreen layout"),
            bind_group_layouts: &[&display_bgl],
            push_constant_ranges: &[],
        });
        let tonemap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tonemap pipeline"),
            layout: Some(&fullscreen_layout),
            vertex: wgpu::VertexState {
                module: &tonemap_shader,
                entry_point: Some("vs_display"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &tonemap_shader,
                entry_point: Some("fs_display"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: DISPLAY_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let panorama_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("panorama layout"),
            bind_group_layouts: &[&scene_bgl],
            push_constant_ranges: &[],
        });
        let panorama_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("panorama pipeline"),
            layout: Some(&panorama_layout),
            vertex: wgpu::VertexState {
                module: &panorama_shader,
                entry_point: Some("vs_pano"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &panorama_shader,
                entry_point: Some("fs_pano"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit layout"),
            bind_group_layouts: &[&blit_bgl],
            push_constant_ranges: &[],
        });
        let make_blit = |label: &str, entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&blit_layout),
                vertex: wgpu::VertexState {
                    module: &blit_shader,
                    entry_point: Some("vs_blit"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &blit_shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: present_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        };
        let blit_pipeline = make_blit("blit", "fs_blit");
        let blit_srgb_pipeline = make_blit("blit srgb", "fs_blit_to_srgb");

        let panorama_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("panorama bind group"),
            layout: &scene_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&prefilter_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&sky_view),
                },
            ],
        });

        let (pw, ph) = (1536u32, 768u32);
        let panorama_hdr = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("panorama hdr"),
            size: wgpu::Extent3d {
                width: pw,
                height: ph,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let panorama_view = panorama_hdr.create_view(&Default::default());
        let panorama_display = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("panorama display"),
            size: wgpu::Extent3d {
                width: pw,
                height: ph,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DISPLAY_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let panorama_display_view = panorama_display.create_view(&Default::default());
        let panorama_display_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("panorama display bind group"),
            layout: &display_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&panorama_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: panorama_uniform.as_entire_binding(),
                },
            ],
        });

        let timestamp_features =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let timestamps = if device.features().contains(timestamp_features) {
            let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("frame timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });
            let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp resolve"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp readback"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            Some(TimestampSupport {
                query_set,
                resolve_buffer,
                readback,
                period_ns: queue.get_timestamp_period(),
                in_flight: false,
                map_requested: false,
                mapped: Arc::new(AtomicBool::new(false)),
            })
        } else {
            None
        };
        let timestamp_note = if timestamps.is_some() {
            "GPU 时间戳查询：可用".to_string()
        } else {
            "GPU 时间戳查询：适配器未同时提供查询与编码器内写入，只能报告 CPU 编码耗时".to_string()
        };

        let mut renderer = Self {
            display_bgl,
            blit_bgl,
            scene_pipeline,
            sky_pipeline,
            tonemap_pipeline,
            panorama_pipeline,
            blit_pipeline,
            blit_srgb_pipeline,
            scene_bind_group,
            scene_uniform,
            object_uniform,
            object_bind_group,
            display_uniform,
            panorama_uniform,
            prefilter_texture,
            sky_texture,
            sampler,
            blit_sampler,
            meshes: HashMap::new(),
            ground,
            targets: None,
            panorama: panorama_view,
            panorama_display_texture: panorama_display,
            panorama_display_view,
            panorama_bind_group,
            panorama_display_bind_group,
            panorama_egui_id: None,
            timestamps,
            timestamp_note,
            gpu_scene_ms: None,
            scene_sh: [[0.0; 4]; 9],
        };
        renderer.set_scene_sh(&precomputed.sh);
        renderer
    }

    /// Upload a replacement environment. Texture identities are preserved so
    /// the bind groups and registered view handles stay valid.
    pub fn update_environment(&mut self, queue: &wgpu::Queue, precomputed: &Precomputed) {
        upload_cube(queue, &self.prefilter_texture, &precomputed.prefilter);
        upload_cube(queue, &self.sky_texture, &[precomputed.sky.clone()]);
        self.set_scene_sh(&precomputed.sh);
    }

    pub fn set_scene_sh(&mut self, sh: &[[f32; 3]; 9]) {
        for i in 0..9 {
            self.scene_sh[i] = [sh[i][0], sh[i][1], sh[i][2], 0.0];
        }
    }

    pub fn ensure_meshes(&mut self, device: &wgpu::Device, kind: ExhibitKind) {
        self.meshes.entry(kind).or_insert_with(|| {
            crate::geometry::build_exhibit(kind)
                .iter()
                .map(|m| GpuMesh::new(device, m))
                .collect()
        });
    }

    pub fn set_panorama_egui_id(&mut self, id: egui::TextureId) {
        self.panorama_egui_id = Some(id);
    }

    pub fn panorama_egui_id(&self) -> Option<egui::TextureId> {
        self.panorama_egui_id
    }

    pub fn panorama_display_view(&self) -> &wgpu::TextureView {
        &self.panorama_display_view
    }

    pub const PANORAMA_SIZE: (u32, u32) = (1536, 768);

    pub fn set_target_egui_id(&mut self, id: egui::TextureId) {
        if let Some(t) = &mut self.targets {
            t.egui_id = Some(id);
        }
    }

    pub fn display_egui_id(&self) -> Option<egui::TextureId> {
        self.targets.as_ref().and_then(|t| t.egui_id)
    }

    pub fn target_view_for_egui(&self) -> Option<&wgpu::TextureView> {
        self.targets.as_ref().map(|t| &t.display_view)
    }

    fn create_target(&mut self, device: &wgpu::Device, width: u32, height: u32) -> ViewTarget {
        let width = width.max(1);
        let height = height.max(1);
        let hdr = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene hdr"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let display = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene display"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DISPLAY_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let hdr_view = hdr.create_view(&Default::default());
        let display_view = display.create_view(&Default::default());
        let display_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene display bind group"),
            layout: &self.display_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.display_uniform.as_entire_binding(),
                },
            ],
        });
        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene blit bind group"),
            layout: &self.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&display_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                },
            ],
        });
        ViewTarget {
            width,
            height,
            hdr_texture: hdr,
            hdr_view,
            depth_view: depth.create_view(&Default::default()),
            display_texture: display,
            display_view,
            display_bind_group,
            blit_bind_group,
            egui_id: None,
        }
    }

    /// Ensure the offscreen target matches the requested size. Returns true if
    /// it was (re)created, in which case the caller should update the
    /// registered egui texture.
    pub fn ensure_target(&mut self, device: &wgpu::Device, width: u32, height: u32) -> bool {
        let recreate = match &self.targets {
            Some(t) => t.width != width.max(1) || t.height != height.max(1),
            None => true,
        };
        if recreate {
            let id = self.targets.as_ref().and_then(|t| t.egui_id);
            let mut target = self.create_target(device, width, height);
            target.egui_id = id;
            self.targets = Some(target);
        }
        recreate
    }

    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        params: &FrameParams,
    ) -> FrameTiming {
        let started = std::time::Instant::now();
        self.ensure_meshes(device, params.exhibit);
        let Some(target) = &self.targets else {
            return FrameTiming::default();
        };

        let aspect = target.width as f32 / target.height as f32;
        let vp = view_proj(&params.camera, aspect);
        let inv_vp = vp.inverse();
        let uniform = SceneUniform {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: inv_vp.to_cols_array_2d(),
            cam_pos: [
                params.camera.position[0],
                params.camera.position[1],
                params.camera.position[2],
                1.0,
            ],
            sh: self.scene_sh,
            params: [
                params.env_rotation,
                params.debug_mode as f32,
                params.forced_lod.unwrap_or(-1.0),
                0.0,
            ],
        };
        queue.write_buffer(&self.scene_uniform, 0, bytemuck::bytes_of(&uniform));
        let disp = DisplayUniform {
            exposure: params.exposure,
            gamma_only: if params.gamma_only { 1.0 } else { 0.0 },
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.display_uniform, 0, bytemuck::bytes_of(&disp));
        queue.write_buffer(&self.panorama_uniform, 0, bytemuck::bytes_of(&disp));

        let parts = self.meshes.get(&params.exhibit).map(|v| v.len()).unwrap_or(0);
        let exhibit_model = glam::Mat4::from_rotation_y(params.exhibit_rotation)
            * glam::Mat4::from_translation(glam::Vec3::new(
                0.0,
                params.exhibit.floor_offset(),
                0.0,
            ));
        let mat = params.material;
        let make_object = |model: glam::Mat4, base: [f32; 3], m: Material| ObjectUniform {
            model: model.to_cols_array_2d(),
            base_color: [base[0], base[1], base[2], 1.0],
            mat: [m.roughness, m.metallic, m.reflectance, m.finish as f32],
        };
        for i in 0..parts {
            let data = make_object(exhibit_model, mat.base_color, mat);
            queue.write_buffer(
                &self.object_uniform,
                i as u64 * OBJECT_SLOT,
                bytemuck::bytes_of(&data),
            );
        }
        let ground_material = Material {
            base_color: [0.24, 0.235, 0.225],
            roughness: 0.72,
            metallic: 0.0,
            reflectance: 0.35,
            finish: 2,
        };
        let ground_data = make_object(
            glam::Mat4::IDENTITY,
            ground_material.base_color,
            ground_material,
        );
        queue.write_buffer(
            &self.object_uniform,
            parts as u64 * OBJECT_SLOT,
            bytemuck::bytes_of(&ground_data),
        );

        let timestamp_valid = self
            .timestamps
            .as_ref()
            .map(|t| !t.in_flight)
            .unwrap_or(false);
        if timestamp_valid {
            encoder.write_timestamp(&self.timestamps.as_ref().unwrap().query_set, 0);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.scene_bind_group, &[]);
            // Sky goes first with depth writes off, so it never occludes the
            // exhibits drawn afterwards.
            pass.set_pipeline(&self.sky_pipeline);
            pass.draw(0..3, 0..1);
            if params.show_ground {
                pass.set_pipeline(&self.scene_pipeline);
                pass.set_bind_group(
                    1,
                    &self.object_bind_group,
                    &[(parts as u64 * OBJECT_SLOT) as u32],
                );
                pass.set_vertex_buffer(0, self.ground.vertex_buffer.slice(..));
                pass.set_index_buffer(self.ground.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.ground.index_count, 0, 0..1);
            }
            if let Some(parts_meshes) = self.meshes.get(&params.exhibit) {
                pass.set_pipeline(&self.scene_pipeline);
                for (i, mesh) in parts_meshes.iter().enumerate() {
                    pass.set_bind_group(
                        1,
                        &self.object_bind_group,
                        &[(i as u64 * OBJECT_SLOT) as u32],
                    );
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("display pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.display_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.tonemap_pipeline);
            pass.set_bind_group(0, &target.display_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("panorama pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.panorama,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.panorama_pipeline);
            pass.set_bind_group(0, &self.panorama_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("panorama display pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.panorama_display_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.tonemap_pipeline);
            pass.set_bind_group(0, &self.panorama_display_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        if timestamp_valid {
            let ts = self.timestamps.as_mut().unwrap();
            encoder.write_timestamp(&ts.query_set, 1);
            encoder.resolve_query_set(&ts.query_set, 0..2, &ts.resolve_buffer, 0);
            encoder.copy_buffer_to_buffer(&ts.resolve_buffer, 0, &ts.readback, 0, 16);
            ts.in_flight = true;
            ts.map_requested = false;
        }

        FrameTiming {
            cpu_encode_ms: started.elapsed().as_secs_f64() * 1000.0,
            gpu_scene_ms: self.gpu_scene_ms,
        }
    }

    /// Poll for the previous frame's GPU timestamp. Call once per frame after
    /// submission; results appear a frame or two later and never block.
    pub fn collect_timestamps(&mut self, device: &wgpu::Device) {
        let Some(ts) = &mut self.timestamps else {
            return;
        };
        if !ts.in_flight {
            return;
        }
        if !ts.map_requested {
            let flag = ts.mapped.clone();
            flag.store(false, Ordering::SeqCst);
            ts.readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    flag.store(r.is_ok(), Ordering::SeqCst);
                });
            ts.map_requested = true;
            let _ = device.poll(wgpu::PollType::Poll);
            return;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        if !ts.mapped.load(Ordering::SeqCst) {
            return;
        }
        {
            let data = ts.readback.slice(..).get_mapped_range();
            let stamps: &[u64] = bytemuck::cast_slice(&data);
            if stamps.len() >= 2 && stamps[1] > stamps[0] {
                let ns = (stamps[1] - stamps[0]) as f64 * ts.period_ns as f64;
                self.gpu_scene_ms = Some(ns / 1e6);
            }
        }
        ts.readback.unmap();
        ts.in_flight = false;
        ts.map_requested = false;
    }

    /// Present the tone-mapped display texture to the window surface.
    pub fn blit_to_surface(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
    ) {
        let Some(target) = &self.targets else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("present pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(if target_format.is_srgb() {
            &self.blit_srgb_pipeline
        } else {
            &self.blit_pipeline
        });
        pass.set_bind_group(0, &target.blit_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Read back the display texture. The bytes are sRGB encoded already, so
    /// they can be written to an 8-bit PNG without further conversion.
    pub fn read_display_rgba(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Option<(u32, u32, Vec<u8>)> {
        let target = self.targets.as_ref()?;
        read_texture_rgba(
            device,
            queue,
            &target.display_texture,
            target.width,
            target.height,
            "display readback",
        )
    }

    pub fn read_hdr_pixels(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Option<Vec<[f32; 4]>> {
        let target = self.targets.as_ref()?;
        let (w, h) = (target.width, target.height);
        let padded_row = (w * 8).div_ceil(256) * 256;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdr sample readback"),
            size: (padded_row as u64) * (h as u64),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hdr readback encoder"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.hdr_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::Wait);
        rx.recv().ok()?.ok()?;
        let data = slice.get_mapped_range();
        let mut out = Vec::new();
        for (x, y) in [(w / 2, h / 2), (w / 3, h / 2), (w / 2, h / 3), (w / 4, h / 4)] {
            let offset = (y as usize * padded_row as usize) + x as usize * 8;
            let stride = &data[offset..offset + 8];
            let mut vals = [0.0f32; 4];
            for i in 0..4 {
                let bits = u16::from_le_bytes([stride[i * 2], stride[i * 2 + 1]]);
                vals[i] = half::f16::from_bits(bits).to_f32();
            }
            out.push(vals);
        }
        drop(data);
        buffer.unmap();
        Some(out)
    }

    pub fn read_panorama_rgba(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Option<(u32, u32, Vec<u8>)> {
        read_texture_rgba(
            device,
            queue,
            &self.panorama_display_texture,
            Self::PANORAMA_SIZE.0,
            Self::PANORAMA_SIZE.1,
            "panorama readback",
        )
    }
}

fn read_texture_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    w: u32,
    h: u32,
    label: &str,
) -> Option<(u32, u32, Vec<u8>)> {
    let padded_row = (w * 4).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (padded_row as u64) * (h as u64),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    rx.recv().ok()?.ok()?;
    let data = slice.get_mapped_range();
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for row in 0..h {
        let start = (row * padded_row) as usize;
        out.extend_from_slice(&data[start..start + (w * 4) as usize]);
    }
    drop(data);
    buffer.unmap();
    Some((w, h, out))
}

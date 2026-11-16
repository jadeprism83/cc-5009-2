# 博物馆展品环境光预览器

用 Rust + wgpu 渲染陶罐、铜器、球体/立方体/圆环测试件，载入 HDR 环境图，
在线性空间计算漫反射与粗糙反射所需的预计算数据，并提供可拖转环境、切换展品、
查看反射细节的交互材质预览窗口。

## 快速开始（界面）

```bash
# 构建（本机需要 CC=/usr/bin/gcc AR=/usr/bin/ar；/usr/local/bin/cc 是包装脚本，
# 会拒绝 Rust 链接参数）
export PATH="$HOME/.cargo/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export CC=/usr/bin/gcc AR=/usr/bin/ar
cargo build --release

# 直接打开交互窗口（有真实显示器时）——默认载入内置的展品光照素材包
./target/release/museum-lighting

# 指定素材清单 / 场景文件 / 普通环境图打开
./target/release/museum-lighting --asset assets/exhibit_lighting/assets/directed_key/manifest.json
./target/release/museum-lighting --scene assets/exhibit_lighting/scenes/reflection_direction_35deg.scene.json
./target/release/museum-lighting --env assets/env/studio_small_09_2k.hdr
```

**图形入口**：`museum-lighting`（无参数）就是展品光照的图形入口，启动后自动
发现并载入 `assets/exhibit_lighting/` 下的素材包与场景。窗口左侧是「展品光照」
面板（由展品光照内核驱动的控件），中间是场景。关键控件：

- **展品光照素材集**：下拉选择素材包；面板列出源文件、SHA-256、预过滤层数，
  以及**载入时重新实测**的最亮方向、主光辐亮度与能量核验（源/SH/mip6），
  并给出「与规格主光偏 X°（阈值 1.5°）」。数值不读清单里的记录值，因此
  篡改过的素材会在这里报错而不是显示成功。展开项可查看旋转核验、来源与参数、
  方向约定。
- **场景**：下拉直接打开 `assets/exhibit_lighting/scenes/` 中的场景文件；场景
  同时设置环境旋转、展品与调试通道，便于对照反射方向（例如 0° 与 35° 两张
  场景里高光位置沿方位角移动；粗糙度展板特写用 -71° 把主光反射带放到锐利端）。
- **复核素材（按规格重建比对）**：从 `spec.json` 重新生成源环境、投影与预过滤
  数据，与当前素材包逐值比较；不一致时在面板内提示。
- **展品**：陶罐 / 铜器 / 球体测试件 / 立方体测试件 / 圆环测试件 / 粗糙度展板。
- **展品世界旋转**：只影响模型矩阵。
- **环境旋转**：只影响环境方向，与模型旋转相互独立。也可在场景中按住左键
  横向拖动，或用「环境全景」窗口拖动；背景与物体反射同步变化。按住右键或
  中键拖动可环绕相机。
- **光照数据**：来源、尺寸、平均/最大亮度、动态范围、最亮方向（方位角/仰角）
  以及缓存 mip 0 的实际辐亮度。
- **显示映射**：曝光、固定线性截断（仅 gamma 编码、不做色调映射）、
  强制 mip 层、调试通道（预过滤层原始 HDR / 漫反射项 / 粗糙度 / 法线 /
  高光项 / 背景方向）。照明始终在线性空间计算。
- **耗时（分别统计）**：预计算耗时、每帧 CPU 命令编码耗时、每帧 GPU 场景绘制
  耗时（时间戳查询可用时）。
- **参考渲染（CPU 独立实现）**：480×360 的 CPU 光线追踪参考图，可与 GPU 结果
  对照；参数变化时自动重建。
- **环境文件**：输入路径并「载入」，或「重新载入默认环境」。不支持的尺寸或格式
  会回报可恢复错误，面板会明确说明「文件未替换当前环境（当前仍显示 …）」。
- **测试图**：单亮区 / 方向标记 / 均匀亮度 / 经纬网格，用于验证反射朝向、
  立方体接缝与粗糙层变化。
- **保存画面 PNG** / **运行校验**：写出 `artifacts/scene.png` /
  `artifacts/panorama.png`，或在窗口内显示与 `--verify` 相同的校验结果。
- 面板底部有「如何启动界面」折叠说明。

## 展品光照素材包

内置素材位于 `assets/exhibit_lighting/`，是一份可直接打开、可重建、可核验的
真实资源（不是成品图片）：

```
assets/exhibit_lighting/
  spec.json                                 素材规格：线性辐亮度、方向约定、预过滤参数
  sources/directed_key_2048x1024.hdr        自制源 HDR（明亮点源 + 暗环境，线性值）
  assets/directed_key/prefilter_f32.bin      预过滤辐亮度（7 层 mip，GGX）
  assets/directed_key/manifest.json         来源/参数/哈希与实测的方向、能量、旋转核验
  scenes/*.scene.json                       可直接打开的场景（含素材引用与预览参数）
```

规格要点：暗室基底约 `0.006`，一盏暖白主光 `[95, 82, 64]`、方位角 35°、
仰角 25°、半径约 5°，另有低能量冷色补光；`roughness = mip / 6`。
方向约定写在 `spec.json` 的 `conventions` 与清单中：`up = +Y`，
方位角 `atan2(z, x)`，仰角 `asin(y)`；环境旋转 +θ 使最亮方位角减少 θ。

重建素材（源 HDR、预过滤数据、清单、场景全部从头生成）：

```bash
./scripts/build_exhibit_lighting_assets.sh --release
```

脚本会构建并运行 `museum-lighting asset build`（从规格重建），随后运行
`asset verify`：核对源与预过滤 SHA-256、实测最亮方向与规格主光的偏差
（阈值 1.5°）、能量（源均值 vs SH 均值 ≤1%，vs mip6 ≤5%）、各旋转预设的
方位角一致性，以及「按规格重建 → 与素材包逐值相等」。构建与重复运行的结果
**逐字节相同**（本仓库已验证），因此素材包不是默认值伪装的占位物。

命令行仅用于辅助素材的构建与核验，查看照明请用图形入口：

```bash
./target/release/museum-lighting                    # 图形入口（自动载入素材包）
./target/release/museum-lighting asset list         # 列出素材与场景
./target/release/museum-lighting asset verify       # 载入并复核素材（含所有场景）
./target/release/museum-lighting asset build        # 仅从规格重建，不开窗口
```

## 无窗口校验

```bash
# 软件 Vulkan（Mesa lavapipe）环境变量见下节
./target/release/museum-lighting --verify
```

会写出 `artifacts/` 下的校验图与 `verify_report.txt`，包括：

- 各 mip 层的实际辐亮度与 `mip→粗糙度 = mip / 6` 对应表；
- 均匀环境 SH 辐照度对 πL 的核对；
- 单亮区反射朝向偏差（阈值 10°）；
- 立方体接缝相对误差（无缝采样 vs 逐面钳制 vs 面内基准）；
- 粗糙层扩散比（锐利/粗糙）；
- CPU 参考图、环境旋转 0°/35° 的同步对比；
- 缓存重载逐像素一致性；
- GPU 与 CPU 参考一致度（平均绝对差）；
- 线性 HDR 抽样与粗糙层对比图。

参数：`--env <hdr>`、`--out <目录>`、`--no-cache`（忽略磁盘缓存强制重建）。

## 无显示器 / 软件渲染

```bash
# 1) 软件 Vulkan：Mesa lavapipe
export VK_ICD_FILENAMES=/path/to/lvp.json
export LD_LIBRARY_PATH=/path/to/vulkan/libs:$LD_LIBRARY_PATH
export LIBGL_ALWAYS_SOFTWARE=1

# 2) 无头 X（如需窗口）
Xvfb :99 -screen 0 1400x900x24 &
export DISPLAY=:99

# 3) 启动
./target/release/museum-lighting
```

在本机（容器、无显示、仅软件 Vulkan）实际验证过的启动脚本：

```bash
#!/bin/bash
export PATH="$HOME/.cargo/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export CC=/usr/bin/gcc AR=/usr/bin/ar
export DISPLAY=:99
export VK_ICD_FILENAMES=/tmp/vkicd/lvp.json
export LD_LIBRARY_PATH=/tmp/sysroot/usr/lib/aarch64-linux-gnu:/tmp
export LIBGL_ALWAYS_SOFTWARE=1
export XKB_CONFIG_ROOT=/tmp/sysroot/usr/share/X11/xkb
export XFILESEARCHPATH=/tmp/sysroot/usr/share/X11/%T/%N%S
cd /workspace
exec ./target/<debug|release>/museum-lighting "$@"
```

`XKB_CONFIG_ROOT` / `XFILESEARCHPATH` 是精简容器里 winit 读取 xkb 布局所需；
有完整 X11 数据的主机通常不需要。

## 缓存

预计算缓存在启动时写入 `<临时目录>/museum-lighting-cache/`，键为
环境 SHA-256 + 算法版本 + 采样数。重载同一环境时命中磁盘缓存，结果逐像素
可复现（`--verify` 会核对）；`--no-cache` 强制重建。

## 设计要点

- **线性照明**：场景目标为 `Rgba16Float`，HDR 在线性空间参与照明；曝光和
  ACES/sRGB 映射只在最后显示阶段应用。
- **预过滤层 ↔ 粗糙度**：7 层 mip，`roughness = mip / 6`，在缓存、着色器、
  CPU 内核与 UI 中显式一致。
- **立方体连续性**：方向↔面/UV 使用同一套约定，跨面采样无缝；校验用解析环境
  与逐面钳制做负对照。
- **旋转解耦**：`exhibit_rotation` 只进入模型矩阵，`env_rotation` 只进入环境方向。
- **可恢复错误**：环境载入/校验失败会返回 `EnvError`，不会沿用旧环境却标新文件名。
- **双实现**：GPU 着色器与 CPU `reference` 内核共用同一预计算缓存与数学定义，
  用于独立对照；`--verify` 报告二者一致度。

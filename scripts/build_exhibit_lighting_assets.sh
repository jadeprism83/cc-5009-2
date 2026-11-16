#!/usr/bin/env bash
# Rebuild the bundled exhibit-lighting asset package from its spec.
#
# Inputs (linear radiance values and direction conventions live in these files):
#   assets/exhibit_lighting/spec.json
# Outputs (overwritten in place):
#   assets/exhibit_lighting/sources/directed_key_2048x1024.hdr
#   assets/exhibit_lighting/assets/directed_key/prefilter_f32.bin
#   assets/exhibit_lighting/assets/directed_key/manifest.json
#   assets/exhibit_lighting/scenes/*.scene.json
#
# The build measures the shipped RGBE-quantised source (not the in-memory
# analytic values), so a later load reproduces the recorded brightest
# direction and energy exactly. Run it from a checkout; it refuses to touch
# anything outside assets/exhibit_lighting.
#
# Usage:
#   scripts/build_exhibit_lighting_assets.sh [--release|--debug]
#
# Environment overrides:
#   CARGO        cargo binary (default: cargo)
#   CC / AR      required on hosts where /usr/local/bin/cc wraps the linker
#
# This script is auxiliary to the graphical preview. To view the result:
#   ./target/release/museum-lighting --asset assets/exhibit_lighting/assets/directed_key/manifest.json
# or simply run ./target/release/museum-lighting and pick a scene in the
# 「展品光照素材集」 panel.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="release"
for arg in "$@"; do
    case "$arg" in
        --release) PROFILE="release" ;;
        --debug) PROFILE="debug" ;;
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "未知参数：$arg（支持 --release / --debug）" >&2; exit 2 ;;
    esac
done

CARGO="${CARGO:-cargo}"
SPEC="$ROOT/assets/exhibit_lighting/spec.json"
ASSET_ROOT="$ROOT/assets/exhibit_lighting"
MANIFEST="$ASSET_ROOT/assets/directed_key/manifest.json"
BIN="$ROOT/target/$PROFILE/museum-lighting"

if [[ ! -f "$SPEC" ]]; then
    echo "找不到规格文件：$SPEC" >&2
    exit 1
fi

cd "$ROOT"

echo "== 构建 museum-lighting（$PROFILE）=="
if [[ "$PROFILE" == "release" ]]; then
    "$CARGO" build --release
else
    "$CARGO" build
fi

echo
echo "== 从规格重建素材包 =="
"$BIN" asset build --spec "$SPEC" --root "$ASSET_ROOT"

echo
echo "== 载入并复核（源/预过滤哈希、反射方向、能量、旋转、重建一致性）=="
"$BIN" asset verify --manifest "$MANIFEST" --spec "$SPEC"

echo
echo "素材包已重建：$ASSET_ROOT"
echo "图形入口：$BIN --asset $MANIFEST"

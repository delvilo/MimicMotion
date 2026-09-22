#!/usr/bin/env bash
# Linux x86_64 installer. Never installs NVIDIA drivers or merges Git branches.
set +x
set -Eeuo pipefail
trap 'printf "\nInstallation failed at line %s. Fix the reported error and rerun.\n" "$LINENO" >&2' ERR

usage() {
  cat <<'EOF'
Usage: bash scripts/install.sh [options]
  --torch-index-url URL  Optional official CUDA wheel index; default: detect driver and resolve
  --prefix DIR           Runtime/environment directory (default: PROJECT/.runtime)
  --skip-system-deps     Do not use apt; require curl, git, tar, ffmpeg, ffprobe, flock
  --skip-models          Install environments only; not a complete ready-to-run install
  --skip-gpu-check       Allow preparation on a host without GPU; GPU checks remain pending
  --dry-run              Show the plan without changing files or downloading anything
  -h, --help             Show this help

Run from a clone of delvilo/MimicMotion main on Linux x86_64 (Ubuntu 22.04/24.04 recommended).
The installer does not switch branches or enable dual-GPU inference.
SVD access: accept the model's terms on Hugging Face yourself, then supply HF_TOKEN
through the environment or an existing Hugging Face login. Tokens are never printed.
EOF
}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PREFIX="$PROJECT/.runtime"
SYSTEM=1 MODELS=1 GPU_CHECK=1 DRY_RUN=0
TORCH_INDEX=""
while (($#)); do
  case "$1" in
    --torch-index-url) [[ $# -ge 2 && -n "$2" ]] || { usage >&2; exit 2; }; TORCH_INDEX="$2"; shift 2 ;;
    --prefix) [[ $# -ge 2 && -n "$2" ]] || { usage >&2; exit 2; }; PREFIX="$2"; shift 2 ;;
    --skip-system-deps) SYSTEM=0; shift ;;
    --skip-models) MODELS=0; shift ;;
    --skip-gpu-check) GPU_CHECK=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Linux x86_64 is required.' >&2; exit 2; }
[[ -f "$PROJECT/inference.py" ]] || { echo 'Run this script from the repository scripts directory.' >&2; exit 2; }
if ((DRY_RUN)); then
  printf 'Project: %s\nRuntime: %s\n' "$PROJECT" "$PREFIX"
  printf 'PyTorch index: %s\n' "${TORCH_INDEX:-auto (driver + official wheel index discovery)}"
  printf 'System packages: %s; model downloads: %s; GPU verification: %s\n' "$SYSTEM" "$MODELS" "$GPU_CHECK"
  cat <<'EOF'
Plan: Python 3.11 (managed uv); isolated main and SAM 2 virtual environments.
Main: resolve matching torch/torchvision releases, requiring CUDA runtime > 11.7.
ONNX Runtime: resolve an ABI-compatible range from actual torch CUDA/cuDNN versions.
SAM 2: torch >= 2.5.1 and matching torchvision; same selected CUDA family, isolated environment.
Models: DWPose x2, MimicMotion 1.1, SVD runtime files, SAM 2.1 small, LaMa TorchScript.
Then verify imports / GPU / model loading, record package versions, and create a single-GPU launcher.
No Git branch changes and no driver installation.
EOF
  exit 0
fi
if ((SYSTEM)); then
  command -v apt-get >/dev/null || { echo 'Use --skip-system-deps on non-Debian systems after installing prerequisites.' >&2; exit 2; }
  PRIV=()
  if ((EUID != 0)); then
    command -v sudo >/dev/null || { echo 'sudo is needed for system packages; or use --skip-system-deps.' >&2; exit 2; }
    PRIV=(sudo)
  fi
  "${PRIV[@]}" apt-get update
  "${PRIV[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates curl git tar util-linux ffmpeg libglib2.0-0 libgl1
fi
for exe in curl git tar ffmpeg ffprobe flock sha256sum; do
  command -v "$exe" >/dev/null || { printf 'Missing prerequisite: %s\n' "$exe" >&2; exit 2; }
done
encoders="$(ffmpeg -hide_banner -encoders 2>/dev/null)"
[[ "$encoders" == *libx264rgb* ]] || { echo 'FFmpeg must include libx264rgb.' >&2; exit 2; }
if ((GPU_CHECK)); then
  command -v nvidia-smi >/dev/null || { echo 'A working NVIDIA host driver is required. Use --skip-gpu-check only for preparation.' >&2; exit 2; }
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi
mkdir -p -- "$PREFIX"
PREFIX="$(cd -- "$PREFIX" && pwd)"
exec 9>"$PREFIX/install.lock"
flock -n 9 || { echo 'Another installer is using this runtime directory.' >&2; exit 2; }
mkdir -p "$PREFIX/bin" "$PREFIX/downloads" "$PREFIX/reports"
# A pinned uv binary bootstraps Python even when the host only has Python 3.12.
UV_VERSION=0.6.17
UV_ARCHIVE=uv-x86_64-unknown-linux-gnu.tar.gz
UV_BASE="https://github.com/astral-sh/uv/releases/download/$UV_VERSION"
UV="$PREFIX/bin/uv"
if [[ ! -x "$UV" ]]; then
  curl --fail --location --retry 3 --proto '=https' --tlsv1.2 "$UV_BASE/$UV_ARCHIVE" -o "$PREFIX/downloads/$UV_ARCHIVE.part"
  curl --fail --location --retry 3 --proto '=https' --tlsv1.2 "$UV_BASE/$UV_ARCHIVE.sha256" -o "$PREFIX/downloads/uv.sha256"
  expected="$(awk '{print $1}' "$PREFIX/downloads/uv.sha256")"
  [[ "$expected" =~ ^[a-fA-F0-9]{64}$ ]] || { echo 'Invalid uv checksum response.' >&2; exit 1; }
  actual="$(sha256sum "$PREFIX/downloads/$UV_ARCHIVE.part")"
  [[ "${actual%% *}" == "$expected" ]] || { echo 'uv archive checksum mismatch.' >&2; exit 1; }
  tar -xzf "$PREFIX/downloads/$UV_ARCHIVE.part" -C "$PREFIX/downloads" uv-x86_64-unknown-linux-gnu/uv
  install -m 755 "$PREFIX/downloads/uv-x86_64-unknown-linux-gnu/uv" "$UV"
fi
[[ "$("$UV" --version)" == "uv $UV_VERSION"* ]] || { echo 'Unexpected uv version in runtime directory.' >&2; exit 2; }
export UV_PYTHON_INSTALL_DIR="$PREFIX/python"
export UV_CACHE_DIR="$PREFIX/uv-cache"
"$UV" python install 3.11
BOOTSTRAP_PY="$("$UV" python find 3.11)"
TORCH_INDEX="$("$BOOTSTRAP_PY" "$SCRIPT_DIR/cuda_runtime.py" --select-index "$TORCH_INDEX")"
ENV_ROOT="$PREFIX/envs/${TORCH_INDEX##*/}"
printf 'Selected CUDA wheel index: %s\nEnvironment root: %s\n' "$TORCH_INDEX" "$ENV_ROOT"
for name in main sam2; do
  ENV_DIR="$ENV_ROOT/$name"
  if [[ -e "$ENV_DIR" && ! -f "$ENV_DIR/.mimicmotion-installer" ]]; then
    echo "Refusing to modify an unowned environment: $ENV_DIR" >&2; exit 2
  fi
  if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    # Ownership marker permits recovery if environment creation was interrupted.
    mkdir -p "$ENV_DIR"
    touch "$ENV_DIR/.mimicmotion-installer"
    "$UV" venv --allow-existing --seed --python 3.11 "$ENV_DIR"
    touch "$ENV_DIR/.mimicmotion-installer"
  fi
  "$ENV_DIR/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,11), "Python 3.11 required"'
  "$ENV_DIR/bin/python" -m pip install 'pip==24.3.1' 'setuptools==75.6.0' 'wheel==0.45.1'
done
MAIN_PY="$ENV_ROOT/main/bin/python"
SAM_PY="$ENV_ROOT/sam2/bin/python"
MAIN_TORCH_REQUIREMENT="$("$MAIN_PY" "$SCRIPT_DIR/cuda_runtime.py" --torch-requirement "$TORCH_INDEX" --role main)"
SAM_TORCH_REQUIREMENT="$("$SAM_PY" "$SCRIPT_DIR/cuda_runtime.py" --torch-requirement "$TORCH_INDEX" --role sam2)"
printf 'Resolving PyTorch from %s\n' "$TORCH_INDEX"
"$MAIN_PY" -m pip install --upgrade "$MAIN_TORCH_REQUIREMENT" torchvision --index-url "$TORCH_INDEX"
"$MAIN_PY" -m pip install -r "$SCRIPT_DIR/requirements-main.txt"
"$MAIN_PY" "$SCRIPT_DIR/cuda_runtime.py" --install-ort
"$SAM_PY" -m pip install --upgrade "$SAM_TORCH_REQUIREMENT" 'torchvision>=0.20.1' --index-url "$TORCH_INDEX"
"$SAM_PY" -m pip install -r "$SCRIPT_DIR/requirements-sam2.txt"
SAM_REV=2b90b9f5ceec907a1c18123530e92e794ad901a4
SAM_SOURCE="$PREFIX/sam2-source"
if [[ ! -d "$SAM_SOURCE" ]]; then
  git clone https://github.com/facebookresearch/sam2.git "$SAM_SOURCE"
fi
[[ "$(git -C "$SAM_SOURCE" remote get-url origin)" == https://github.com/facebookresearch/sam2.git ]] || { echo 'Unexpected SAM 2 source remote.' >&2; exit 2; }
[[ -z "$(git -C "$SAM_SOURCE" status --porcelain --untracked-files=no)" ]] || { echo 'SAM 2 source has local edits; preserve them before rerunning.' >&2; exit 2; }
git -C "$SAM_SOURCE" fetch origin "$SAM_REV"
git -C "$SAM_SOURCE" checkout --detach "$SAM_REV"
# Optional postprocessing extension is disabled so nvcc/system CUDA toolkit is not required.
SAM2_BUILD_CUDA=0 "$SAM_PY" -m pip install --no-build-isolation --no-deps "$SAM_SOURCE"
"$MAIN_PY" -m pip check
"$SAM_PY" -m pip check

# Main environment CUDA/cuDNN libraries must not leak into the independent SAM 2 process.
# The SAM launcher strips the exact main-library prefix added by the main launcher.
SAM_LAUNCHER="$PREFIX/bin/sam2-python"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  printf 'SAM_PY=%q\n' "$SAM_PY"
  cat <<'EOF'
if [[ -n "${MIMIC_MAIN_LIBS:-}" ]]; then
  if [[ "${LD_LIBRARY_PATH:-}" == "$MIMIC_MAIN_LIBS" ]]; then
    unset LD_LIBRARY_PATH
  elif [[ "${LD_LIBRARY_PATH:-}" == "$MIMIC_MAIN_LIBS:"* ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#"$MIMIC_MAIN_LIBS:"}"
  fi
fi
unset MIMIC_MAIN_LIBS
exec "$SAM_PY" "$@"
EOF
} > "$SAM_LAUNCHER"
chmod +x "$SAM_LAUNCHER"
MAIN_LIBS="$("$MAIN_PY" - <<'PY'
import site
from pathlib import Path
paths=[]
for root in site.getsitepackages():
    paths += [str(p) for p in Path(root).glob('nvidia/*/lib') if p.is_dir()]
    p=Path(root)/'torch/lib'
    if p.is_dir(): paths.append(str(p))
print(':'.join(paths))
PY
)"
export MIMIC_MAIN_LIBS="$MAIN_LIBS"
export LD_LIBRARY_PATH="$MAIN_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
CHECK_ARGS=()
((GPU_CHECK)) || CHECK_ARGS+=(--skip-gpu-check)
"$MAIN_PY" "$SCRIPT_DIR/install_support.py" --project "$PROJECT" --check-env "${CHECK_ARGS[@]}"
"$SAM_LAUNCHER" "$PROJECT/scripts/replacement_masks.py" --check
if ((GPU_CHECK)); then
  "$SAM_LAUNCHER" -c 'import torch; assert torch.cuda.is_available(); x=torch.ones(8,device="cuda:0"); assert x.sum().item()==8; print("SAM 2 CUDA smoke check passed")'
fi
if ((MODELS)); then
  "$MAIN_PY" "$SCRIPT_DIR/install_support.py" --project "$PROJECT" --download --check-models "${CHECK_ARGS[@]}"
  "$SAM_LAUNCHER" - "$PROJECT" <<'PY'
import sys
from sam2.build_sam import build_sam2_video_predictor
from pathlib import Path
model=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml', str(Path(sys.argv[1])/'models/sam2.1_hiera_small.pt'), device='cpu')
print('SAM 2 configuration and checkpoint loaded successfully on CPU')
PY
fi
"$MAIN_PY" -m pip freeze > "$PREFIX/reports/main-packages.txt"
"$SAM_PY" -m pip freeze > "$PREFIX/reports/sam2-packages.txt"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  printf 'PROJECT=%q\nMAIN_PY=%q\nSAM_LAUNCHER=%q\n' "$PROJECT" "$MAIN_PY" "$SAM_LAUNCHER"
  printf 'export MIMIC_MAIN_LIBS=%q\n' "$MAIN_LIBS"
  cat <<'EOF'
export LD_LIBRARY_PATH="$MIMIC_MAIN_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
cd "$PROJECT"
exec "$MAIN_PY" inference.py \
  --base_model_path "$PROJECT/models/svd" \
  --ckpt_path "$PROJECT/models/MimicMotion_1-1.pth" \
  --sam2_python "$SAM_LAUNCHER" \
  --sam2_checkpoint "$PROJECT/models/sam2.1_hiera_small.pt" \
  --lama_checkpoint "$PROJECT/models/big-lama.pt" "$@"
EOF
} > "$PREFIX/bin/run-mimicmotion"
chmod +x "$PREFIX/bin/run-mimicmotion"
printf '\nEnvironment installation completed. Launcher:\n  %s --ref_video_path VIDEO --ref_image_path IMAGE --device cuda:0\n' "$PREFIX/bin/run-mimicmotion"
if ((!MODELS || !GPU_CHECK)); then
  echo 'Preparation only: skipped model and/or GPU checks must be completed before claiming readiness.'
else
  echo 'Dependency/model smoke checks passed. Real dance-video inference still needs validation.'
fi

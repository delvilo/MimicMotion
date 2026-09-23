#!/usr/bin/env bash
# Colab entry point for the existing main-branch installer and inference launcher.
set -Eeuo pipefail
set +x

usage() {
  cat <<'HELP'
Usage:
  bash scripts/colab.sh --install-only [--torch-index-url URL]
  bash scripts/colab.sh [--run-only] [--torch-index-url URL] -- \
    --ref_video_path /content/dance.mp4 --ref_image_path /content/actor.jpg \
    --output_file /content/result.mp4 [other inference.py options]

Defaults to installing the complete single-GPU environment and running inference.
--install-only        Install and verify dependencies/models, then exit.
--run-only            Run using the already installed environment, without reinstalling.
--torch-index-url URL Select an official CUDA wheel index for the installer.
--skip-system-deps    Skip apt when required system packages are already present.
--dry-run             Show the installation plan and inference command; do not install/run.
--help                Show this help.

Use a Colab GPU runtime and supply access to the gated SVD model before installing:
https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1
Provide a read-only Hugging Face token through HF_TOKEN, never on the command line.
Input/output paths in the inference command are relative to the repository root.
Use local /content for output; copy finished files to Drive to keep them.
See COLAB.md or open MimicMotion_Colab.ipynb for the complete workflow.
HELP
}

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_ONLY=0
RUN_ONLY=0
DRY_RUN=0
INSTALL_ARGS=()
INFER_ARGS=()
while (($#)); do
  case "$1" in
    --install-only) INSTALL_ONLY=1; shift ;;
    --run-only) RUN_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-system-deps) INSTALL_ARGS+=(--skip-system-deps); shift ;;
    --torch-index-url)
      [[ $# -ge 2 && -n "$2" ]] || { usage >&2; exit 2; }
      INSTALL_ARGS+=(--torch-index-url "$2")
      shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; INFER_ARGS=("$@"); break ;;
    *) printf 'Unknown launcher option: %s (pass inference options after --)\n' "$1" >&2
       usage >&2; exit 2 ;;
  esac
done
(( !INSTALL_ONLY || !RUN_ONLY )) || { echo 'Choose either --install-only or --run-only.' >&2; exit 2; }
if ((RUN_ONLY && ${#INSTALL_ARGS[@]})); then
  echo '--torch-index-url and --skip-system-deps only apply during installation.' >&2
  exit 2
fi
if ((INSTALL_ONLY)); then
  ((${#INFER_ARGS[@]} == 0)) || { echo '--install-only does not accept inference arguments.' >&2; exit 2; }
else
  ((${#INFER_ARGS[@]} > 0)) || { echo 'Specify inference options after --, or use --install-only.' >&2; exit 2; }
fi

cd -- "$PROJECT"
if ((!INSTALL_ONLY && !DRY_RUN)); then
  # parse_args uses only the Python standard library and validates modes/extensions.
  # Validate local inputs before installing large dependencies or downloading models.
  python3 - "${INFER_ARGS[@]}" <<'PY'
import sys
from pathlib import Path
from inference import parse_args
args = parse_args(sys.argv[1:])
for source in args.ref_video_path + args.ref_image_path:
    path = Path(source).expanduser()
    if not path.is_file() or not path.stat().st_size:
        raise SystemExit(f'Input file is missing or empty: {path}')
print('Input paths and inference options passed CLI preflight.', flush=True)
PY
fi

INSTALL=(bash "$PROJECT/scripts/install.sh" "${INSTALL_ARGS[@]}")
LAUNCHER="$PROJECT/.runtime/bin/run-mimicmotion"
if ((DRY_RUN)); then
  if ((!RUN_ONLY)); then
    "${INSTALL[@]}" --dry-run
  fi
  if ((!INSTALL_ONLY)); then
    printf 'Inference: '
    printf '%q ' "$LAUNCHER" "${INFER_ARGS[@]}"
    printf '\n'
  fi
  exit 0
fi
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null; then
  echo 'Select a GPU runtime in Colab and reconnect before installing or running.' >&2
  exit 2
fi
if ((!RUN_ONLY)); then
  "${INSTALL[@]}"
fi
if ((!INSTALL_ONLY)); then
  [[ -x "$LAUNCHER" ]] || { echo 'Launcher missing; run bash scripts/colab.sh --install-only first.' >&2; exit 2; }
  exec "$LAUNCHER" "${INFER_ARGS[@]}"
fi

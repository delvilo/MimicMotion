"""Resolve CUDA wheel families and ONNX Runtime ABI ranges, without exact torch pins."""
import argparse
import importlib.metadata
import re
import subprocess
import sys
import urllib.request

BASE = 'https://download.pytorch.org/whl'


def cuda_version(value):
    if not value or not re.fullmatch(r'\d+\.\d+(?:\.\d+)?', value):
        raise ValueError(f'A CUDA-enabled PyTorch build is required (reported CUDA: {value!r})')
    version = tuple(int(v) for v in value.split('.')[:2])
    if version <= (11, 7):
        raise ValueError('This installation requires PyTorch CUDA runtime > 11.7')
    return version


def wheel_family(url):
    match = re.fullmatch(re.escape(BASE) + r'/cu(\d{3})/?', url)
    if not match:
        raise ValueError('Use an official CUDA wheel index such as https://download.pytorch.org/whl/cu126')
    digits = match.group(1)
    return cuda_version(f'{int(digits[:-1])}.{int(digits[-1])}')


def driver_cuda(output):
    match = re.search(r'CUDA Version:\s*(\d+\.\d+)', output)
    if not match:
        raise ValueError('Cannot determine driver CUDA support; supply --torch-index-url explicitly')
    return cuda_version(match.group(1))


def choose_index(html, maximum):
    families = set(re.findall(r'(?:href=["\'][^"\']*)?\bcu(\d{3})(?:/|["\'])', html))
    candidates = []
    for digits in families:
        version = (int(digits[:-1]), int(digits[-1]))
        if (11, 7) < version <= maximum and version[0] in (11, 12, 13):
            candidates.append((version, f'{BASE}/cu{digits}'))
    if not candidates:
        raise ValueError('No compatible CUDA > 11.7 wheel index found; check driver or set --torch-index-url')
    return max(candidates)[1]


def select_index(explicit):
    if explicit:
        wheel_family(explicit)
        return explicit.rstrip('/')
    try:
        output = subprocess.check_output(['nvidia-smi'], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError('GPU/driver detection is unavailable; set --torch-index-url for the intended target machine') from exc
    maximum = driver_cuda(output)
    with urllib.request.urlopen(BASE + '/', timeout=30) as response:
        html = response.read().decode('utf-8')
    return choose_index(html, maximum)


def torch_requirement(index, role):
    family = wheel_family(index)
    if role == 'sam2':
        return 'torch>=2.5.1'
    # CUDA 11 ONNX wheels need cuDNN 8, as supplied by pre-2.4 torch.
    # A compatibility interval is necessary here, not an exact release pin.
    return 'torch>=2.0,<2.4' if family[0] == 11 else 'torch>=2.4'


def ort_requirement(cuda, cudnn):
    family = cuda_version(cuda)
    if not cudnn:
        raise ValueError('PyTorch has no usable cuDNN runtime')
    major = cudnn // 10000 if cudnn >= 90000 else cudnn // 1000
    if family[0] == 11 and major == 8:
        return 'onnxruntime-gpu>=1.16,<1.19'
    if family[0] == 12 and major == 9:
        return 'onnxruntime-gpu>=1.19,<1.27'
    if family[0] == 13 and major == 9:
        return 'onnxruntime-gpu>=1.27,<1.30'
    raise ValueError(f'No verified ONNX Runtime wheel family for CUDA {cuda} / cuDNN {major}. '
                     'Choose a supported official wheel index and rerun the installer. '
                     'CUDA 11.8 requires cuDNN 8 in the main environment; SAM 2 stays separate.')


def installed_selection():
    import torch
    cuda = torch.version.cuda
    cudnn = torch.backends.cudnn.version()
    return dict(torch=torch.__version__, cuda=cuda, cudnn=cudnn,
                ort_requirement=ort_requirement(cuda, cudnn))


def verify_installed():
    from packaging.requirements import Requirement
    selection = installed_selection()
    required = Requirement(selection['ort_requirement'])
    actual = importlib.metadata.version('onnxruntime-gpu')
    if actual not in required.specifier:
        raise ValueError(f'Installed ONNX Runtime {actual} conflicts with {selection}; '
                         'run python scripts/cuda_runtime.py --install-ort')
    return selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--select-index', nargs='?', const='', default=None)
    parser.add_argument('--torch-requirement')
    parser.add_argument('--role', choices=('main', 'sam2'), default='main')
    parser.add_argument('--install-ort', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.select_index is not None:
        print(select_index(args.select_index))
    elif args.torch_requirement:
        print(torch_requirement(args.torch_requirement, args.role))
    elif args.install_ort:
        selection = installed_selection()
        print('Runtime selection:', selection, flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', selection['ort_requirement']], check=True)
        print('Runtime ABI check:', verify_installed())
    elif args.verify:
        print(verify_installed())
    else:
        parser.error('Choose --select-index, --torch-requirement, --install-ort or --verify')


if __name__ == '__main__':
    main()

"""Installer model downloads and smoke checks. Never prints authentication tokens."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def download_url(url, destination, recorded):
    previous=recorded.get(destination.name,{})
    if destination.is_file() and previous.get('sha256') == digest(destination):
        print(f'Already downloaded: {destination.name}')
        return
    # Preserve pre-existing model files; validation below will detect invalid contents.
    if destination.is_file() and not previous:
        print(f'Using existing model (will validate): {destination.name}')
        return
    request=urllib.request.Request(url,headers={'User-Agent':'MimicMotion-installer/1.0'})
    partial=destination.with_name(destination.name+'.download')
    try:
        with urllib.request.urlopen(request,timeout=90) as response, partial.open('wb') as output:
            shutil.copyfileobj(response,output,8*1024*1024)
        if partial.stat().st_size < 1024:
            raise ValueError(f'Download is unexpectedly small: {destination.name}')
        os.replace(partial,destination)
    finally:
        partial.unlink(missing_ok=True)


def download_models(project):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
    models=project/'models'; models.mkdir(exist_ok=True)
    manifest=models/'install-manifest.json'
    recorded=json.loads(manifest.read_text()).get('files',{}) if manifest.exists() else {}
    api=HfApi(); revisions={}
    svd_repo='stabilityai/stable-video-diffusion-img2vid-xt-1-1'
    try:
        revisions[svd_repo]=api.model_info(svd_repo).sha
        # Check gated access before downloading the remaining multi-GB artifacts.
        hf_hub_download(svd_repo,'model_index.json',revision=revisions[svd_repo])
    except HfHubHTTPError as exc:
        raise RuntimeError('SVD download access failed. Open https://huggingface.co/'+svd_repo+
                           ', obtain model access, then set HF_TOKEN (read access) or log in with Hugging Face. '
                           'The installer never accepts the model terms for you.') from None
    svd_dir=models/'svd'
    snapshot_download(svd_repo,revision=revisions[svd_repo],local_dir=str(svd_dir),
                      allow_patterns=['model_index.json','unet/config.json','vae/config.json',
                                      'vae/*fp16.safetensors','image_encoder/config.json',
                                      'image_encoder/*fp16.safetensors','scheduler/*','feature_extractor/*'])
    hf_models=[('yzd-v/DWPose','yolox_l.onnx',models/'DWPose/yolox_l.onnx'),
               ('yzd-v/DWPose','dw-ll_ucoco_384.onnx',models/'DWPose/dw-ll_ucoco_384.onnx'),
               ('tencent/MimicMotion','MimicMotion_1-1.pth',models/'MimicMotion_1-1.pth')]
    for repo,name,target in hf_models:
        if repo not in revisions:
            revisions[repo]=api.model_info(repo).sha
        source=Path(hf_hub_download(repo,name,revision=revisions[repo]))
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_file():
            if digest(target) != digest(source):
                raise ValueError(f'Existing model differs from downloaded upstream file: {target}. Move it aside explicitly if replacement is intended.')
        else:
            temporary=target.with_name(target.name+'.download')
            shutil.copyfile(source,temporary);os.replace(temporary,target)
    urls={
        'sam2.1_hiera_small.pt':'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt',
        'big-lama.pt':'https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt'}
    for name,url in urls.items():
        download_url(url,models/name,recorded)
    paths=[target for _,_,target in hf_models]+[models/name for name in urls]
    paths += [p for p in svd_dir.rglob('*') if p.is_file() and '.cache' not in p.parts]
    records={str(p.relative_to(models)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in paths}
    manifest.write_text(json.dumps(dict(huggingface_revisions=revisions,direct_urls=urls,files=records),indent=2)+'\n')
    print('Model downloads completed; revisions and local SHA-256 hashes recorded.')


def check_environment(project, skip_gpu):
    from cuda_runtime import verify_installed
    print("Selected runtime:",verify_installed())
    import numpy
    import torch
    import torchvision
    import cv2
    import onnxruntime as ort
    import av
    from mimicmotion.utils.loader import create_pipeline
    from mimicmotion.utils.replacement import render
    if not skip_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError('PyTorch cannot use CUDA. Check NVIDIA driver and GPU visibility.')
        conv=torch.nn.Conv2d(3,4,3).cuda().half()
        output=conv(torch.ones(1,3,16,16,device='cuda',dtype=torch.float16))
        assert torch.isfinite(output).all()
        del conv,output
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('ONNX Runtime has no CUDAExecutionProvider')
        print('CUDA FP16 convolution passed:',torch.cuda.get_device_name(0))
    print('Main imports passed:',torch.__version__,torchvision.__version__,ort.__version__)


def check_models(project, skip_gpu):
    import torch
    import numpy as np
    import onnxruntime as ort
    from diffusers import AutoencoderKLTemporalDecoder
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from diffusers.schedulers import EulerDiscreteScheduler
    from mimicmotion.modules.unet import UNetSpatioTemporalConditionModel
    models=project/'models'
    # Validate precisely the local files that create_pipeline reads, without GPU generation.
    svd=str(models/'svd')
    UNetSpatioTemporalConditionModel.load_config(svd,subfolder='unet',local_files_only=True)
    CLIPImageProcessor.from_pretrained(svd,subfolder='feature_extractor',local_files_only=True)
    EulerDiscreteScheduler.from_pretrained(svd,subfolder='scheduler',local_files_only=True)
    vae=AutoencoderKLTemporalDecoder.from_pretrained(svd,subfolder='vae',variant='fp16',torch_dtype=torch.float16,local_files_only=True)
    del vae
    encoder=CLIPVisionModelWithProjection.from_pretrained(svd,subfolder='image_encoder',variant='fp16',torch_dtype=torch.float16,local_files_only=True)
    del encoder
    state=torch.load(models/'MimicMotion_1-1.pth',map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or not any(k.startswith('pose_net.') for k in state):
        raise ValueError('Invalid MimicMotion state dictionary')
    del state
    lama=torch.jit.load(str(models/'big-lama.pt'),map_location='cpu').eval()
    with torch.inference_mode():
        image=torch.zeros(1,3,64,64);mask=torch.zeros(1,1,64,64);mask[:,:,16:32,16:32]=1
        result=lama(image,mask)
        assert tuple(result.shape)==(1,3,64,64) and torch.isfinite(result).all()
    del lama,result
    providers=['CPUExecutionProvider'] if skip_gpu else [('CUDAExecutionProvider',{'device_id':0}),'CPUExecutionProvider']
    for name in ('yolox_l.onnx','dw-ll_ucoco_384.onnx'):
        session=ort.InferenceSession(str(models/'DWPose'/name),providers=providers)
        if not skip_gpu and session.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('DWPose silently fell back to CPU; check CUDA/cuDNN shared libraries')
        input_info=session.get_inputs()[0]
        if not all(isinstance(n,int) and n>0 for n in input_info.shape):
            raise ValueError(f'Unexpected dynamic input for {name}')
        value=np.zeros(input_info.shape,dtype=np.float32)
        session.run(None,{input_info.name:value})
        del session
    print('SVD components, MimicMotion weights, LaMa forward and DWPose sessions passed.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--check-env',action='store_true')
    parser.add_argument('--check-models',action='store_true')
    parser.add_argument('--skip-gpu-check',action='store_true')
    args=parser.parse_args();project=args.project.resolve()
    sys.path.insert(0,str(project))
    if args.check_env:check_environment(project,args.skip_gpu_check)
    if args.download:download_models(project)
    if args.check_models:check_models(project,args.skip_gpu_check)


if __name__=='__main__':
    main()

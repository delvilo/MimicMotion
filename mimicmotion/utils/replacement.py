"""Source-first actor replacement; generation never supplies the visible background."""
import copy
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)






def check_dependencies(args):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise ValueError('Replacement requires ffmpeg and ffprobe on PATH')
    transparent = args.transparent_background
    if transparent:
        from mimicmotion.utils.transparent import check_encoder
        check_encoder(args.alpha_codec)
    if not transparent and not args.no_ai_background and not Path(args.lama_checkpoint).is_file():
        raise ValueError(f'LaMa checkpoint missing: {args.lama_checkpoint}; supply it or use --no_ai_background')
    if not (args.replacement_masks and (transparent or args.source_masks)):
        if not Path(args.sam2_checkpoint).is_file():
            raise ValueError(f'SAM 2 checkpoint missing: {args.sam2_checkpoint}')
        worker = Path(__file__).resolve().parents[2] / 'scripts' / 'replacement_masks.py'
        result = subprocess.run([args.sam2_python, str(worker), '--check'], capture_output=True, text=True)
        if result.returncode:
            raise ValueError(f'SAM 2 environment unavailable: {result.stderr[-2000:]}')
    if not transparent:
        import cv2  # ORB camera registration, already used by DWPose.


def probe_source(video):
    result = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                             'stream=width,height,avg_frame_rate,r_frame_rate,start_time,sample_aspect_ratio:stream_tags=rotate:stream_side_data=rotation',
                             '-of','json',str(video)], capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout)['streams']
    if not streams:
        raise ValueError('No video stream')
    info = streams[0]
    if any(abs(float(item.get('rotation', 0))) > 0.01 for item in info.get('side_data_list', [])) or float(info.get('tags', {}).get('rotate', 0)):
        raise ValueError('Bake the video rotation metadata before replacement')
    if abs(float(info.get('start_time', 0))) > 0.01:
        raise ValueError('Normalize the source video start timestamp to zero before replacement')
    from fractions import Fraction
    try:
        rate = Fraction(info['avg_frame_rate'])
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError('Source has no valid average frame rate') from exc
    if rate <= 0:
        raise ValueError('Source frame rate must be positive')
    if info.get('sample_aspect_ratio','1:1') not in ('1:1','0:1','N/A'):
        raise ValueError('Normalize non-square pixels before replacement')
    return info


def mask_paths(directory, count):
    paths = [Path(directory) / f'{i:06d}.png' for i in range(count)]
    if any(not p.is_file() for p in paths):
        raise ValueError(f'Mask directory must contain 000000.png through {count-1:06d}.png: {directory}')
    return paths


def read_mask(path, shape):
    array = np.asarray(Image.open(path).convert('L'))
    if array.shape != tuple(shape):
        raise ValueError(f'Mask size mismatch: {path}; expected {shape}')
    return array


def dilate(mask, padding):
    if padding == 0:
        return mask.astype(bool)
    return np.asarray(Image.fromarray(mask.astype('uint8') * 255).filter(ImageFilter.MaxFilter(2 * padding + 1))) > 0


def foreground_alpha(mask, radius):
    raw = mask.astype(np.float32) / 255
    if radius == 0:
        return raw
    blurred = np.asarray(Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255
    # Never extend the generated foreground into pixels outside its mask.
    return np.minimum(raw, blurred)


def box_iou(a, b):
    lo = np.maximum(a[:2], b[:2]); hi = np.minimum(a[2:], b[2:])
    area = np.prod(np.maximum(hi-lo, 0))
    return float(area / max(1e-6, np.prod(a[2:]-a[:2]) + np.prod(b[2:]-b[:2]) - area))


def select_actor(pose, previous_box, width, height):
    bodies = pose['bodies']; boxes = []
    for i, subset in enumerate(bodies['subset']):
        indices = subset[subset >= 0].astype(int)
        if len(indices) < 4:
            continue
        points = bodies['candidate'][indices] * [width, height]
        if not np.isfinite(points).all():
            continue
        lo, hi = points.min(0), points.max(0)
        margin = np.maximum((hi-lo)*0.18, 8)
        box = np.r_[np.maximum(lo-margin, 0), np.minimum(hi+margin, [width-1, height-1])]
        boxes.append((i, box))
    if not boxes:
        raise ValueError('No sufficiently visible actor pose; tracking cannot continue safely')
    if previous_box is None:
        if len(boxes) != 1:
            raise ValueError('Multiple people in first frame; use --person_box X0 Y0 X1 Y1')
        index, box = boxes[0]
    else:
        ranked = sorted(((box_iou(b, previous_box), i, b) for i, b in boxes), key=lambda x:x[0], reverse=True)
        if ranked[0][0] < 0.1 or (len(ranked) > 1 and ranked[0][0]-ranked[1][0] < 0.05):
            raise ValueError('Actor tracking lost or ambiguous; split the shot or supply clearer footage')
        _, index, box = ranked[0]
    selected = copy.deepcopy(pose)
    selected['bodies'] = dict(candidate=bodies['candidate'][index*18:(index+1)*18].copy(),
                              subset=np.where(bodies['subset'][index:index+1] >= 0, bodies['subset'][index:index+1]-18*index, -1),
                              score=bodies['score'][index:index+1].copy())
    count = len(bodies['subset'])
    selected['faces'] = pose['faces'][index:index+1].copy()
    selected['faces_score'] = pose['faces_score'][index:index+1].copy()
    selected['hands'] = pose['hands'][[index, index+count]].copy()
    selected['hands_score'] = pose['hands_score'][[index, index+count]].copy()
    return selected, box, [b.tolist() for other, b in boxes if other != index]


def geometry(width, height, resolution):
    scale = resolution / min(width, height)
    rw, rh = round(width * scale), round(height * scale)
    cw, ch = math.ceil(rw/64)*64, math.ceil(rh/64)*64
    return dict(width=width, height=height, resized_width=rw, resized_height=rh,
                canvas_width=cw, canvas_height=ch, x=(cw-rw)//2, y=(ch-rh)//2)


def prepare(task, args, processor, directory):
    import decord
    import torch
    from mimicmotion.dwpose.util import draw_pose
    from mimicmotion.dwpose.preprocess import get_image_pose
    from PIL import ImageOps
    reader = decord.VideoReader(task['video'], ctx=decord.cpu(0))
    first = reader[0].asnumpy(); h, w = first.shape[:2]
    g = geometry(w, h, args.resolution)
    source_dir = directory / 'source'; source_dir.mkdir()
    reference = Image.open(task['image']).convert('RGB')
    reference = ImageOps.pad(reference, (g['canvas_width'],g['canvas_height']), color=(127,127,127))
    ref_array = np.asarray(reference)
    poses = [get_image_pose(ref_array, processor=processor)]
    previous = np.asarray(args.person_box) if args.person_box else None
    boxes, blocked = [], []
    for i in range(len(reader)):
        rgb = reader[i].asnumpy()
        pose, previous, all_boxes = select_actor(processor(rgb), previous, w, h)
        boxes.append(previous.tolist()); blocked.append(all_boxes)
        # Coordinates remain those of the source camera; no alignment to reference body proportions.
        for key, field in (('bodies','candidate'), ('faces',None), ('hands',None)):
            points = pose[key][field] if field else pose[key]
            points[...,0] = (points[...,0]*g['resized_width'] + g['x'])/g['canvas_width']
            points[...,1] = (points[...,1]*g['resized_height'] + g['y'])/g['canvas_height']
        poses.append(draw_pose(pose, g['canvas_height'], g['canvas_width']))
        # JPEGs are segmentation inputs only; final RGB pixels are read again from the original video.
        if not args.transparent_background:
            Image.fromarray(rgb).save(source_dir/f'{i:06d}.jpg', quality=95)
    (directory/'prompts.json').write_text(json.dumps(dict(boxes=boxes, blockers=blocked, geometry=g)))
    pose_tensor = torch.from_numpy(np.stack(poses).copy()).float() / 127.5 - 1
    image_tensor = torch.from_numpy(np.asarray(reference).copy().transpose(2,0,1)[None]).float() / 127.5 - 1
    return pose_tensor, image_tensor, g


def restore_frame(frame, g):
    image = Image.fromarray(frame)
    image = image.crop((g['x'],g['y'],g['x']+g['resized_width'],g['y']+g['resized_height']))
    return np.asarray(image.resize((g['width'],g['height']), Image.Resampling.LANCZOS))


def segment(frames, args, directory, g, generated_only=False):
    generated = directory/'generated'; generated.mkdir()
    for i, frame in enumerate(frames):
        rgb = restore_frame(frame.permute(1,2,0).numpy(), g)
        Image.fromarray(rgb).save(generated/f'{i:06d}.jpg', quality=95)
    jobs = []
    inputs = [('generated',args.replacement_masks)] if generated_only else [
        ('source',args.source_masks), ('generated',args.replacement_masks)]
    for kind, override in inputs:
        if override:
            mask_paths(override,len(frames))
        else:
            jobs.append(dict(images=str(directory/kind), masks=str(directory/(kind+'_masks'))))
    if jobs:
        job_file = directory/'segmentation.json'
        job_file.write_text(json.dumps(dict(jobs=jobs, prompts=str(directory/'prompts.json'))))
        worker = Path(__file__).resolve().parents[2]/'scripts'/'replacement_masks.py'
        log_path = directory/'sam2.log'
        with log_path.open('w') as log:
            result = subprocess.run([args.sam2_python,str(worker),'--jobs',str(job_file),
                                     '--checkpoint',str(Path(args.sam2_checkpoint).resolve()),
                                     '--model_config',args.sam2_model_config,'--device',args.device],
                                    stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('SAM 2 failed: '+log_path.read_text()[-3000:])

    return (args.source_masks or directory/'source_masks',
            args.replacement_masks or directory/'generated_masks')


class SourceBackground:
    """Conservative planar camera registration using only unoccluded original pixels."""
    def __init__(self, reader, source_masks, blockers, padding, candidate_limit=0):
        import cv2
        from functools import lru_cache
        self.cv2 = cv2; self.reader = reader; self.paths = mask_paths(source_masks,len(reader))
        self.blockers = blockers; self.padding = padding; self.limit = candidate_limit
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.features = lru_cache(maxsize=128)(self._features)
        self.blocked = lru_cache(maxsize=32)(self._blocked)

    def _blocked(self, index):
        rgb = self.reader[index].asnumpy(); h,w = rgb.shape[:2]
        mask = dilate(read_mask(self.paths[index],(h,w))>127,self.padding+2)
        for box in self.blockers[index]:
            x0,y0,x1,y1 = np.rint(box).astype(int)
            mask[max(0,y0):min(h,y1+1),max(0,x0):min(w,x1+1)] = True
        return mask

    def _features(self, index):
        cv2 = self.cv2
        rgb = self.reader[index].asnumpy()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        points, descriptors = self.orb.detectAndCompute(gray, (~self.blocked(index)).astype('uint8')*255)
        return np.float32([p.pt for p in points]).reshape(-1,2), descriptors

    def registered(self, donor, target, target_rgb, hole):
        cv2 = self.cv2
        src_pts, src_desc = self.features(donor); dst_pts, dst_desc = self.features(target)
        if src_desc is None or dst_desc is None or min(len(src_desc),len(dst_desc)) < 20:
            return None
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(src_desc,dst_desc,k=2)
        good = [pair[0] for pair in matches if len(pair)==2 and pair[0].distance < 0.7*pair[1].distance]
        if len(good) < 20:
            return None
        src = np.float32([src_pts[m.queryIdx] for m in good]); dst = np.float32([dst_pts[m.trainIdx] for m in good])
        homography, inliers = cv2.findHomography(src,dst,cv2.RANSAC,2.5)
        if homography is None or not np.isfinite(homography).all() or inliers.sum() < 15 or inliers.mean() < .55:
            return None
        h,w = target_rgb.shape[:2]
        rgb = self.reader[donor].asnumpy()
        valid = cv2.warpPerspective((~self.blocked(donor)).astype('uint8')*255, homography,(w,h),flags=cv2.INTER_NEAREST)>127
        # Erode interpolation borders so no actor pixel is sampled at their edge.
        valid = cv2.erode(valid.astype('uint8'),np.ones((3,3),np.uint8))>0
        warped = cv2.warpPerspective(rgb,homography,(w,h),flags=cv2.INTER_LINEAR)
        known = valid & ~self.blocked(target)
        if np.count_nonzero(known) < 100:
            return None
        error = np.abs(warped.astype(np.float32)-target_rgb).mean(2)
        if np.median(error[known]) > 15 or np.percentile(error[known],90) > 45:
            return None
        # Validate alignment close to the actor, not just on distant scenery.
        rim = dilate(hole,12) & ~hole & known
        if np.count_nonzero(rim) < 32 or np.median(error[rim]) > 18:
            return None
        return warped, valid

    def fill(self, index, source, hole):
        background = source.copy(); remaining = hole.copy(); used = []
        candidates = sorted((j for j in range(len(self.reader)) if j != index),key=lambda j:abs(j-index))
        if self.limit and len(candidates) > self.limit:
            # Nearby frames first, then spread the remaining budget across the entire shot.
            near = candidates[:max(1,self.limit//2)]
            rest = [j for j in candidates if j not in near]
            picks = np.linspace(0,len(rest)-1,self.limit-len(near),dtype=int)
            candidates = near + [rest[j] for j in picks]
        for donor in candidates:
            if not remaining.any():
                break
            aligned = self.registered(donor,index,source,hole)
            if aligned is None:
                continue
            warped, valid = aligned
            take = remaining & valid
            if take.any():
                background[take] = warped[take]
                remaining[take] = False
                used.append(dict(frame=donor,pixels=int(take.sum())))
        return background, remaining, used


class LamaFill:
    """Load the documented two-tensor Simple-LaMa TorchScript export only on demand."""
    def __init__(self, checkpoint, device):
        self.checkpoint = checkpoint; self.device = device; self.model = None

    def __call__(self, background, missing):
        import torch
        if not missing.any():
            return background
        if self.model is None:
            self.model = torch.jit.load(self.checkpoint,map_location=self.device).eval().to(self.device)
        ys,xs = np.where(missing); h,w = missing.shape
        x0,x1 = max(0,int(xs.min())-128),min(w,int(xs.max())+129)
        y0,y1 = max(0,int(ys.min())-128),min(h,int(ys.max())+129)
        crop = background[y0:y1,x0:x1]; mask = missing[y0:y1,x0:x1]
        ch,cw = mask.shape
        # The model expects dimensions divisible by eight.
        image = np.pad(crop,((0,(-ch)%8),(0,(-cw)%8),(0,0)),mode='edge')
        padded_mask = np.pad(mask,((0,(-ch)%8),(0,(-cw)%8)),mode='constant')
        image = torch.from_numpy(image.transpose(2,0,1).copy()).float()[None].to(self.device)/255
        mask_tensor = torch.from_numpy(padded_mask.astype('float32'))[None,None].to(self.device)
        with torch.inference_mode():
            prediction = self.model(image,mask_tensor)
        restored = prediction[0].permute(1,2,0).detach().cpu().float().numpy()[:ch,:cw]
        restored = np.rint(np.clip(restored,0,1)*255).astype('uint8')
        result = background.copy()
        view = result[y0:y1,x0:x1]
        view[mask] = restored[mask]
        return result


def composite(source, foreground, old_mask, alpha, background, protected=None):
    """All pixels outside old/new actor support remain exactly equal to source RGB."""
    protected = np.zeros(old_mask.shape,bool) if protected is None else protected
    support = (old_mask | (alpha>0)) & ~protected
    base = source.copy()
    removal = old_mask & ~protected
    base[removal] = background[removal]
    blended = np.rint(foreground.astype(np.float32)*alpha[...,None] + base*(1-alpha[...,None])).clip(0,255).astype('uint8')
    result = source.copy(); result[support] = blended[support]
    return result, support


def render(frames, task, args, directory, g, temp_output):
    import decord
    reader = decord.VideoReader(task['video'],ctx=decord.cpu(0))
    if len(reader) != len(frames):
        raise ValueError('Generated/source frame counts differ; refusing to change timing')
    source_dir, generated_dir = segment(frames,args,directory,g)
    source_paths = mask_paths(source_dir,len(frames)); generated_paths = mask_paths(generated_dir,len(frames))
    protect_paths = mask_paths(args.occlusion_masks,len(frames)) if args.occlusion_masks else None
    prompts = json.loads((directory/'prompts.json').read_text())
    background = SourceBackground(reader,source_dir,prompts['blockers'],args.mask_padding,args.background_candidates)
    lama = LamaFill(args.lama_checkpoint,args.device)
    info = task['media']['source_stream']; fps = info['avg_frame_rate']
    cmd = ['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f"{g['width']}x{g['height']}",
           '-framerate',fps,'-i','pipe:0','-i',task['video'],'-map','0:v:0']
    if not args.mute:
        cmd += ['-map','1:a:0?','-c:a','copy']
    cmd += ['-c:v','libx264rgb','-crf','0','-preset','medium','-pix_fmt','rgb24','-movflags','+faststart',str(temp_output)]
    rows = []; provenance = directory/'provenance'
    if args.keep_intermediates:
        provenance.mkdir()
    error_path = directory/'ffmpeg.log'
    with error_path.open('wb') as err:
        proc = subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=err)
        try:
            for i, frame in enumerate(frames):
                source = reader[i].asnumpy(); shape = source.shape[:2]
                foreground = restore_frame(frame.permute(1,2,0).numpy(),g)
                old = dilate(read_mask(source_paths[i],shape)>127,args.mask_padding)
                alpha = foreground_alpha(read_mask(generated_paths[i],shape),args.edge_feather)
                if not old.any() or not (alpha>.5).any():
                    raise ValueError(f'Empty actor mask at frame {i}; review masks before continuing')
                protected = read_mask(protect_paths[i],shape)>127 if protect_paths else np.zeros(shape,bool)
                hole = old & (alpha<.999) & ~protected
                repaired, missing, donors = background.fill(i,source,hole)
                if missing.any():
                    if args.no_ai_background:
                        raise ValueError(f'Frame {i}: {missing.sum()} pixels not found in original footage; AI fallback disabled')
                    repaired = lama(repaired,missing)
                output,support = composite(source,foreground,old,alpha,repaired,protected)
                if not np.array_equal(output[~support],source[~support]):
                    raise AssertionError('Source background preservation invariant violated')
                proc.stdin.write(output.tobytes())
                rows.append(dict(frame=i,original_frame_pixels=int((~support).sum()),
                                 temporal_background_pixels=int((hole & ~missing).sum()),
                                 ai_background_pixels=int(missing.sum()),donors=donors))
                if args.keep_intermediates:
                    # 0 original; 85 other original frame; 170 AI; 255 generated actor.
                    provenance_map = np.zeros(shape,np.uint8)
                    provenance_map[hole & ~missing] = 85; provenance_map[missing] = 170
                    provenance_map[(alpha>0)&~protected] = 255
                    Image.fromarray(provenance_map).save(provenance/f'{i:06d}.png')
                if i % 10 == 0:
                    logger.info('Composite %d/%d: original-frame repair=%d, AI repair=%d pixels',i+1,len(frames),(hole & ~missing).sum(),missing.sum())
            proc.stdin.close()
            if proc.wait() != 0:
                raise OSError('ffmpeg failed: '+error_path.read_text()[-2000:])
        except BaseException:
            proc.kill(); proc.wait()
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            raise
    return dict(mode='replace',fps=fps,source_resolution=[g['width'],g['height']],
                background_strategy='same original frame -> registered original frames -> LaMa remaining holes',
                encoder='libx264rgb CRF 0',audio='omitted' if args.mute else 'first original audio stream copied if present',
                per_frame=rows,temporal_background_pixels=sum(r['temporal_background_pixels'] for r in rows),
                ai_background_pixels=sum(r['ai_background_pixels'] for r in rows))

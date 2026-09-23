"""Straight-alpha actor export. No source background recovery or inpainting."""
import json
from pathlib import Path
import subprocess

import numpy as np


CODECS = {
    'prores4444': ['-c:v', 'prores_ks', '-profile:v', '4', '-pix_fmt', 'yuva444p10le',
                   '-alpha_bits', '16'],
    'qtrle': ['-c:v', 'qtrle', '-pix_fmt', 'argb'],
}


def check_encoder(codec):
    result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                            capture_output=True, text=True, check=True)
    encoder = CODECS[codec][1]
    if not any(len(row.split()) > 1 and row.split()[1] == encoder
               for row in result.stdout.splitlines()):
        raise ValueError(f'FFmpeg encoder {encoder} is unavailable; install a build with this encoder')


def encode_rgba(frames, output, width, height, fps, codec, directory, audio_source=None):
    """Stream uint8 RGBA frames to MOV; preserve fractional FPS and optional audio."""
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgba',
           '-s', f'{width}x{height}', '-framerate', str(fps), '-i', 'pipe:0']
    if audio_source:
        cmd += ['-i', str(audio_source)]
    cmd += ['-map', '0:v:0']
    if audio_source:
        # MOV accepts PCM regardless of the original audio codec.
        cmd += ['-map', '1:a:0?', '-c:a', 'pcm_s16le']
    cmd += CODECS[codec] + ['-movflags', '+faststart', '-f', 'mov', str(output)]
    error_path = Path(directory) / 'ffmpeg-alpha.log'
    count = 0
    with error_path.open('wb') as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err)
        try:
            for frame in frames:
                if frame.dtype != np.uint8 or frame.shape != (height, width, 4):
                    raise ValueError('Alpha encoder requires uint8 RGBA frames at the output resolution')
                proc.stdin.write(frame.tobytes())
                count += 1
            proc.stdin.close()
            if proc.wait() != 0:
                raise OSError('Alpha encoder failed: ' + error_path.read_text()[-2000:])
            if not count or not Path(output).stat().st_size:
                raise OSError('Alpha encoder produced no video frames')
        except BaseException:
            proc.kill()
            proc.wait()
            if not proc.stdin.closed:
                proc.stdin.close()
            raise
    return count


def render_transparent(frames, task, args, directory, g, output, processor):
    from mimicmotion.utils.replacement import (
        foreground_alpha, mask_paths, read_mask, restore_frame, segment, select_actor,
    )
    if args.mode == 'replace':
        if len(frames) != task['media']['source_frames']:
            raise ValueError('Generated/source frame counts differ; refusing to change timing')
        fps = task['media']['source_stream']['avg_frame_rate']
    else:
        height, width = frames.shape[-2:]
        g = dict(width=width, height=height, resized_width=width, resized_height=height,
                 canvas_width=width, canvas_height=height, x=0, y=0)
        fps = args.fps
        if not args.replacement_masks:
            boxes = []
            previous = np.asarray(args.person_box) if args.person_box else None
            try:
                for frame in frames:
                    rgb = frame.permute(1, 2, 0).numpy()
                    _, previous, _ = select_actor(processor(rgb), previous, width, height)
                    boxes.append(previous.tolist())
            finally:
                processor.release_memory()
            (directory / 'prompts.json').write_text(json.dumps(dict(boxes=boxes)))
    if args.replacement_masks:
        # Reviewed masks require neither SAM 2 nor intermediate JPEGs.
        masks = mask_paths(args.replacement_masks, len(frames))
    else:
        _, generated = segment(frames, args, directory, g, generated_only=True)
        masks = mask_paths(generated, len(frames))

    def rgba_frames():
        for index, frame in enumerate(frames):
            rgb = restore_frame(frame.permute(1, 2, 0).numpy(), g)
            alpha = foreground_alpha(read_mask(masks[index], rgb.shape[:2]), args.edge_feather)
            if not (alpha > .5).any():
                raise ValueError(f'Empty actor mask at frame {index}; review --replacement_masks')
            alpha = np.rint(alpha * 255).astype(np.uint8)
            rgba = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
            rgba[..., :3] = rgb  # Straight alpha: do not multiply RGB by alpha.
            rgba[..., 3] = alpha
            rgba[alpha == 0, :3] = 0
            yield rgba

    audio = task['video'] if args.mode == 'replace' and not args.mute else None
    count = encode_rgba(rgba_frames(), output, g['width'], g['height'], fps,
                        args.alpha_codec, directory, audio)
    return dict(transparent_background=True, alpha='straight', alpha_codec=args.alpha_codec,
                container='mov', frames=count, fps=fps, resolution=[g['width'], g['height']],
                background_strategy='none', ai_background_pixels=0,
                masks='reviewed' if args.replacement_masks else 'SAM 2',
                audio='first source audio stream as PCM if present' if audio else 'omitted')

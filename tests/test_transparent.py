"""CPU-only regression checks: python -m unittest discover -s tests -v"""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

import inference
from inference import parse_args
from mimicmotion.utils.replacement import check_dependencies, segment
from mimicmotion.utils.transparent import check_encoder, encode_rgba, render_transparent


class Tensor:
    """Minimal CPU frame adapter; no model installation needed for encoder tests."""
    def __init__(self, data): self.data = data
    def permute(self, *axes): return Tensor(self.data.transpose(axes))
    def numpy(self): return self.data


class Frames:
    def __init__(self, data): self.data = data; self.shape = data.shape
    def __len__(self): return len(self.data)
    def __iter__(self): return (Tensor(frame) for frame in self.data)


class TransparentTests(unittest.TestCase):
    def args(self, *extra):
        return parse_args(['--ref_video_path', 'dance.mp4', '--ref_image_path', 'actor.png', *extra])

    def test_subprocess_cmd_args(self):
        from mimicmotion.utils.replacement import render
        frames = Frames(np.zeros((1, 3, 32, 48), np.uint8))
        g = dict(width=48, height=32, resized_width=48, resized_height=32, x=0, y=0)
        task = dict(video='test_video.mp4', media=dict(source_stream=dict(avg_frame_rate=30)))
        args = self.args('--mode', 'replace', '--mute')
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / 'prompts.json').write_text(json.dumps(dict(blockers=[[]])))
            mock_bg = MagicMock()
            mock_bg.fill.return_value = (np.zeros((32, 48, 3), np.uint8), np.zeros((32, 48), bool), [])
            with patch('mimicmotion.utils.replacement.segment', return_value=(temp, temp)), \
                 patch('mimicmotion.utils.replacement.mask_paths', return_value=[directory / '0.png']), \
                 patch('mimicmotion.utils.replacement.SourceBackground', return_value=mock_bg), \
                 patch('mimicmotion.utils.replacement.LamaFill'), \
                 patch('subprocess.Popen') as mock_popen:
                mock_proc = MagicMock()
                mock_proc.wait.return_value = 0
                mock_proc.stdin = MagicMock()
                mock_proc.stdin.closed = False
                mock_popen.return_value = mock_proc
                with patch('mimicmotion.utils.replacement.read_mask', return_value=np.ones((32, 48), np.uint8) * 255), \
                     patch('mimicmotion.utils.replacement.restore_frame', return_value=np.zeros((32, 48, 3), np.uint8)), \
                     patch.dict('sys.modules', {'decord': MagicMock()}):
                    import sys
                    mock_decord = sys.modules['decord']
                    mock_decord.VideoReader.return_value.__len__.return_value = 1
                    mock_decord.VideoReader.return_value.__getitem__.return_value.asnumpy.return_value = np.zeros((32, 48, 3), np.uint8)
                    render(frames, task, args, directory, g, directory / 'out.mp4')

                self.assertTrue(mock_popen.called)
                cmd = mock_popen.call_args[0][0]
                kwargs = mock_popen.call_args[1]
                self.assertFalse(kwargs.get('shell', False))
                self.assertTrue(all(isinstance(item, str) for item in cmd))
                self.assertIn('30', cmd)
                self.assertIn('test_video.mp4', cmd)

    def test_cli(self):
        self.assertFalse(self.args().transparent_background)
        args = self.args('--transparent_background', '--output_file', 'actor.mov')
        self.assertEqual(args.alpha_codec, 'prores4444')
        for extra in [('--output_file', 'actor.mp4'), ('--source_masks', 'masks'),
                      ('--occlusion_masks', 'masks'), ('--edge_feather', 'nan'),
                      ('--alpha_codec', 'h264')]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): self.args('--transparent_background', *extra)
        self.assertEqual(self.args('--mode', 'generate', '--transparent_background').fps, 15)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_alpha_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            rgba = np.full((32, 48, 4), 180, np.uint8)
            rgba[:, :16, 3] = 0
            rgba[:, 16:32, 3] = 128
            rgba[:, 32:, 3] = 255
            for codec in ('qtrle', 'prores4444'):
                with self.subTest(codec=codec):
                    check_encoder(codec)
                    output = Path(temp) / (codec + '.mov')
                    encode_rgba(iter([rgba] * 3), output, 48, 32, '30000/1001', codec, temp)
                    decoded = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output),
                        '-f', 'rawvideo', '-pix_fmt', 'rgba', 'pipe:1'])
                    actual = np.frombuffer(decoded, np.uint8).reshape(3, 32, 48, 4)
                    np.testing.assert_allclose(actual[0, ..., 3], rgba[..., 3], atol=1)
                    if codec == 'qtrle': np.testing.assert_array_equal(actual[0], rgba)
                    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
                        '-select_streams', 'v:0', '-show_entries', 'stream=avg_frame_rate,nb_frames,pix_fmt',
                        '-of', 'json', str(output)]))['streams'][0]
                    self.assertEqual(info['nb_frames'], '3')
                    self.assertEqual(info['avg_frame_rate'], '30000/1001')
                    self.assertIn(info['pix_fmt'], ('argb', 'yuva444p12le'))

    def test_reviewed_masks_bypass_background_and_sam(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            mask = np.zeros((32, 48), np.uint8); mask[8:24, 16:32] = 255
            Image.fromarray(mask).save(directory / '000000.png')
            args = self.args('--transparent_background', '--replacement_masks', temp,
                             '--alpha_codec', 'qtrle', '--edge_feather', '0', '--mute')
            args.device = 'cpu'
            frames = Frames(np.full((1, 3, 32, 48), 180, np.uint8))
            g = dict(width=48,height=32,resized_width=48,resized_height=32,x=0,y=0)
            task = dict(video='nonexistent.mp4', media=dict(source_frames=1,
                        source_stream=dict(avg_frame_rate='24/1')))
            with patch('mimicmotion.utils.transparent.check_encoder'), patch(
                    'mimicmotion.utils.replacement.subprocess.run', side_effect=AssertionError('SAM check')):
                check_dependencies(args)  # No LaMa file or SAM installation needed.
            with patch('mimicmotion.utils.replacement.segment', side_effect=AssertionError('SAM run')), patch(
                    'mimicmotion.utils.replacement.SourceBackground', side_effect=AssertionError('background')), patch(
                    'mimicmotion.utils.replacement.LamaFill', side_effect=AssertionError('LaMa')):
                result = render_transparent(frames,task,args,directory,g,directory/'actor.mov',None)
            self.assertEqual(result['ai_background_pixels'], 0)
            self.assertEqual(result['background_strategy'], 'none')
            args.mode = 'generate'; args.fps = 15
            result = render_transparent(frames,task,args,directory,None,directory/'generated.mov',None)
            self.assertEqual(result['fps'], 15)
            decoded = subprocess.check_output(['ffmpeg','-v','error','-i',str(directory/'actor.mov'),
                                              '-f','rawvideo','-pix_fmt','rgba','pipe:1'])
            actual = np.frombuffer(decoded,np.uint8).reshape(32,48,4)
            np.testing.assert_array_equal(actual[...,3],mask)
            np.testing.assert_array_equal(actual[12,20,:3],[180,180,180])

    def test_main_output_routing(self):
        for mode in ('replace', 'generate'):
            for transparent in (False, True):
                with self.subTest(mode=mode, transparent=transparent), tempfile.TemporaryDirectory() as temp:
                    directory = Path(temp)
                    checkpoint = directory/'weights'; checkpoint.write_bytes(b'test')
                    args = self.args('--mode', mode, '--device', 'cpu', '--ckpt_path', str(checkpoint),
                                     '--output_dir', temp, *(['--transparent_background'] if transparent else []))
                    output = directory / ('actor.mov' if transparent else 'actor.mp4')
                    task = dict(video='dance.mp4',image='actor.png',output=output,error=None,media={})
                    frames = Frames(np.zeros((1,3,32,48),np.uint8))
                    torch = SimpleNamespace(no_grad=contextlib.nullcontext,
                                            cuda=SimpleNamespace(OutOfMemoryError=MemoryError))
                    modules = {'torch':torch,
                        'mimicmotion.utils.geglu_patch':SimpleNamespace(patch_geglu_inplace=lambda:None),
                        'mimicmotion.utils.loader':SimpleNamespace(create_pipeline=MagicMock()),
                        'mimicmotion.dwpose.dwpose_detector':SimpleNamespace(DWposeDetector=MagicMock())}
                    def rendered(*values):
                        values[5].write_bytes(b'video')
                        return {'tested':True}
                    def saved(frames, output, args, record): output.write_bytes(b'video')
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(patch.dict(sys.modules,modules))
                        for name, value in [('plan_tasks',[task]),('probe_media',{}),
                            ('resolve_runtime',(SimpleNamespace(type='cpu'),'float32')),
                            ('environment_info',{}),('preprocess',(None,None)),('run_pipeline',frames)]:
                            stack.enter_context(patch.object(inference,name,return_value=value))
                        stack.enter_context(patch('mimicmotion.utils.replacement.check_dependencies'))
                        stack.enter_context(patch('mimicmotion.utils.replacement.prepare',return_value=(None,None,{})))
                        ordinary = stack.enter_context(patch('mimicmotion.utils.replacement.render',side_effect=rendered))
                        alpha = stack.enter_context(patch('mimicmotion.utils.transparent.render_transparent',side_effect=rendered))
                        generate = stack.enter_context(patch.object(inference,'save_result',side_effect=saved))
                        self.assertEqual(inference.main(args),0)
                        self.assertEqual(alpha.call_count,int(transparent))
                        self.assertEqual(ordinary.call_count,int(not transparent and mode=='replace'))
                        self.assertEqual(generate.call_count,int(not transparent and mode=='generate'))
                    self.assertTrue(output.is_file())
                    if transparent:
                        self.assertIn('transparency',json.loads(output.with_suffix('.json').read_text()))
                    self.assertFalse(list(directory.glob('.replacement-*')))
                    self.assertFalse(list(directory.glob('.transparent-*')))

    def test_generated_only_segmentation_job(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            args = self.args('--transparent_background'); args.device = 'cuda:0'
            # Only the dual branch exposes aux_device.
            expected_device = 'cuda:0'
            if hasattr(args,'aux_device'):
                args.aux_device = 'cuda:1'; expected_device = 'cuda:1'
            frames = Frames(np.zeros((1,3,32,48),np.uint8))
            g = dict(width=48,height=32,resized_width=48,resized_height=32,x=0,y=0)
            with patch('mimicmotion.utils.replacement.subprocess.run',
                       return_value=SimpleNamespace(returncode=0)) as run:
                segment(frames,args,directory,g,generated_only=True)
            payload = json.loads((directory/'segmentation.json').read_text())
            self.assertEqual(len(payload['jobs']),1)
            self.assertEqual(Path(payload['jobs'][0]['images']).name,'generated')
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[cmd.index('--device')+1],expected_device)


if __name__ == '__main__': unittest.main()

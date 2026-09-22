"""Run in a separate SAM 2 environment, without importing MimicMotion."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--jobs')
    parser.add_argument('--checkpoint')
    parser.add_argument('--model_config')
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor
    if args.check:
        return
    if not all((args.jobs,args.checkpoint,args.model_config)):
        parser.error('jobs, checkpoint and model_config are required')
    payload = json.loads(Path(args.jobs).read_text())
    boxes = json.loads(Path(payload['prompts']).read_text())['boxes']
    predictor = build_sam2_video_predictor(args.model_config,args.checkpoint,device=args.device)
    with torch.inference_mode():
        for job in payload['jobs']:
            output = Path(job['masks']); output.mkdir(parents=True,exist_ok=True)
            state = predictor.init_state(video_path=job['images'],offload_video_to_cpu=True,offload_state_to_cpu=True)
            try:
                if state['num_frames'] != len(boxes):
                    raise ValueError('SAM 2 input frame count mismatch')
                # Sparse box refreshes help reduce drift while temporal propagation links frames.
                for index in range(0,len(boxes),12):
                    predictor.add_new_points_or_box(state,frame_idx=index,obj_id=1,box=np.asarray(boxes[index],np.float32))
                seen = set()
                for index, ids, logits in predictor.propagate_in_video(state,start_frame_idx=0):
                    obj_index = list(ids).index(1)
                    mask = (logits[obj_index,0] > 0).cpu().numpy().astype('uint8')*255
                    if not mask.any():
                        raise ValueError(f'SAM 2 lost actor at frame {index}')
                    Image.fromarray(mask).save(output/f'{index:06d}.png')
                    seen.add(index)
                if seen != set(range(len(boxes))):
                    raise ValueError('SAM 2 did not return every frame')
            finally:
                predictor.reset_state(state)
                del state


if __name__ == '__main__':
    main()

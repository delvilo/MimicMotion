"""Lightweight argument definitions; no model or image imports."""
import math
import sys


def add_arguments(parser):
    parser.add_argument('--transparent_background', action='store_true',
                        help='Export only the generated actor with alpha to .mov; skip all background reconstruction')
    parser.add_argument('--alpha_codec', choices=('prores4444', 'qtrle'), default='prores4444',
                        help='Transparent MOV encoder: editing-friendly ProRes 4444 or lossless QTRLE')
    parser.add_argument('--mode', choices=('replace', 'generate'), default='replace',
                        help='replace keeps the source shot; generate is the previous full-frame mode')
    parser.add_argument('--sam2_python', default=sys.executable, help='Python executable in a separate SAM 2 environment')
    parser.add_argument('--sam2_checkpoint', default='models/sam2.1_hiera_small.pt')
    parser.add_argument('--sam2_model_config', default='configs/sam2.1/sam2.1_hiera_s.yaml')
    parser.add_argument('--lama_checkpoint', default='models/big-lama.pt', help='Simple-LaMa two-tensor TorchScript model')
    parser.add_argument('--no_ai_background', action='store_true', help='Fail if original frames cannot fill every visible hole')
    parser.add_argument('--source_masks', help='Optional reviewed source actor masks directory (single task)')
    parser.add_argument('--replacement_masks', help='Optional reviewed generated actor alpha directory (single task)')
    parser.add_argument('--occlusion_masks', help='Optional foreground occluder masks directory, white means keep source (single task)')
    parser.add_argument('--person_box', nargs=4, type=float, metavar=('X0','Y0','X1','Y1'),
                        help='Actor XYXY: source pixels in replace mode; generated output pixels in transparent generate mode')
    parser.add_argument('--background_candidates', type=int, default=0, help='0 searches all source frames; positive caps candidates per frame')
    parser.add_argument('--mask_padding', type=int, default=3, help='Expand source removal mask in original pixels')
    parser.add_argument('--edge_feather', type=float, default=1.0, help='Inward foreground feather radius')
    parser.add_argument('--keep_intermediates', action='store_true', help='Retain masks, camera-coordinate prompts and provenance maps')
    parser.add_argument('--mute', action='store_true', help='Omit original audio')


def validate_arguments(args, parser):
    if args.transparent_background and (args.source_masks or args.occlusion_masks):
        parser.error('Transparent output uses --replacement_masks only; source/occlusion masks are for background compositing')
    if args.mode != 'replace' and not args.transparent_background:
        return
    if args.mode == 'replace' and (args.sample_stride != 1 or args.fps is not None):
        parser.error('replace mode requires --sample_stride 1 and uses source FPS; omit --fps')
    if args.background_candidates < 0 or not 0 <= args.mask_padding <= 64:
        parser.error('background_candidates must be >= 0; mask_padding must be between 0 and 64')
    if not math.isfinite(args.edge_feather) or not 0 <= args.edge_feather <= 20:
        parser.error('edge_feather must be finite and between 0 and 20')
    if args.person_box and (not all(math.isfinite(v) for v in args.person_box) or
                           args.person_box[2] <= args.person_box[0] or args.person_box[3] <= args.person_box[1]):
        parser.error('person_box must be finite XYXY coordinates with positive width and height')
    if len(args.ref_video_path) > 1 and any((args.source_masks, args.replacement_masks, args.occlusion_masks, args.person_box)):
        parser.error('Masks and person_box are single-task options; run those shots individually')

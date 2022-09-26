import os
from argparse import Namespace

def setup_multi_gpu_slurm(args: Namespace):
    is_slurm = os.getenv('SLURM_JOB_ID') is not None
    if is_slurm:
        args.world_size = int(os.getenv('SLURM_NTASKS'))
        args.local_rank = int(os.getenv('SLURM_LOCALID'))
        args.global_rank = int(os.getenv('SLURM_PROCID'))
        args.device = f'cuda:{args.local_rank}'
        print(f'SLURM job: global rank {args.global_rank}, GPU device {args.device}')
    else:
        args.world_size = 1
        args.local_rank = args.global_rank = 0
        if not hasattr(args, 'device') or not args.device:  # set device if not specified
            args.device = f'cuda:{args.local_rank}'
        print(f"Local job: global rank {args.global_rank}, GPU device {args.device}")
    args.is_multi = args.world_size > 1

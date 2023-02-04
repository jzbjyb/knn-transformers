from typing import List
import os
from argparse import Namespace
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(20)

def setup_multi_gpu_slurm(args: Namespace):
    is_slurm = os.getenv('SLURM_JOB_ID') is not None
    if is_slurm:
        args.world_size = int(os.getenv('SLURM_NTASKS'))
        args.local_rank = int(os.getenv('SLURM_LOCALID'))
        args.global_rank = int(os.getenv('SLURM_PROCID'))
        args.device = torch.device(f'cuda:{args.local_rank}')
        logger.info(f'SLURM job: global rank {args.global_rank}, GPU device {args.device}')
    else:
        args.world_size = 1
        args.local_rank = args.global_rank = 0
        if not hasattr(args, 'device') or not args.device:  # set device if not specified
            args.device = torch.device(f'cuda:{args.local_rank}')
        logger.info(f"Local job: global rank {args.global_rank}, GPU device {args.device}")
    args.is_multi = args.world_size > 1

class StridedTensorCore:
    def __init__(self, packed_tensor, lengths, dim=None):
        self.dim = dim
        self.tensor = packed_tensor
        self.inner_dims = self.tensor.size()[1:]

        self.lengths = lengths.long() if torch.is_tensor(lengths) else torch.LongTensor(lengths)
        self.device = self.lengths.device  # put length on cuda is affordable

        self.strides = _select_strides(self.lengths, [.5, .75, .9, .95]) + [self.lengths.max().item()]
        self.max_stride = self.strides[-1]

        zero = torch.zeros(1, dtype=torch.long, device=self.device)
        self.offsets = torch.cat((zero, torch.cumsum(self.lengths, dim=0)))

        if self.offsets[-2] + self.max_stride > self.tensor.size(0):
            padding = torch.zeros(self.max_stride, *self.inner_dims, dtype=self.tensor.dtype, device=self.tensor.device)
            self.tensor = torch.cat((self.tensor, padding))

        self.views = {stride: _create_view(self.tensor, stride, self.inner_dims) for stride in self.strides}

def _select_strides(lengths, quantiles):
    if lengths.size(0) < 5_000:
        return _get_quantiles(lengths, quantiles)

    sample = torch.randint(0, lengths.size(0), size=(2_000,))

    return _get_quantiles(lengths[sample], quantiles)

def _get_quantiles(lengths, quantiles):
    return torch.quantile(lengths.float(), torch.tensor(quantiles, device=lengths.device)).int().tolist()

def _create_view(tensor, stride, inner_dims):
    outdim = tensor.size(0) - stride + 1
    size = (outdim, stride, *inner_dims)

    inner_dim_prod = int(np.prod(inner_dims))

    # e.g., inner_dims [2, 3, 4] -> inner_stride [12, 4, 1]
    inner_stride = ([1] + np.cumprod(inner_dims[::-1]).tolist()[:-1])[::-1] if len(inner_dims) else []

    multidim_stride = [inner_dim_prod, inner_dim_prod] + inner_stride

    return torch.as_strided(tensor, size=size, stride=multidim_stride)

def _create_mask(lengths, stride, like=None):
    mask = torch.arange(stride).to(lengths.device) + 1
    mask = mask.unsqueeze(0) <= lengths.unsqueeze(-1)

    if like is not None:
        for _ in range(like.dim() - mask.dim()):
            mask = mask.unsqueeze(-1)

    return mask

class StridedTensor(StridedTensorCore):
    def __init__(self, packed_tensor, lengths, dim=None):
        super().__init__(packed_tensor, lengths, dim=dim)

    def _prepare_lookup(self, pids):
        assert pids.dim() == 1
        pids = pids.long()
        lengths = self.lengths[pids]
        offsets = self.offsets[pids]
        return pids, lengths, offsets

    def lookup(self, pids, output='packed'):
        ori_device = pids.device
        pids = pids.to(self.device)

        pids, lengths, offsets = self._prepare_lookup(pids)

        maxlen = lengths.max().item()
        stride = next(s for s in self.strides if maxlen <= s)

        tensor = self.views[stride][offsets].to(self.device)
        mask = _create_mask(lengths, stride)

        if output == 'padded':
            tensor = tensor[:, :maxlen].contiguous()
            mask = mask[:, :maxlen].contiguous()
            return tensor.to(ori_device), mask.to(ori_device)

        assert output == 'packed'
        tensor = tensor[mask]
        return tensor.to(ori_device), lengths.to(ori_device)


def yesno_metric(
    predictions: List[str],
    references: List[str],
    anchor_text: str = 'the final answer is',
):
    metrics = {
        'correct': 0,
        'incorrect': 0,
        'formaterror': 0,
    }

    assert len(predictions) == len(references)
    for pred, ref in zip(predictions, references):
        ref = ref.lower().strip()
        pred = pred.strip().lower()
        if anchor_text in ref:
            ref = ref[ref.find(anchor_text) + len(anchor_text):].strip()[:-1]
            assert ref in {'yes', 'no'}
        if anchor_text not in pred:
            metrics['formaterror'] += 1
        else:
            if ref in pred[pred.find(anchor_text) + len(anchor_text):].strip():
                metrics['correct'] += 1
            else:
                metrics['incorrect'] += 1
    metrics = {k: v / len(predictions) for k, v in metrics.items()}
    return metrics


def remove_prefix(text: str, prefix: str):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


strategy_qa_examplars = [
    {
        'question': 'Do hamsters provide food for any animals?',
        'answer': 'Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. Therefore, the final answer is yes.',
    },
    {
        'question': 'Could Brooke Shields succeed at University of Pennsylvania?',
        'answer': 'Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. Therefore, the final answer is yes.',
    },
    {
        'question': "Hydrogen's atomic number squared exceeds number of Spice Girls?",
        'answer': "Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5. Therefore, the final answer is no.",
    },
    {
        'question': "Is it common to see frost during some college commencements?",
        'answer': "College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. Therefore, the final answer is yes.",
    },
    {
        'question': "Could a llama birth twice during War in Vietnam (1945-46)?",
        'answer': "The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. Therefore, the final answer is no.",
    },
    {
        'question': "Would a pear sink in water?",
        'answer': "The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float. Therefore, the final answer is no.",
    }
]

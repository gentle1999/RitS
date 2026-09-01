"""Minimal two-GPU collective probe for diagnosing the local DDP backend."""

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    value = torch.tensor([float(dist.get_rank() + 1)], device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    torch.cuda.synchronize()
    print(f"rank={dist.get_rank()} value={value.item()}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

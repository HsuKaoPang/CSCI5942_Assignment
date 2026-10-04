"""Data parallel training.

Every rank holds a full copy of the model, sees a different slice of
every batch, and after each backward pass the gradients are made
identical. Three things for you to implement in this file.

    broadcast_module_state    to synchronize initial state
    naive_all_reduce_grads    a slow version of gradient averaging
    OverlappedDDP             a pipelined version of gradient averaging


To run the correctness suite

    python -m pytest tests/test_ddp.py -q
"""

from __future__ import annotations

import contextlib

import torch
import torch.distributed as dist
import torch.nn as nn

# Some helper functions
def group_world_size(group=None) -> int:
    """Ranks in `group`, or 1 when this process is not distributed."""
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    return dist.get_world_size(group=group)


def group_src_rank(group=None) -> int:
    """The *global* rank of the first member of `group`.
    This function returns the global rank of this group,
    regardless of which rank is calling it.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    if group is None:
        return 0
    return dist.get_process_group_ranks(group)[0]


# ---------------------------------------------------------------------
# 1. identical replicas
# ---------------------------------------------------------------------
def broadcast_module_state(module: nn.Module, group=None) -> None:
    """Make every rank's copy of `module` bit-identical to the first rank's.

    Here, we choose the `group_src_rank` of the group as the source of truth, 
    and broadcast every parameter to other ranks in the group.

    Hint: use `dist.broadcast(tensor.data, src=..., group=...)`. Synchronize
    `module.parameters()` and `module.buffers()` across
    ranks of the group.
    """
    src = group_src_rank(group)
    for t in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(t.data, src=src, group=group)


# ---------------------------------------------------------------------
# 2. the naive version
# ---------------------------------------------------------------------
def naive_all_reduce_grads(module: nn.Module, group=None) -> None:
    """Average every gradient across the data parallel group, in place.
    This is automatically called after `loss.backward()` and 
    before `opt.step()`.

    Two things to note.

    **Sum versus average.** `dist.all_reduce` defaults to `ReduceOp.SUM`.
    `ReduceOp.AVG` exists, but only on NCCL. You will have to 
    divide by `group_world_size(group)` yourself to support gloo.

    **Parameters with no gradient.** `p.grad` is None for anything that
    did not participate. Skip these.
    """
    world = group_world_size(group)
    for p in module.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, group=group)
        p.grad /= world


# ---------------------------------------------------------------------
# 3. the overlapped version
# ---------------------------------------------------------------------
class OverlappedDDP(nn.Module):
    """Data parallel that reduces each gradient as soon as it exists.

    Gradients become available as the backward pass propagates, so
    the reduction for the last layer can be in flight while the first
    layer is still computing.

    What to write:

    * In `__init__`, register a **post-accumulate-grad hook** on every
      parameter that requires grad, which fires immediately after `p.grad` 
      is updated.
      Hint: refer to `register_post_accumulate_grad_hook` in the PyTorch docs.

    * In the hook, launch an asynchronous all-reduce and keep the handle
        in `self._handles`. 

    * In `finish_gradient_synchronization`, wait on every handle and
      clear the list. The training loop calls this after the backward
      pass and before `opt.step()`.

    Keep `self.module` as the attribute holding the wrapped model.
    """

    def __init__(self, module: nn.Module, group=None,
                 broadcast_state: bool = True):
        super().__init__()
        self.module = module
        self.group = group
        self.world_size = group_world_size(group)
        self.require_backward_grad_sync = True
        self._handles: list = []
        if broadcast_state and self.world_size > 1:
            broadcast_module_state(module, group)
        if self.world_size > 1:
            self._register_hooks()

    def _register_hooks(self) -> None:
        for p in self.module.parameters():
            if p.requires_grad:  #if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._reduce) #when p.grad is updated, call self._reduce(p)

    def forward(self, *args, **kwargs):
        if self._handles:
            # A new forward pass with reductions still in flight means
            # the next backward will accumulate into a `p.grad` that an
            # all-reduce is concurrently reading. 
            raise RuntimeError(
                f"{len(self._handles)} gradient all-reduces are still in "
                f"flight at the start of a forward pass.")
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for handle in self._handles:
            handle.wait()
        self._handles.clear()
    
        
    def _reduce(self, p: torch.Tensor) -> None:
        if not self.require_backward_grad_sync:
            return

        # Perform an asynchronous all-reduce on `p.grad` over the
        # group -- `p` itself is the parameter, not the gradient.
        # hint: use the `async_op` argument
        # Divide before launching: once the async all-reduce is in
        # flight, `p.grad` must not be touched until it is waited on.
        p.grad /= self.world_size
        # async_op=True means that the all-reduce is launched asynchronously, 
        # and returns a handle that can be waited on later
        handle = dist.all_reduce(p.grad, group=self.group, async_op=True) 
        self._handles.append(handle)

    @contextlib.contextmanager
    def no_sync(self):
        """Accumulate gradients locally, without communicating.
        This function ensures that only the last micro-batch of a step will trigger gradient
        synchronization. We wrap every micro-batch of a step except the last:

            for i in range(accum):
                ctx = model.no_sync() if i < accum - 1 else nullcontext()
                with ctx:
                    (model(x, y)[1] / accum).backward()
            model.finish_gradient_synchronization()
        """
        old = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = old
        return

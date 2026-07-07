# NPU Event Timer

`putils.timer` provides two timer families:

- `timer()` / `timer_decorator()` for CPU wall-clock timing.
- `event_timer` / `CUDAEVENT_TIMER` for NPU event timing on compute streams and communication streams.

The NPU event timer writes one CSV file per rank:

```text
${LOG_DIR:-./logs}/event_timer_rank${rank}.csv
```

By default `event_timer.disable=True`, so users must enable it explicitly.

## Basic setup

```python
import os
from putils.timer import event_timer

os.environ["LOG_DIR"] = "./event_logs"
event_timer.disable = False
event_timer.csv_output = True
```

Call `flush()` after the measured region. `flush()` synchronizes the NPU, calculates elapsed time for paired events, appends one row to the rank-local CSV file, and clears the current buffers.

```python
event_timer.flush()
```

## Patch FSDP all_gather

`patch_fsdp_all_gather()` wraps PyTorch FSDP's private `foreach_all_gather` function and inserts an NPU start/end event around each call.

Enable the patch before the training loop:

```python
from putils.timer import event_timer

event_timer.disable = False
event_timer.patch_fsdp_all_gather("fsdp_all_gather", patch=True)
```

Run training as usual, then flush periodically:

```python
for step, batch in enumerate(dataloader):
    loss = train_step(batch)

    if step % 10 == 0:
        event_timer.flush()
```

Restore the original FSDP function when the measurement is finished:

```python
event_timer.patch_fsdp_all_gather("fsdp_all_gather", patch=False)
```

If you want to record events on the communication stream of a specific process group, pass `group`:

```python
event_timer.patch_fsdp_all_gather(
    "fsdp_all_gather_tp",
    patch=True,
    group=tp_group,
)
```

The CSV columns for this label include:

```text
fsdp_all_gather_max_ms
fsdp_all_gather_min_ms
fsdp_all_gather_avg_ms
fsdp_all_gather_times_ms
```

## Use add_event on the compute stream

`add_event(label)` records an event on the current NPU compute stream. Events with the same label are interpreted as start/end pairs.

```python
from putils.timer import event_timer

event_timer.disable = False

event_timer.add_event("mlp")
hidden_states = mlp(hidden_states)
event_timer.add_event("mlp")

event_timer.add_event("attention")
hidden_states = attention(hidden_states)
event_timer.add_event("attention")

event_timer.flush()
```

You can add multiple pairs with the same label before one `flush()`. The CSV row will report max/min/avg and the raw list of pair times.

Tensor shapes can be written into the same CSV row:

```python
event_timer.add_tensor_shape("hidden_states", hidden_states)
```

CPU-side timing can also be recorded:

```python
event_timer.add_cpu()
do_python_side_work()
event_timer.add_cpu()
event_timer.flush()
```

## Use add_event on a communication stream

`add_event(label, group=...)` can record an event on the communication stream associated with a distributed process group:

```python
event_timer.add_event("tp_all_reduce", group=tp_group)
torch.distributed.all_reduce(tensor, group=tp_group, async_op=False)
event_timer.add_event("tp_all_reduce", group=tp_group)
```

This is useful when the communication stream is independent from the compute stream. However, stream timing is subtle: the exact position of communication events may include extra `wait_event` or synchronization inserted by PyTorch, FSDP, the backend, or surrounding compute kernels. In that case, the measured interval may not show the pure communication duration.

When communication timing looks suspicious, record paired events on both streams:

```python
event_timer.add_event("all_reduce_compute_side")
event_timer.add_event("all_reduce_comm_side", group=tp_group)

work = torch.distributed.all_reduce(tensor, group=tp_group, async_op=True)
work.wait()

event_timer.add_event("all_reduce_comm_side", group=tp_group)
event_timer.add_event("all_reduce_compute_side")
event_timer.flush()
```

Compare the compute-side and communication-side labels in the CSV. This usually makes it easier to tell whether the measured interval is dominated by actual communication, compute-stream waiting, or extra stream synchronization.

## Common notes

- Keep event labels paired. If a label has an odd number of events, `flush()` writes `${label}_status=unpaired`.
- `flush()` calls `torch.npu.synchronize()`, so avoid calling it every step unless the synchronization cost is acceptable.
- `patch_fsdp_all_gather()` touches PyTorch private FSDP symbols. Re-check the patch when upgrading PyTorch.
- `LOG_DIR` controls the CSV output directory.


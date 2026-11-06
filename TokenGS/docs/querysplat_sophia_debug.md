# QuerySplat DDP stall investigation

The available code and log do not establish the cause of the reported step-10 stall.
There is no step-11 branch in the loop. Every rank uses the same checkpoint condition
and DistributedSampler supplies equal epoch lengths. The saved message is printed
before the post-write barrier, so it does not prove that all ranks left checkpointing.
Also, with the default `--log-every 10`, step 11 is not printed; the next regular
message is step 20. Metrics are appended on every completed step.

Concrete issues addressed:

* Provider previously caught every exception and retried forever without reporting
  the failed scene. It now reports rank, scene index, attempt and exception and
  raises after 32 attempts. A filesystem read that never returns can still block;
  bounded retries cannot interrupt an in-flight read.
* Resume previously fetched and fully preprocessed every skipped batch. New runs
  use provider seed/epoch/scene-index sampling and skip sampler indices before
  dataset access. This preserves dynamic sampling across epochs and makes view
  selection and reflection independent of worker scheduling. It changes the random
  sample sequence for new runs, not the architecture or losses. Old checkpoints
  retain sequential sampling and batch replay for compatibility; their exact
  multiworker sampling replay is not guaranteed because worker states were never
  saved. Use the same worker count when debugging an old checkpoint.
* Saving previously queried CUDA RNG on every visible GPU, and loading restored
  rank 0's all-device RNG list on every process. Saving now accesses only the local
  device. DDP loading defers RNG restoration until after construction and restores
  the authoritative per-rank CPU/CUDA state already stored in the checkpoint.
* The temporary checkpoint is flushed and fsynced before atomic replacement of
  latest.pt. An interrupted/failed write leaves the previous latest.pt available.

## Sophia diagnostic run

Keep the existing launch arguments and append:

```bash
--diagnostics --diagnostic-cuda-sync --stack-dump-seconds 120 --log-every 1
```

Use `PYTHONUNBUFFERED=1` in the PBS environment. Start with workers=0 and the same
20-step contract used to create the old checkpoint. The distributed timeout can be
set with `--distributed-timeout-seconds` (default 1800). It limits collective waits,
not dataset I/O. Stack dumps are repeated Python traces from each training process;
with worker processes enabled, they do not dump worker stacks. CUDA synchronization
adds overhead; disable it after identifying the slow phase.

Each BEGIN/END includes rank and elapsed wall time; BEGIN also identifies the
completed global step. At global step 10, the forward phase is computing step 11.
Compare all four ranks, not only rank 0:

* `checkpoint.post_write_barrier` BEGIN without END: checkpoint synchronization.
* `loader.next` BEGIN without END: data loading or preprocessing. Look for preceding
  scene warnings and stack traces. Other ranks may wait in forward/backward.
* `forward`, `loss`, or `backward` BEGIN without END: inspect that rank's stack and
  GPU activity. With CUDA synchronization enabled, END includes GPU completion.
* `metrics_and_images`: rank 0 writes metrics and optional image grids; other ranks
  may progress to the next collective while it does so.
* `resume.load`: checkpoint deserialization/model/optimizer/EMA restoration.
* `startup.ddp` or `startup.lpips`: initialization before training resumes.

Do not infer a DDP deadlock just because three ranks are waiting in a collective:
one slow rank doing data I/O or GPU work produces that symptom as well.

## Long jobs and verification

For a new 300000-step run, set `--total-steps 300000` from the beginning and keep the
same value on resume: total_steps is part of both the resume contract and LR schedule.
A 20-step checkpoint cannot currently be resumed with total_steps=300000. Keep the
same seed, data root/scene inventory, config, VGM weights, world size and training
hyperparameters. The new sampling version is recorded in checkpoint metadata.
Resume reconstitutes epoch and rank-local offset from global_step and the full
sampler length, including at epoch boundaries.

Save periodically at an interval comfortably shorter than PBS walltime. After a
hard kill, resume from latest.pt; computation since the last completed checkpoint
is repeated. There is no new PBS signal handler or automatic resubmission. Metrics
and images can contain entries from work after the last durable checkpoint, so
metrics consumers should account for repeated steps after restart. Choose the save
interval after measuring checkpoint time and step time; do not assume a 30-minute
allocation can complete a particular number of steps.

Before production, run an uninterrupted 20-step job and an interrupted/resumed job
with checkpoints at step 10, then repeat across an epoch boundary. Confirm all ranks
reach step 20, scheduler/EMA restore, and a new sampling-version checkpoint reports
`fast_resume=True`. Run the CPU regression tests in the training environment:

```bash
python -m unittest discover -s tests -p test_training_state.py
python -m unittest discover -s tests -p test_multiscene_training.py
```

Local validation here: edited Python files compile. Runtime tests could not import
PyTorch in the available Python environment; no Sophia GPU/DDP validation was run.

# Running the experiments on a Cloud TPU VM

Every JAXBench workload runs on a single device, so one v5e chip (`v5litepod-1`) is
enough. Renting 8 chips costs 8x for no benefit here. Billing runs from creation to
deletion, so delete the VM as soon as you are done (step 8).

## Access options

| Option | Cost | Notes |
|---|---|---|
| **Google Cloud TPU VM** (these instructions) | Pay per chip-hour (see the Cloud TPU pricing page); Spot VMs are cheaper but can be preempted | Full control. New projects usually start with zero TPU quota, so the quota request (step 2) is the slow part. |
| **TPU Research Cloud (TRC)** | Free | Google grants free TPU time to researchers; apply via the form on sites.research.google/trc/about. No approval timeline is published, so apply in parallel. |
| **Google Colab TPU runtime** | Free / Colab Pro | Browser notebook, single chip, availability varies. Fine for a quick check, awkward for a full sweep. |

TPUs are only sold by Google, so there is no third-party rental market for them.

## One-time setup (on your laptop)

1. Create a Google Cloud account and a project with billing enabled. Set a budget
   alert (Billing -> Budgets & alerts) so a forgotten VM cannot run up a large bill.
2. Install and authenticate the CLI:
   ```bash
   brew install --cask google-cloud-sdk
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   gcloud services enable tpu.googleapis.com
   ```
3. Request TPU quota: Console -> IAM & Admin -> Quotas, filter for "TPU v5e" (the
   entries are split into on-demand, Spot and reserved), pick a zone that offers v5e
   (`us-central1-a`, `us-west4-a`, `us-west1-c`, `us-south1-a`, `europe-west4-b`), and
   request 4 chips. Approval can take hours to days. Ask for quota in the same zone
   you plan to create the VM in.

## Create the VM

```bash
export ZONE=us-central1-a
export TPU=hawkeye
gcloud compute tpus tpu-vm create $TPU \
  --zone=$ZONE \
  --accelerator-type=v5litepod-1 \
  --version=v2-alpha-tpuv5-lite
```

- Add `--spot` for a cheaper VM that can be preempted (the eval runner resumes, see below).
- If creation fails for lack of capacity, try another zone from the list above, or use
  `v5litepod-4` (more likely to be available than nothing, at 4x the price).
- If direct creation is rejected, Google's v5e guide creates VMs through queued
  resources instead:
  ```bash
  gcloud compute tpus queued-resources create hawkeye-qr \
    --node-id=$TPU --zone=$ZONE \
    --accelerator-type=v5litepod-1 --runtime-version=v2-alpha-tpuv5-lite
  gcloud compute tpus queued-resources list --zone=$ZONE   # wait for ACTIVE
  ```

## Connect and install

```bash
gcloud compute tpus tpu-vm ssh $TPU --zone=$ZONE
```

Then, on the VM:

```bash
bash <(curl -s https://raw.githubusercontent.com/m8ngotree/tpu-hawkeye/main/scripts/tpu_vm_setup.sh)
cd ~/tpu-hawkeye && source .venv-tpu/bin/activate
```

The last line of the setup output must list a TPU device (for example
`[TpuDevice(id=0, ...)]`). If it shows a CPU device, stop and fix that first.

Run everything below inside `tmux` so an SSH drop does not kill it
(`tmux new -s run`; reattach with `tmux attach -t run`; install with
`sudo apt-get install -y tmux` if missing).

## Step 1: verify the taxonomy cells on hardware

```bash
python scripts/verify_cells.py
```

This runs every cell's naive and optimized kernel on the chip and prints status,
median time, speedup and whether the outputs match. Read it as follows:

- `ERROR` on a variant: the kernel failed to compile or run on real Mosaic. Note the
  message; the cell needs changing (for example a naive variant that violates a
  hardware constraint may be rejected outright instead of running slowly).
- A speedup near 1.00x where the cell claims a gap: the cell does not demonstrate its
  technique at this problem size.
- `match False`: the two variants disagree numerically; investigate before anything else.

Results are saved to `results/cell_verification_summary_hardware.json`.

## Step 2: smoke-test the agent

```bash
export LLM_API_KEY=your-deepseek-key      # typed in the shell only; never commit it
python -m eval.run_agent_eval --workloads 12p_RMSNorm --conditions taxonomy --max-turns 10 --tag smoke
```

One real agent run of at most 10 turns (cents of API cost). Check that it completes
without `harness_error`, and read
`results/runs/smoke/taxonomy/12p_RMSNorm_r0/trajectory.jsonl` to confirm the agent read
files and ran `eval.py`. `result.json` in the same directory holds the independently
re-scored outcome.

## Step 3: pilot

```bash
python -m eval.run_agent_eval \
  --workloads 12p_RMSNorm,8p_GEMM,41k_Gemm_Add_ReLU \
  --conditions taxonomy none --reps 1 --max-turns 30 --tag pilot
```

Record API cost, turns used, and how often the final kernel is correct in each
condition. Runs that already have a `result.json` are skipped, so re-running the same
command after a crash or preemption continues where it stopped (`--force` redoes them).

## Step 4: full sweep

Same command with all workloads (`python -m JAXBench list` prints the names),
`--reps 3`, and identical settings in both conditions.

## Copy results back

From your laptop:

```bash
gcloud compute tpus tpu-vm scp --zone=$ZONE --recurse $TPU:~/tpu-hawkeye/results ./results-from-tpu
```

## Delete the VM

```bash
gcloud compute tpus tpu-vm delete $TPU --zone=$ZONE --quiet
gcloud compute tpus tpu-vm list --zone=$ZONE        # must show no TPUs
```

If you used queued resources, also delete the queued resource
(`gcloud compute tpus queued-resources delete hawkeye-qr --zone=$ZONE --quiet`).

# Agentic Shopping Kit Example

This example trains a policy on a multi-turn tool-calling task. Each sample asks
the model to build a small shopping kit with one item from each required
category, while respecting feature constraints and a total budget. The agent runs
four model turns:

1. `search_catalog`
2. `inspect_items`
3. `check_kit`
4. `submit_bundle`

The reward function scores the final submitted bundle and gives full credit only
when the expected tool sequence is used and the bundle satisfies every category,
feature, and budget constraint.

## Generate Tasks

```bash
python examples/agentic/shopping/dataset_generator.py \
  --output /tmp/areno-shopping.jsonl \
  --count 2048 \
  --seed 2026
```

## Train

```bash
areno train \
  --ckpt Qwen/Qwen3-1.7B \
  --dataset-path /tmp/areno-shopping.jsonl \
  --dataset-loader-fn examples/agentic/shopping/dataset_loader.py \
  --reward-fn-path examples/agentic/shopping/reward.py \
  --agent-fn examples/agentic/shopping/run_agent.py \
  --algo gspo \
  --batch-size 8 \
  --n-samples 4 \
  --max-new-tokens 128
```

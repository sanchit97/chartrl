# ChartRL: Improving Chart understanding in VLMs 

Repo for RL post-training on Chart data. We are happy to present a small 3 billion chart understanding model which outputs answers AND rationales with no fine-tuning.

The design hinges on training VLMs using GRPO with verifiable rewards.
The training data is sampled from train sets of - ChartQA, PlotQA, FigQA and ChartFC.


Huge thanks to the team at Morgan Stanley for their support.



## Main Benchmark Results

| Model (≈ 2‑3 B params)     | Strategy        | ChartQA&nbsp;(RA) | Aug&nbsp;(RA) | Human&nbsp;(RA) | Notes                     |
|----------------------------|-----------------|------------------|---------------|-----------------|---------------------------|
| Qwen‑2.5 VL‑Instruct       | Direct          | 82.84            | 94.48         | 71.20           | No rationales             |
| Qwen‑2.5 VL‑Instruct – SFT | Direct          | —                | —             | —               | No rationales             |
| Qwen‑2.5 VL‑Instruct       | CoT             | 73.12            | 93.44         | 52.80           | Very bad rationales       |
| Qwen‑2.5 VL‑Instruct – SFT | CoT             | —                | —             | —               | No rationales             |
| Qwen‑2.5 VL‑Instruct       | GRPO (naïve)    | 77.00            | 89.44         | 64.56           | Good rationales           |
| **Qwen‑2.5 VL‑Instruct**   | **Ours**        | **86.80**        | **94.40**     | **79.20**       | **Solid rationales**      |



Our approach boosts performance on human annotated splits of ChartQA 


## Loss ablations for DPO (old work not relevant now)

| Model          | Loss       | RA    |
|----------------|------------|-------|
| Qwen-2VL-Base  | –          | 71.48 |
|                | Sigmoid    | 71.92 |
|                | IPO        | 61.28 |
|                | Sppo_hard  | 71.88 |
|                | Hinge      | **72.04** |
|                | Apo Zero   | **72.04** |


![image info](./ablation-dpo-loss.png "Ablation DPO Loss")




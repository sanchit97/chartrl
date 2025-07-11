# ChartRL: Improving Chart understanding in VLMs 

Repo for RL post-training on Chart data

## Main Inference Results

| Model                     | Strategy  | ChartQA (RA) | PlotQA (RA)  | FigQA (RA)
|---------------------------|-----------|-------------:|-------------:|------------:
| Qwen-2VL-Instruct         | Base      | 75.00        | 83.36        |
| Qwen-2VL-Instruct         | DPO-Direct| 75.48 (+0.48)| 83.50 (+0.14)|

## Loss ablations for DPO

| Model          | Loss       | RA    |
|----------------|------------|-------|
| Qwen-2VL-Base  | –          | 71.48 |
|                | Sigmoid    | 71.92 |
|                | IPO        | 61.28 |
|                | Sppo_hard  | 71.88 |
|                | Hinge      | **72.04** |
|                | Apo Zero   | **72.04** |


![image info](./ablation-dpo-loss.png "Ablation DPO Loss")




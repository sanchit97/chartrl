# ChartRL: Improving Chart understanding in VLMs 

Repo for RL post-training on Chart data

## Main Inference Results

| Model                     | Training  | ChartQA (RA) | PlotQA (RA) |
|---------------------------|-----------|-------------:|------------:|
| Llava-Next (7 B)          | Base      | 51.12        | 39.38       |
| Llava-Next (7 B)          | SFT       | 66.12        |             |
| Llava-Next (7 B)          | DPO - factuality| 66.36  |             |
| Qwen-2VL-Instruct         | Base (COT)     | 69.96        | 61.12       |
| Qwen-2VL-Instruct         | SFT  (COT)     | 69.44*        |             |
| Qwen-2VL-Instruct         | DPO - factuality (COT)| 70.16  |             |
| Qwen-2VL-Instruct         | DPO - all | 72.24  |             |

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




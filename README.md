# ChartRL: Improving Chart understanding in VLMs 

Repo for RL post-training on Chart data. We present a small 3 billion chart understanding model which outputs answers AND rationales with no fine-tuning or distilling reasoning traces from larger frontier LLMs. Currently the models are based on Qwen2.5VL (3 billion).

The design hinges on training VLMs using GRPO with verifiable rewards.
The training data is sampled from train sets of - ChartQA, PlotQA, FigQA and ChartFC.


Huge thanks to the team at Morgan Stanley for their support.



## Main Benchmark Results

| Approach                           | Exp? | ChartQA | PlotQA | ChartFC | EvoChart | ChartQAPro | 
|------------------------------------|------|---------|--------|---------|----------|------------|
| **Direct Prompting**               |      |         |        |         |          |            |
| Q2.5VL-Ins                         | ✗    | 82.0    | 80.5   | 74.4    | 48.72    | 25.7       |
| **Explainable Models - with Rationales** |      |         |        |         |          |      |            
| Q2.5VL-Ins (CoT)                   | ✔    | 73.12   | 52.72  | 69.20   | 29.6     | 15.80      | 
| ChartGemma                         | ✔    | 76.44   | 33.28  | 70.33   | 36.96    | 10.93      | 
| **Fine-tuned Models - with Rationales** |      |         |        |         |          |       | 
| Q2.5VL-SFT                         | ✔    | 83.08   | 74.18  | 77.30   | 46.08    | 23.56      | 
| Q2.5VL-DPO                         | ✔    | 75.42   | 53.86  | 70.1    | 34.8     | 15.95      | 
| Q2.5VL-Ins (F+L+Tasks)             | ✔    | 81.8    | 76.24  | 63.85   | 51.68    | 27.66      | 
| **Chart-RVR-3B (Ours)**            | ✔    | 84.56   | **78.68** | 77.62   | 53.36    | 28.38   |  
| **Data Curated Fine-tuned Models - with Rationales** |      |         |        |         |          |            |            |
| Q2.5VL-SFT-Hard                    | ✔    | 84.28   | 75.54  | 77.90   | 49.36    | 23.20      | 
| **Chart-RVR-3B-Hard (Ours)**       | ✔    | **85.76** | 77.9   | **80.07** | **54.24** | **28.64** | 






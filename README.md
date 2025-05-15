# KETCHUP: KETCHUP: K-Step Return Estimation for Sequential Knowledge Distillation

## Models

Check out our GRM series below, which are evlauated on [reward-bench](https://huggingface.co/spaces/allenai/reward-bench).

Performance on Flan-T5(2024, GOOGLE) series models


|       Approaches               | Xsum(ROUGE1)       |  MT EN-NL(BLEU4)     |     GSM8K(ACCURACY)      |   
|:-------------------------:|:-------------:|:---------:|:---------:|
|Prompt Teacher**(Flan-T5-3B)**| 41.32 | 25.36 | 40.71 |
|Prompt Student**(T5-0.4B)**|1 9.60 | 0.95 | 0.00 |
|SeqKD| 33.54 | 22.09 | 20.02 |
|KL| 34.36 | 22.35 | 23.96 |
|JS| 34.87 |22.55|24.72|
|TVD|35.17|22.63|24.94|
|LLMR|35.54|22.72|25.21|
|LLMR + Mean baseline|35.60|22.67|25.39|
|LLMR + Min-Var baseline|35.59|22.70|25.10|
|KETCHUP(best K)|36.03|22.95|25.71|


Performance on Qwen1.5(2024, Alibaba) series models
|       Approaches               | Xsum(ROUGE1)       |  MT EN-NL(BLEU4)     |     GSM8K(ACCURACY)      |   
|:-------------------------:|:-------------:|:---------:|:---------:|
|Prompt Teacher**(Qwen1.5-4B)**| 38.15 |21.36 |40.71|
|Prompt Student**(T5-0.4B)**|8.80|0.51|0|
|SeqKD||||
|KL||||
|JS||||
|TVD||||
|LLMR||||
|LLMR + Mean baseline||||
|LLMR + Min-Var baseline||||
|KETCHUP(best K)||||
## Usage 
First set the environment variable.
```
export HF_HOME='your HF token'
```
Then install the environment. Note that we found error in transformers==4.51, please use early versions.
```
pip install -r requirements.txt
```


**Note: please set the path to your dataset, student model, and teacher model in the corresponding shells.**
```
cd scripts
sh distill_xsum.sh
sh reinforce_xsum.sh
```

#### PPO
Go to the `scripts/rlhf/ppo' folder and train the gemma-2b-it model with the default parameters.

**Note: please set the path to your reward model in the corresponding shells.**
```
cd scripts/rlhf/ppo
sh train_ppo.sh
sh train_ppo.grm.sh
sh train_ppo_ensemble.sh
```


# KETCHUP: KETCHUP: K-Step Return Estimation for Sequential Knowledge Distillation

## Models


Performance on Flan-T5(2024, GOOGLE) series models. The best K values are 2, 4, and 8 for the three datasets, respectively.


|       Approaches               | Xsum(ROUGE1)       |  MT EN-NL(BLEU4)     |     GSM8K(ACCURACY)      |   
|:-------------------------:|:-------------:|:---------:|:---------:|
|Prompt Teacher**(Flan-T5-3B)**| 41.32 | 25.36 | 40.71 |
|Prompt Student**(T5-0.4B)**|19.60 | 0.95 | 0.00 |
|SeqKD| 33.54 | 22.09 | 20.02 |
|KL| 34.36 | 22.35 | 23.96 |
|JS| 34.87 |22.55|24.72|
|TVD|35.17|22.63|24.94|
|LLMR|35.54|22.72|25.21|
|LLMR + Mean baseline|35.60|22.67|25.39|
|LLMR + Min-Var baseline|35.59|22.70|25.10|
|KETCHUP(best K)|36.03|22.95|25.71|


KD studies on seq2seq tasks have largely centred on encoder-decoder structures such as T5 and BART models. To answer reviewers’ likely question about KETCHUP’s behaviour on recent popular decoder-only architectures, we also applied it to the Qwen1.5 model series and report the results in the following table. The best K values are 2, 2, and 16 for the three datasets, respectively.
|       Approaches               | Xsum(ROUGE1)       |  MT EN-NL(BLEU4)     |     GSM8K(ACCURACY)      |   
|:-------------------------:|:-------------:|:---------:|:---------:|
|Prompt Teacher**(Qwen1.5-4B)**| 38.15 |21.32 |42.08|
|Prompt Student**(Qwen1.5-0.5B)**|8.80|0.02|0|
|KL|31.29|15.76|26.31|
|TVD|31.18|16.22|26.99|
|LLMR|31.61|15.90|27.29|
|KETCHUP(best K)|32.28|16.46|28.13|
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
sh distill_xsum.sh #non-RL distillation
sh reinforce_xsum.sh  #RL distillation
```




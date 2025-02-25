#  中文拼写和语法纠错
[**🇨🇳中文**](https://github.com/TW-NLP/ChineseErrorCorrector/blob/main/README.md) 

<div align="center">
  <a href="https://github.com/TW-NLP/ChineseErrorCorrector">
    <img src="images/image_fx_.jpg" alt="Logo" height="156">
  </a>
</div>



-----------------



## 介绍
支持中文拼写和语法错误纠正，并开源拼写和语法错误的增强工具，荣获2024CCL 冠军 🏆，[查看论文](https://aclanthology.org/2024.ccl-3.31/) 。

## 🔥🔥🔥 新闻

[2025/02/25] 使用200万纠错数据进行多轮迭代训练，发布了[twnlp/ChineseErrorCorrector2-7B](https://huggingface.co/twnlp/ChineseErrorCorrector2-7B)，在 [NaCGEC-2023NLPCC官方评测数据集](https://github.com/masr2000/NaCGEC)上，超越第一名华为10个点，遥遥领先，推荐使用！

[2025/02] 为方便部署，发布了[twnlp/ChineseErrorCorrector-1.5B](https://huggingface.co/twnlp/ChineseErrorCorrector-1.5B)

[2025/01] 使用38万开源数据，基于Qwen2.5训练中文拼写纠错模型，支持语似、形似等错误纠正，发布了[twnlp/ChineseErrorCorrector-7B](https://huggingface.co/twnlp/ChineseErrorCorrector-7B)，[twnlp/ChineseErrorCorrector-32B-LORA](https://huggingface.co/twnlp/ChineseErrorCorrector-32B-LORA/tree/main)

[2024/06] v0.1.0版本：开源一键语法错误增强工具，该工具可以进行14种语法错误的增强，不同行业可以根据自己的数据进行错误替换，来训练自己的语法和拼写模型。详见[Tag-v0.1.0](https://github.com/TW-NLP/ChineseErrorCorrector/tree/0.1.0)

## 数据集

| 数据集名称     | 数据链接                                                                                                      | 数据量和类别说明                                                                                                            | 描述          |
|:--------------|:-----------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------|:------------|
| CSC（拼写纠错数据集） |[twnlp/csc_data](https://huggingface.co/datasets/twnlp/csc_data)  | W271K：279,816 条，Medical：39,303 条，Lemon：22,259 条，ECSpell：6,688 条，CSCD：35,001 条 | 中文拼写纠错的数据集 |
| CGC（语法纠错数据集） |[twnlp/cgc_data](https://huggingface.co/datasets/twnlp/cgc_data)  | CGED：20449 条，FCGEC：37354 条，MuCGEC：2467 条，NaSGEC：7568条 | 中文语法纠错的数据集 |
| Lang8+HSK（百万语料-拼写和语法错误混合数据集） |[twnlp/lang8_hsk](https://huggingface.co/datasets/twnlp/lang8_hsk)  | 1568885条 | 中文拼写和语法数据集 |



## 拼写纠错评测
- 评估指标：F1


| Model Name       | Model Link                                                                                                              | Base Model                 | Avg        | SIGHAN-2015(通用) | EC-LAW(法律)| EC-MED(医疗)| EC-ODW(公文)|
|:-----------------|:------------------------------------------------------------------------------------------------------------------------|:---------------------------|:-----------|:------------|:-------|:-------|:--------|
| twnlp/ChineseErrorCorrector-1.5B        | https://huggingface.co/twnlp/ChineseErrorCorrector-1.5B/tree/main                                    | Qwen/Qwen2.5-1.5B-Instruct | 0.459     | 0.346      | 0.517 | 0.433 | 0.540     |
| twnlp/ChineseErrorCorrector-7B        | https://huggingface.co/twnlp/ChineseErrorCorrector-7B/tree/main                                    | Qwen/Qwen2.5-7B-Instruct | 0.712     | 0.592      | 0.787 | 0.677 | 0.793     |
| twnlp/ChineseErrorCorrector-32B-LORA        | https://huggingface.co/twnlp/ChineseErrorCorrector-32B-LORA/tree/main                                    | Qwen/Qwen2.5-32B-Instruct |  0.757   |    0.594   | 0.776 |0.794 |   0.864  |

## 文本纠错评测（拼写错误+语法错误）
- 评估工具：ChERRANT  [评测工具](https://github.com/HillZhang1999/MuCGEC) 
- 评估指标：F1-0.5(语法)、F1(拼写)


| Test Dataset       | Model Name       | Model Link                                                                                                              | Base Model                 |    Prec     | Rec | F0.5 |
|:-----------------|:-----------------|:------------------------------------------------------------------------------------------------------------------------|:---------------------------|:-----------|:------------|:-------|
| NaSGEC-NLPCC2023    |  twnlp/ChineseErrorCorrector2-7B | https://huggingface.co/twnlp/ChineseErrorCorrector2-7B       | Qwen/Qwen2.5-7B-Instruct                                  |  | 待评测     | 待评测      | 待评测 |


## 使用
### transformers 
```shell
# pip install transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "twnlp/ChineseErrorCorrector-7B"

device = "cuda" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

input_content = "你是一个拼写纠错专家，对原文进行错别字纠正，不要更改原文字数，原文为：\n少先队员因该为老人让坐。"

messages = [{"role": "user", "content": input_content}]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)

print(input_text)

inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=1024, temperature=0, do_sample=False, repetition_penalty=1.08)

print(tokenizer.decode(outputs[0]))

```

### VLLM

```shell
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("twnlp/ChineseErrorCorrector-7B")

# Pass the default decoding hyperparameters of twnlp/ChineseErrorCorrector-7B
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(repetition_penalty=1.05, max_tokens=512)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model="twnlp/ChineseErrorCorrector-7B")

# Prepare your prompts
prompt = "少先队员因该为老人让坐。"
messages = [
    {"role": "system", "content": "你是一个拼写纠错专家，对原文进行错别字纠正，不要更改原文字数，原文为："},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# generate outputs
outputs = llm.generate([text], sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}") 
```

## Citation

If this work is helpful, please kindly cite as:

```bibtex

@inproceedings{wei2024中小学作文语法错误检测,
  title={中小学作文语法错误检测, 病句改写与流畅性评级的自动化方法研究},
  author={Wei, Tian},
  booktitle={Proceedings of the 23rd Chinese National Conference on Computational Linguistics (Volume 3: Evaluations)},
  pages={278--284},
  year={2024}
}
```


## Star History

![Star History Chart](https://api.star-history.com/svg?repos=TW-NLP/ChineseErrorCorrector&type=Date)
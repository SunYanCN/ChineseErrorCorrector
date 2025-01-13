#  中文拼写和语法纠错
[**🇨🇳中文**](https://github.com/TW-NLP/ChineseErrorCorrector/blob/main/README.md) 

<div align="center">
  <a href="https://github.com/TW-NLP/ChineseErrorCorrector">
    <img src="images/image_fx_.jpg" alt="Logo" height="156">
  </a>
</div>



-----------------



## 介绍
支持中文拼写和语法错误纠正，以及拼写和语法错误的增强工具。

## 新闻
[2025/01] 新增了基于Qwen2.5的中文拼写纠错模型，支持语似、形似等错误纠正，发布了[twnlp/ChineseErrorCorrector-7B](https://huggingface.co/twnlp/ChineseErrorCorrector-7B)。

[2024/06] v0.1.0版本：开源一键语法错误增强工具，该工具可以进行14种语法错误的增强，不同行业可以根据自己的数据进行错误替换，来训练自己的语法和拼写模型。详见[Tag-v0.1.0](https://github.com/TW-NLP/ChineseErrorCorrector/tree/0.1.0)

### 评估结果
- 评估指标：F1


| Model Name       | Model Link                                                                                                              | Base Model                 | Avg        | SIGHAN-2015 | EC-LAW | EC-MED   | EC-ODW |
|:-----------------|:------------------------------------------------------------------------------------------------------------------------|:---------------------------|:-----------|:------------|:-------|:-------|:--------|
| twnlp/ChineseErrorCorrector-7B        | https://huggingface.co/twnlp/ChineseErrorCorrector-7B/tree/main                                    | Qwen/Qwen2.5-7B-Instruct | 0.712     | 0.592      | 0.787 | 0.677 | 0.793     |
| twnlp/ChineseErrorCorrector-32B        | https://huggingface.co/twnlp/ChineseErrorCorrector-32B/tree/main                                    | Qwen/Qwen2.5-32B-Instruct |     |       |  | |     |

## 使用

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
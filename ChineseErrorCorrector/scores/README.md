# 通用文本纠错评测工具（Common-ERRANT）

## 一、环境准备

``` sh
conda create -n common_errant -y python=3.10
conda activate common_errant
pip install -r requirements.txt
# If you are in mainland China, you can set the mirror as follows:
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

- `test.txt`：测试文本文件
- `label.txt`：标签（参考答案）文件
- `pred.txt`：模型预测结果文件

---

## 二、生成 M2 文件

进入命令目录：

```bash
cd ChineseErrorCorrector/scores/commands
```

生成参考标准 M2 文件：

```bash
python parallel_to_m2.py -orig path/to/test.txt -cor path/to/label.txt -out path/to/reference.m2 -lang {language_code}
```

生成预测结果 M2 文件：

```bash
python parallel_to_m2.py -orig path/to/test.txt -cor path/to/pred.txt -out path/to/hypothesis.m2 -lang {language_code}
```

> 其中，`{language_code}` 例如：`zh`、`en` 等，依据具体语言填写。

---

## 三、计算 F1 值

运行以下命令对比参考标准和预测结果，计算 F1 分数：

```bash
python compare_m2.py reference.m2 hypothesis.m2
```



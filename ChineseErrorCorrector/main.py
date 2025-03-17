import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
from ChineseErrorCorrector.llm.hf_infer import HFTextCorrectInfer
from ChineseErrorCorrector.llm.vllm_infer import VLLMTextCorrectInfer
from ChineseErrorCorrector.config import Qwen2TextCorConfig
from ChineseErrorCorrector.utils import res_format


class ErrorCorrect(object):
    """
    中文拼写和语法错误纠正
    """

    def __init__(self):
        if Qwen2TextCorConfig.USE_VLLM:
            self.llm_correct_vllm = VLLMTextCorrectInfer()
        else:
            self.llm_correct_hf = HFTextCorrectInfer()

    async def vllm_infer(self, input_list):
        res = await self.llm_correct_vllm.infer_sentence(input_list)
        res_ = res_format(input_list, res)
        return res_

    def hf_infer(self, input_list):
        res = self.llm_correct_hf.infer(input_list)
        res_ = res_format(input_list, res)
        return res_


if __name__ == '__main__':
    ec = ErrorCorrect()

    # 测试用例
    input_text = [
        "少先队员因该为老人让坐。",
        "大约半个小时左右"
    ]
    if Qwen2TextCorConfig.USE_VLLM:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(ec.vllm_infer(input_text))
        print(result)
    else:
        result = ec.hf_infer(input_text)
        print(result)

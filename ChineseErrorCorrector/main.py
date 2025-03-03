import os
import sys
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ChineseErrorCorrector.llm.qwen_text_correct_infer import VLLMTextCorrectInfer


class ErrorCorrect(object):
    """
    中文拼写和语法错误纠正
    """

    def __init__(self):
        self.llm_correct = VLLMTextCorrectInfer()

    async def text_infer(self, input_list):
        result = await self.llm_correct.infer_sentence(input_list)
        return result


if __name__ == '__main__':
    ec = ErrorCorrect()

    #
    input_text = [
        "少先队员因该为老人让坐。"
    ]
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(ec.text_infer(input_text))
    print(result)

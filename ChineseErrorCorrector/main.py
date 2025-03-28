import os
import sys

from ChineseErrorCorrector.utils.correct_tools import res_format

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
from ChineseErrorCorrector.llm.infer.hf_infer import HFTextCorrectInfer
from ChineseErrorCorrector.llm.infer.vllm_infer import VLLMTextCorrectInfer
from ChineseErrorCorrector.config import Qwen2TextCorConfig
from tqdm import tqdm


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
    # input_text = [
    #     "党的十八大以来，以习近平主席为核心的党中央引领我国经济社会发展取得历史性成就、发生历史性变革，在实践中形成和发展了习近平经济思想，为新征程上做好经济工作提供了行动指南。",
    #     "近段时间发布的多个数据显示，，餐饮、文旅等线下消费热度上升，",
    #     "全国各地的“烟火气”火速回归，为提振全年经济开了好头。",
    #     "作为拉动我国经济增涨的“第一动力”，消费升温为上下游市场负苏增添了暖意。",
    #     "业内专家指出，随着传统消回暖，叠加IP消费、健康消费、兴趣消费等新增长增长点的迅速崛起，",
    #     "消费增长的结构性潜力将加快释放，我国经济回稳之势进一步确立。",
    #     "二O二三年以来，餐饮、电影、旅游多种等消费快速恢复。餐厅等位已成常态，",
    #     "春节期间，湖北长沙市文和友海信广场店以排队超过4500桌冲上了微博热搜。",
    #     "电影票房也迎来高增长，猫眼电影数据展示，截至2月6日，春节档电影《满江红》和《流浪地球2票房总额已超82亿元。 ",
    #     "国信中心大数据发展部研究员表示，各地市委政府多项高频消费数据反映今年以来我国消费市场持续回暖。",
    #     "线下消费热度快速恢复，至2月31日已较去年12月的低点大幅提升22.8个点,各大商圈人气日渐兴旺."
    # ]
    input_text = [i.strip().split("\t")[-1] for i in
                  open("/home/tianwei/TW_NLP/ChineseErrorCorrector/data/business_data/MuCGEC_test.txt",
                       encoding="utf-8")]
    count = 1
    with open("./infer.txt", "w", encoding="utf-8") as file_write:
        for i in tqdm(input_text):
            re_i = ec.hf_infer([i])
            file_write.write(str(count) + "\t" + i + "\t" + re_i[0]['target'] + "\n")
            count += 1
    file_write.close()

    # if Qwen2TextCorConfig.USE_VLLM:
    #     loop = asyncio.get_event_loop()
    #     result = loop.run_until_complete(ec.vllm_infer(input_text))
    #     print(result)
    # else:
    #     result = ec.hf_infer(input_text)
    #     with open("./infer.txt", "w", encoding="utf-8") as file_write:
    #         for line in result:
    #             print(line)
    #             file_write.write(line['target'] + "\n")
    #     file_write.close()
    # print(result)

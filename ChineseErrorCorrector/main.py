from ChineseErrorCorrector.dat import GrammarErrorDat

from ChineseErrorCorrector.llm.qwen_text_correct_infer import VLLMTextCorrectInfer


class ErrorCorrect(object):
    """
    中文拼写和语法错误纠正
    """

    def __init__(self):
        self.spell = VLLMTextCorrectInfer()
        self.dat = GrammarErrorDat()

    def spell_infer(self, input_list):
        result = self.spell.infer_sentence(input_list)
        return result


if __name__ == '__main__':
    ec = ErrorCorrect()

    spell_test = [
        "少先队员因该为老人让坐。",
        "我的明字叫小明",
        "你好"
    ]
    print(ec.spell_infer(spell_test))

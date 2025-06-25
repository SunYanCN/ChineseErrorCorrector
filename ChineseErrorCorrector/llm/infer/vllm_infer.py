import torch
import uuid

from ChineseErrorCorrector.config import DEVICE, DEVICE_COUNT, TextCorrectConfig
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, AutoModel, \
    AutoModelForCausalLM, set_seed
from vllm import SamplingParams, AsyncLLMEngine, AsyncEngineArgs
import time
import copy
import asyncio
import traceback
from ChineseErrorCorrector.utils.correct_tools import torch_gc


class VLLMTextCorrectInfer(object):

    def __init__(self, ):
        set_seed(42)
        self.prompt_prefix = "你是一个文本纠错专家，纠正输入句子中的语法错误，并输出正确的句子，输入句子为："

        if DEVICE == 'cpu':
            raise ValueError("VLLM does not support CPU inference, please use GPU for inference.")
        else:
            device = torch.device(DEVICE)
            capability = torch.cuda.get_device_capability(device)
            # T4 算力为7.5 无法使用BF16，改为float16
            if capability[0] < 8:
                model_args = AsyncEngineArgs(TextCorrectConfig.DEFAULT_CKPT_PATH,
                                             tensor_parallel_size=DEVICE_COUNT,
                                             dtype='float16',
                                             trust_remote_code=True
                                             , gpu_memory_utilization=TextCorrectConfig.GPU_MEMARY,
                                             max_model_len=TextCorrectConfig.MAX_LENGTH)
            else:
                model_args = AsyncEngineArgs(TextCorrectConfig.DEFAULT_CKPT_PATH,
                                             tensor_parallel_size=DEVICE_COUNT,
                                             trust_remote_code=True
                                             , gpu_memory_utilization=TextCorrectConfig.GPU_MEMARY,
                                             max_model_len=TextCorrectConfig.MAX_LENGTH)
            self.model = AsyncLLMEngine.from_engine_args(model_args)
            self.tokenizer = AutoTokenizer.from_pretrained(TextCorrectConfig.DEFAULT_CKPT_PATH, trust_remote_code=True)

    async def generate_async(self, query):
        """
        完整的异步生成方法
        :param query:
        :return:
        """
        try:
            # 初始化 SamplingParams
            sampling_params = SamplingParams(
                seed=42,
                temperature=0.6,
                top_k=20,
                top_p=0.95,
                stop=[],
                max_tokens=TextCorrectConfig.MAX_LENGTH
            )

            request_id = str(uuid.uuid4())
            prompt = self.prompt_prefix + query  # 添加 no_thinking 后缀以避免思考过程

            messages = [
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Set to False to strictly disable thinking
            )

            # # 获取异步生成器
            result = self.model.generate(
                text,  # 或者使用 prompt=messages
                sampling_params=sampling_params,
                request_id=request_id
            )

            # 使用异步非流式输出处理结果
            complete_text = await self.no_stream_output(result, request_id)

            return complete_text

        except Exception as e:
            print(f"Error in generation: {e}")
            return None

    async def no_stream_output(self, result, request_id):
        """
        纠错大模型的异步非流式
        :param result: 异步迭代器
        :param request_id: 请求id
        :return:
        """

        # 用于累积完整响应
        complete_text = ""

        try:
            count = 0

            # 使用 async for 来迭代异步生成器
            async for request_output in result:
                count += 1

                # 检查是否是我们要的请求ID
                if request_output.request_id == request_id:

                    if request_output.outputs and len(request_output.outputs) > 0:
                        output = request_output.outputs[0]

                        # 累积完整文本（VLLM会返回累积的完整文本，不需要手动拼接）
                        complete_text = output.text

                        # 如果生成完成，退出循环
                        if request_output.finished:
                            break
                else:
                    pass

        except Exception as e:
            traceback.print_exc()
            raise

        # 检查是否获得了完整结果
        if not complete_text:
            raise ValueError("生成器没有返回任何文本")

        return complete_text

    async def infer_sentence(self, user_inputs):
        tasks = [self.generate_async(query_i) for query_i in user_inputs]
        result = await asyncio.gather(*tasks, return_exceptions=True)

        return result


if __name__ == '__main__':
    pass

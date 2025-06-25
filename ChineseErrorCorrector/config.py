import os
import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(PROJECT_DIR, 'pre_model')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_COUNT = 1 if torch.cuda.device_count() == 1 else int(torch.cuda.device_count() // 2) * 2


class LTPPath(object):
    LTP_MODEL_DIR = os.path.join(MODEL_DIR, 'ltp_tiny')
    LTP_DATA_PATH = os.path.join(DATA_DIR, 'dat_data')


# 需要将ChineseErrorCorrector3-4B下载下来，放在pre_model中
class TextCorrectConfig(object):
    """
    模型位置
    """
    # 是否采用VLLM进行异步推理，工程化推荐
    USE_VLLM = False
    MAX_LENGTH = 32000
    DEFAULT_CKPT_PATH = os.path.join(MODEL_DIR, 'ChineseErrorCorrector3-4B')
    GPU_MEMARY = 0.9


class TrainConfig(object):
    """
    模型数据与模型保存
    """
    TRAIN_PATH = os.path.join(DATA_DIR, 'business_data', 'train.json')
    DEV_PATH = os.path.join(DATA_DIR, 'business_data', 'valid.json')
    SAVE_PATH = os.path.join(DATA_DIR, 'business_data', 'model_output')
    CACHE_PATH = os.path.join(DATA_DIR, 'cache')
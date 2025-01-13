import os
import torch

# os.environ['CUDA_VISIBLE_DEVICES']="2,3"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(PROJECT_DIR, 'pre_model')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_COUNT = 1 if torch.cuda.device_count() == 1 else int(torch.cuda.device_count() // 2) * 2


class LTPPath(object):
    LTP_MODEL_DIR = os.path.join(MODEL_DIR, 'ltp_tiny')
    LTP_DATA_PATH = os.path.join(DATA_DIR, 'dat_data')


class Qwen2TextCorConfig(object):
    MAX_LENGTH = 32000
    DEFAULT_CKPT_PATH = os.path.join(MODEL_DIR, 'ChineseErrorCorrector-7B')
    GPU_MEMARY = 0.7

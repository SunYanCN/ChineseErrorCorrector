import argparse
from ChineseErrorCorrector.config import TrainConfig, Qwen2TextCorConfig
from loguru import logger
import torch

from ChineseErrorCorrector.llm.train.train_lora import TrainLLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file',
                        default=TrainConfig.TRAIN_PATH,
                        type=str, help='Train file')
    parser.add_argument('--dev_file',
                        default=TrainConfig.DEV_PATH,
                        type=str, help='Dev file')
    parser.add_argument('--model_type', default='auto', type=str, help='Transformers model type')

    parser.add_argument('--model_name', default=Qwen2TextCorConfig.DEFAULT_CKPT_PATH, type=str,
                        help='LLM path')
    parser.add_argument('--do_train', default=True, help='Whether to run training.')
    parser.add_argument('--do_predict', default=True, help='Whether to run predict.')
    parser.add_argument('--output_dir', default=TrainConfig.SAVE_PATH,
                        type=str, help='Model output directory')
    parser.add_argument('--device_map', default="auto")
    parser.add_argument('--int8', default=False)
    parser.add_argument('--int4', default=False)
    parser.add_argument('--bf16', default=True, help='Whether to use bf16 mixed precision training.')
    parser.add_argument('--fp16', default=False)
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--qlora', default=False)
    parser.add_argument('--num_train_epochs', default=5, help='train llm epoch')
    parser.add_argument('--learning_rate', default=2e-5, help='llm learning rate')
    parser.add_argument('--logging_steps', default=50, help='currect % logging_steps==0,print log')
    parser.add_argument('--max_steps', default=-1)
    parser.add_argument('--per_device_train_batch_size', default=8)
    parser.add_argument('--gradient_checkpointing', default=True)
    parser.add_argument('--torch_compile', default=False)
    parser.add_argument('--gradient_accumulation_steps', default=1)
    parser.add_argument('--warmup_steps', default=50)
    parser.add_argument('--save_steps', default=1000)
    parser.add_argument('--optimizer', default='adamw_torch')
    parser.add_argument('--save_strategy', default='steps')
    parser.add_argument('--eval_steps', default=1000)
    parser.add_argument('--save_total_limit', default=10)
    parser.add_argument('--remove_unused_columns', default=False)
    parser.add_argument('--report_to', default='tensorboard')
    parser.add_argument('--overwrite_output_dir', default=True)
    parser.add_argument('--max_eval_samples', default=1000)

    parser.add_argument('--peft_type', default='LORA')
    parser.add_argument('--use_peft', default=True)
    parser.add_argument('--lora_target_modules', default=['all'])
    parser.add_argument('--lora_r', default=8)
    parser.add_argument('--lora_alpha', default=16)
    parser.add_argument('--lora_dropout', default=0.05)
    parser.add_argument('--lora_bias', default='none')
    parser.add_argument('--no_cache', default=False)
    parser.add_argument('--dataset_class', default=None)
    parser.add_argument('--cache_dir', default=TrainConfig.CACHE_PATH)
    parser.add_argument('--preprocessing_num_workers', default=4)
    parser.add_argument('--reprocess_input_data', default=True)
    parser.add_argument("--resume_from_checkpoint", default=TrainConfig.SAVE_PATH)

    parser.add_argument('--prompt_template_name', default='qwen', type=str, help='Prompt template name')
    parser.add_argument('--max_seq_length', default=512, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=512, type=int, help='Output max sequence length')
    parser.add_argument("--local_rank", type=int, help="Used by dist launchers")
    args = parser.parse_args()
    logger.info(args)

    if args.do_train:
        model = TrainLLM(args)
        model.train_model(train_data=args.train_file, output_dir=args.output_dir, eval_data=args.dev_file)


if __name__ == '__main__':
    main()

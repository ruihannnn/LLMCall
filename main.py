import os
import time
import logging
import asyncio
import json
from dotenv import load_dotenv
from dataset_config import DatasetConfig
from chatllm import ChatLLM
import response_processor


def setup_logging():
    """设置日志配置"""
    log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), logging.INFO)
    
    # 获取日志文件路径
    log_file = os.getenv('LOG_FILE', 'log/log.txt')
    
    # 创建日志目录（如果不存在）
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        print(f"✅ 创建日志目录: {log_dir}")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.getLogger(__name__).info("日志系统初始化完成")


def init_chat_llm():
    """初始化ChatLLM实例"""
    
    # 必要配置检查
    required_vars = ['LLM_URL', 'INPUT_PATH', 'OUTPUT_PATH', 'INPUT_COLUMNS', 'OUTPUT_COLUMN', 'PROMPT_KEY']
    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"缺少必要的环境变量: {var}")
    
    # 解析配置
    input_columns = [col.strip() for col in os.getenv('INPUT_COLUMNS').split(',')]
    
    output_column = os.getenv('OUTPUT_COLUMN')
    if ',' in output_column:
        output_column = [col.strip() for col in output_column.split(',')]
    
    output_prompt_column = os.getenv('OUTPUT_PROMPT_COLUMN')
    if output_prompt_column:
        if ',' in output_prompt_column:
            output_prompt_column = [col.strip() for col in output_prompt_column.split(',')]
    
    prompt_keys = os.getenv('PROMPT_KEY').split(',') if ',' in os.getenv('PROMPT_KEY') else [os.getenv('PROMPT_KEY')]
    prompt_keys = [key.strip() for key in prompt_keys]
    
    # 数据集配置
    dataset_config = DatasetConfig(
        input_path=os.getenv('INPUT_PATH'),
        output_path=os.getenv('OUTPUT_PATH'),
        input_columns=input_columns,
        output_column=output_column,
        output_prompt_column=output_prompt_column,
        batch_size=int(os.getenv('BATCH_SIZE', 1000)),
        num_responses=int(os.getenv('NUM_RESPONSES', 1)),
        max_rows=int(os.getenv('MAX_ROWS', 0)) or None,
        max_concurrent_tasks=int(os.getenv('MAX_CONCURRENT_TASKS', 512))
    )
    
    # LLM配置
    stop_str = os.getenv("STOP", '["<|endoftext|>"]')
    try:
        stop_tokens = json.loads(stop_str)
    except json.JSONDecodeError:
        stop_tokens = [stop_str.strip('"\'')]
    
    llm_config = {
        "model": os.getenv('MODEL_NAME', 'qwen'),
        "temperature": float(os.getenv('TEMPERATURE', 0.6)),
        "top_p": float(os.getenv('TOP_P', 0.95)),
        "max_tokens": int(os.getenv('MAX_TOKENS', 100000)),
        "stop": stop_tokens
    }
    
    # 获取响应处理器 - 直接从模块获取，无需字典映射
    processor_name = os.getenv('RESPONSE_PROCESSOR', 'simple_response_processor')
    try:
        response_processor_func = getattr(response_processor, processor_name)
    except AttributeError:
        raise ValueError(f"未找到响应处理器: {processor_name}")
    
    return ChatLLM(
        llm_url=os.getenv('LLM_URL'),
        prompt_keys=prompt_keys,
        response_processor=response_processor_func,
        dataset_config=dataset_config,
        api_key=os.getenv('API_KEY', 'test'),
        generate_config=llm_config
    )


async def main():
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        chat_llm = init_chat_llm()
        logger.info(f"开始处理: {chat_llm.dataset_config.input_path}")
        
        start_time = time.time()
        await chat_llm.process_dataset()
        elapsed_time = time.time() - start_time
        logger.info(f"处理完成，用时: {elapsed_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    load_dotenv('.env', override=True)  # 使用.env文件
    asyncio.run(main())
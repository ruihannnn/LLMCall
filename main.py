import os
import time
import logging
from dotenv import load_dotenv
from dataset_config import DatasetConfig
from chat_llm import ChatLLM
import response_processor
import json


def setup_logging():
    """设置日志配置"""
    log_file = os.getenv('LOG_FILE', 'log/log.txt')
    log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), logging.INFO)
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True) if os.path.dirname(log_file) else None
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.getLogger(__name__).info(f"日志初始化完成: {log_file}")


def init_chat_llm():
    """初始化ChatLLM实例"""
    
    # 解析多个PROMPT_KEY
    prompt_keys = [key.strip() for key in os.getenv('PROMPT_KEY', '').split(',')]
    output_columns = [col.strip() for col in os.getenv('OUTPUT_COLUMN', '').split(',')]
    output_prompt_columns_str = os.getenv('OUTPUT_PROMPT_COLUMN', '')
    output_prompt_columns = [col.strip() for col in output_prompt_columns_str.split(',')] if output_prompt_columns_str else []
    
    # 解析响应处理器
    response_processors_str = os.getenv('RESPONSE_PROCESSOR', '')
    response_processor_names = [name.strip() for name in response_processors_str.split(',')]
    
    # 验证数量一致性 - 支持新功能
    if len(prompt_keys) == 1:
        # 单个prompt，支持多个输出列和多个处理器
        # OUTPUT_PROMPT_COLUMN必须为None或者为1个
        if output_prompt_columns and len(output_prompt_columns) != 1:
            raise ValueError(f"单个PROMPT_KEY时，OUTPUT_PROMPT_COLUMN数量({len(output_prompt_columns)})必须为None或1个")
        
        # RESPONSE_PROCESSOR必须与OUTPUT_COLUMN数量相同
        if len(response_processor_names) != len(output_columns):
            raise ValueError(f"单个PROMPT_KEY时，RESPONSE_PROCESSOR数量({len(response_processor_names)})必须与OUTPUT_COLUMN数量({len(output_columns)})相同")
            
        # 多个处理器，按顺序对应
        response_processors = [getattr(response_processor, name) for name in response_processor_names]
            
    else:
        # 多个prompt，必须一一对应
        if len(prompt_keys) != len(output_columns):
            raise ValueError(f"多个PROMPT_KEY时，PROMPT_KEY数量({len(prompt_keys)})与OUTPUT_COLUMN数量({len(output_columns)})必须一致")
        
        if output_prompt_columns and len(prompt_keys) != len(output_prompt_columns):
            raise ValueError(f"多个PROMPT_KEY时，PROMPT_KEY数量({len(prompt_keys)})与OUTPUT_PROMPT_COLUMN数量({len(output_prompt_columns)})必须一致")
        
        if len(response_processor_names) != 1 and len(response_processor_names) != len(prompt_keys):
            raise ValueError(f"多个PROMPT_KEY时，RESPONSE_PROCESSOR数量({len(response_processor_names)})必须为1个或与PROMPT_KEY数量({len(prompt_keys)})相同")
        
        # 处理响应处理器
        if len(response_processor_names) == 1:
            # 单个处理器，复制到所有prompt
            response_processors = [getattr(response_processor, response_processor_names[0])] * len(prompt_keys)
        else:
            # 多个处理器，按顺序对应
            response_processors = [getattr(response_processor, name) for name in response_processor_names]
    
    # 数据集配置
    dataset_config = DatasetConfig(
        input_path=os.getenv('INPUT_PATH'),
        output_path=os.getenv('OUTPUT_PATH'),
        input_columns=os.getenv('INPUT_COLUMNS', '').split(','),
        output_column=output_columns,
        output_prompt_column=output_prompt_columns if output_prompt_columns else None,
        batch_size=int(os.getenv('BATCH_SIZE', 1000)),
        max_rows=int(os.getenv('MAX_ROWS', 0)) or None,
        max_thread_num=int(os.getenv('MAX_THREAD_NUM', 512))
    )
    
    # LLM配置
    llm_config = {
        "model": os.getenv('MODEL_NAME', 'qwen'),
        "temperature": float(os.getenv('TEMPERATURE', 0.6)),
        "top_p": float(os.getenv("TOP_P", 0.95)),
        "max_tokens": int(os.getenv('MAX_TOKENS', 4096)),
        "stop": json.loads(os.getenv("STOP", '["<|endoftext|>"]'))
    }
    
    # 创建ChatLLM实例
    return ChatLLM(
        llm_url=os.getenv('LLM_URL'),
        prompt_key=prompt_keys,
        response_processor=response_processors,
        dataset_config=dataset_config,
        api_key=os.getenv('API_KEY', 'test'),
        generate_config=llm_config
    )


def main():
    
    load_dotenv(dotenv_path=".env.example",override=True)
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        chat_llm = init_chat_llm()
        logger.info(f"开始处理: {chat_llm.dataset_config.input_path} -> {chat_llm.dataset_config.output_path}")
        
        start_time = time.time()
        chat_llm.process_dataset()
        logger.info(f"处理完成，用时: {time.time() - start_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
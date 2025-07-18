import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# =================重要！！！！！！！！！！！=================
# 如果想要增加回调函数，那么回调函数的参数必须为字符串类型，且必须传入单个参数
# （类似于def simple_response_processor(response: str)的形式）
# 
# 在该文件下增加回调函数后，将.env.example文件中RESPONSE_PROCESSOR的值改为新增函数名即可
# 例如，新增的函数名为new_response_processor,那么将RESPONSE_PROCESSOR修改为new_response_processor，就会自动调用这个函数对输出进行后处理
# =============================================================


def json_load_response_processor(response: str) -> Any:
    """解析JSON"""
    if not response:
        logger.warning("收到空响应")
        return None
    
    # 移除thinking标签后的内容
    response = response.split('</think>')[-1]
    
    try:
        # 清理格式并解析JSON
        cleaned_response = response.strip('\n').strip('```json\n').strip('```')
        result = json.loads(cleaned_response)
        logger.debug("成功解析JSON响应")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {e}")
        return None
    except Exception as e:
        logger.error(f"处理响应时发生未知错误: {e}")
        return None


def no_think_response_processor(response: str) -> str:
    """移除</think>标签后的内容"""
    if not response:
        logger.warning("收到空响应")
        return None
    
    result = response.split('</think>')[-1]
    logger.debug("成功处理no_think响应")
    return result


def simple_response_processor(response: str) -> str:
    """直接返回文本"""
    if not response:
        logger.warning("收到空响应")
        return None
    
    result = response.strip()
    logger.debug("成功处理simple响应")
    return result

def test_respoonse_processor(response:str)-> str:
    return '111'
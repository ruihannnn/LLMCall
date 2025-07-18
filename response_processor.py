import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def simple_response_processor(response: str) -> Optional[str]:
    """简单响应处理器 - 直接返回清理后的文本"""
    if not response:
        logger.warning("收到空响应")
        return None
    
    result = response.strip()
    return result if result else None


def no_think_response_processor(response: str) -> Optional[str]:
    """移除thinking标签的响应处理器"""
    if not response:
        logger.warning("收到空响应")
        return None
    
    # 移除thinking标签后的内容
    result = response.split('</think>')[-1].strip()
    return result if result else None


def json_load_response_processor(response: str) -> Optional[Any]:
    """JSON响应处理器 - 解析JSON格式的响应"""
    if not response:
        logger.warning("收到空响应")
        return None
    
    # 移除thinking标签
    response = response.split('</think>')[-1]
    
    try:
        # 清理JSON格式
        cleaned = _clean_json_format(response)
        result = json.loads(cleaned)
        logger.debug("成功解析JSON响应")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {e}")
        return None
    except Exception as e:
        logger.error(f"处理JSON响应失败: {e}")
        return None


def _clean_json_format(response: str) -> str:
    """清理JSON响应格式"""
    response = response.strip()
    
    # 移除markdown代码块标记
    response = re.sub(r'^```(?:json)?\s*\n?', '', response, flags=re.IGNORECASE | re.MULTILINE)
    response = re.sub(r'\n?```\s*$', '', response, flags=re.MULTILINE)
    
    # 移除首尾的非JSON字符
    return response.strip('\n\r\t `')


import os
import logging
from typing import List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """数据集配置类"""
    
    input_path: str
    output_path: str
    input_columns: List[str]
    output_column: Union[str, List[str]]
    output_prompt_column: Optional[Union[str, List[str]]] = None
    batch_size: int = 1000
    num_responses: int = 1
    max_rows: Optional[int] = None
    max_concurrent_tasks: int = 512
    
    def __post_init__(self):
        """初始化后处理和验证"""
        # 转换为列表格式
        if not isinstance(self.output_column, list):
            self.output_column = [self.output_column]
            
        if self.output_prompt_column and not isinstance(self.output_prompt_column, list):
            self.output_prompt_column = [self.output_prompt_column]
            
        self._validate()
        logger.info(f"数据集配置创建成功: {self.input_path} -> {self.output_path}")
    
    def _validate(self):
        """验证配置参数"""
        # 基本验证
        if not self.input_path or not os.path.exists(self.input_path):
            raise FileNotFoundError(f"输入文件不存在: {self.input_path}")
        
        if not self.input_path.endswith('.jsonl'):
            raise ValueError("输入文件必须是.jsonl格式")
        
        if not self.input_columns or not self.output_column:
            raise ValueError("输入列和输出列不能为空")
        
        # 数值验证
        if self.batch_size <= 0 or self.num_responses <= 0 or self.max_concurrent_tasks <= 0:
            raise ValueError("batch_size、num_responses和max_concurrent_tasks必须大于0")
        
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows必须大于0或为None")
        
        # 创建输出目录
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 路径重复警告
        if os.path.abspath(self.input_path) == os.path.abspath(self.output_path):
            logger.warning("输入输出路径相同，将覆盖原文件")
    
    def __str__(self):
        """配置信息字符串"""
        file_size = os.path.getsize(self.input_path) / (1024 * 1024)
        return f"""DatasetConfig:
  输入: {self.input_path} ({file_size:.1f}MB)
  输出: {self.output_path}
  输入列: {self.input_columns}
  输出列: {self.output_column}
  输出Prompt列: {self.output_prompt_column or '无'}
  最大并发: {self.max_concurrent_tasks}"""
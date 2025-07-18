import os
import logging
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

class DatasetConfig:
    """数据集配置类
        Args:
            input_path: 输入数据集文件的路径，支持JSONL格式。
            output_path: 输出结果文件的保存路径。
            input_columns: 用作输入的数据列名列表，这些列的内容将传递给LLM。
            output_column: 输出结果保存的列名，支持单个或多个。
            output_prompt_column: 可选的输出prompt列名，用于保存该行数据的prompt。
                如果为None，则不保存prompt信息。
            batch_size: 批处理大小，默认为1000。控制每次处理的数据行数。
            max_rows: 最大处理行数限制。如果为None，则处理所有数据行。
            max_thread_num: 最大线程数，默认为512。控制并发处理的线程数量。
    """
    
    def __init__(
        self,
        input_path: str,
        output_path: str,
        input_columns: List[str],
        output_column: Union[str, List[str]],
        output_prompt_column: Optional[Union[str, List[str]]] = None,
        batch_size: int = 1000,
        max_rows: Optional[int] = None,
        max_thread_num: int = 512
    ):

        self.input_path = input_path
        self.output_path = output_path
        self.input_columns = input_columns
        self.output_column = output_column if isinstance(output_column, list) else [output_column]
        self.output_prompt_column = output_prompt_column  # 保持原始格式，由ChatLLM处理
        self.batch_size = batch_size
        self.max_rows = max_rows  # None表示处理全部文件，否则只处理前max_rows行
        self.max_thread_num = max_thread_num
        
        # 简化日志输出，只记录关键配置信息
        logger.info(f"数据集配置: {self.input_columns} -> {self.output_column}")
        if self.max_rows:
            logger.info(f"限制处理行数: {self.max_rows}")
        
        self._validate()
    
    def _validate(self):
        """验证配置参数"""
        logger.debug("开始验证配置参数")
        
        if not self.input_path:
            raise ValueError("输入文件路径不能为空")
            
        if not os.path.exists(self.input_path):
            logger.error(f"输入文件不存在: {self.input_path}")
            raise ValueError(f"输入文件不存在: {self.input_path}")
            
        if not self.input_path.lower().endswith('.jsonl'):
            logger.error(f"输入文件必须是.jsonl格式: {self.input_path}")
            raise ValueError(f"输入文件必须是.jsonl格式: {self.input_path}")
            
        if self.batch_size <= 0:
            logger.error(f"batch_size必须大于0，当前值: {self.batch_size}")
            raise ValueError("batch_size必须大于0")
            
        if not self.input_columns:
            logger.error("input_columns不能为空")
            raise ValueError("input_columns不能为空")
            
        if not self.output_column:
            logger.error("output_column不能为空")
            raise ValueError("output_column不能为空")
            
        # 警告：输入输出路径相同
        if self.input_path == self.output_path:
            logger.warning("输入和输出路径相同，将创建临时文件")
            
        logger.debug("配置参数验证通过")
    
    def __str__(self):
        """配置信息的字符串表示，用于调试"""
        return f"""DatasetConfig:
                输入文件: {self.input_path}
                输出文件: {self.output_path}
                输入列: {self.input_columns}
                输出列: {self.output_column}
                Prompt列: {self.output_prompt_column}
                批次大小: {self.batch_size}
                最大行数: {self.max_rows if self.max_rows else '无限制'}
                最大线程数: {self.max_thread_num}"""
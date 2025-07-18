import os
import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompt import all_prompt_dict
from tqdm import tqdm
from typing import Dict, List, Optional, Callable, Any, Union
from openai import OpenAI
from dataset_config import DatasetConfig

logger = logging.getLogger(__name__)


class ChatLLM:
    """聊天LLM处理类
    Args:
        llm_url: LLM服务的基础URL地址。
        prompt_key: 用于从all_prompt_dict中获取prompt模板的键名，支持单个或多个。
        response_processor: 用于格式化LLM响应的输出解析器，支持单个或多个。
        generate_config: LLM生成的配置信息
        dataset_config: 数据集配置信息。
        api_key: LLM服务的API密钥，如果是本地部署则不需要密钥
        grouped_mode: 是否开启分组模式
        grouped_output_columns: 分组模式下的输出列分组信息
    """
    
    def __init__(
        self,
        llm_url: str,
        prompt_key: Union[str, List[str]],
        response_processor: Union[Callable[[str], Any], List[Callable[[str], Any]], List[List[Callable[[str], Any]]]],
        generate_config: Dict,
        dataset_config: DatasetConfig,
        api_key: str = "test",
        grouped_mode: bool = False,
        grouped_output_columns: Optional[List[List[str]]] = None,
    ):
        """初始化ChatLLM实例
        
        Args:
            llm_url: LLM服务URL地址
            prompt_key: prompt模板键名，支持单个字符串或字符串列表
            response_processor: 输出解析器，支持单个函数、函数列表或分组函数列表
            generate_config: LLM生成配置参数
            dataset_config: 数据集配置对象
            api_key: API密钥，默认为"test"
            grouped_mode: 是否开启分组模式
            grouped_output_columns: 分组模式下的输出列分组信息
        """
        self.llm_url = llm_url
        self.prompt_keys = prompt_key if isinstance(prompt_key, list) else [prompt_key]
        self.dataset_config = dataset_config
        self.api_key = api_key
        self.generate_config = generate_config or {}
        
        # 分组模式相关属性
        self.grouped_mode = grouped_mode
        self.grouped_output_columns = grouped_output_columns
        
        if grouped_mode:
            # 分组模式：response_processor应该是List[List[Callable]]
            self.grouped_response_processors = response_processor
            logger.info(f"初始化ChatLLM（分组模式），URL: {llm_url}")
            logger.info(f"Prompt Keys: {self.prompt_keys}")
            logger.info(f"分组数量: {len(self.grouped_response_processors)}")
            for i, group in enumerate(self.grouped_response_processors):
                logger.info(f"第{i+1}组处理器: {[proc.__name__ for proc in group]}")
        else:
            # 原有模式
            self.response_processors = response_processor if isinstance(response_processor, list) else [response_processor]
            logger.info(f"初始化ChatLLM，URL: {llm_url}")
            logger.info(f"Prompt Keys: {self.prompt_keys}")
            logger.info(f"Response Processors: {[proc.__name__ for proc in self.response_processors]}")
        
        # 验证prompt_key
        for pk in self.prompt_keys:
            if pk not in all_prompt_dict:
                raise ValueError(f"prompt_key '{pk}' 不存在于all_prompt_dict中")
        
        # 初始化LLM客户端
        self.client = OpenAI(base_url=llm_url, api_key=api_key, max_retries=10)

    def _call_llm(self, prompt: str) -> Optional[str]:
        """调用LLM生成回答
        
        Args:
            prompt: 输入的prompt文本
            
        Returns:
            LLM生成的响应文本，失败时返回None
        """
        try:
            messages = [
                {"role": "system", "content": "你叫理想同学，你是一个有用的助手。"},
                {"role": "user", "content": prompt}
            ]
            completion = self.client.chat.completions.create(messages=messages, **self.generate_config)
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return None

    def _generate_responses(self, prompt: str, processor: Callable[[str], Any], max_retries: int = 5) -> List[Any]:
        """为单个prompt生成响应
        
        Args:
            prompt: 输入的prompt文本
            processor: 对应的响应处理函数
            max_retries: 每个响应的最大重试次数
            
        Returns:
            处理后的响应列表
        """
        responses = []
        for retry in range(max_retries):
            raw_response = self._call_llm(prompt)
            if not raw_response or raw_response == '<|wrong data|>':
                continue
                
            try:
                processed_response = processor(raw_response)
                if processed_response is not None:
                    responses.append(processed_response)
                    break
            except Exception as e:
                logger.debug(f"响应处理失败: {e}")
                continue
        return responses

    def process_entry(self, data_row: Dict) -> Dict:
        """处理单个数据条目 - 支持分组模式
        
        Args:
            data_row: 单行数据字典
            
        Returns:
            处理后的数据字典，包含生成的响应和prompt
        """
        try:
            if self.grouped_mode:
                # 分组模式处理
                for idx, prompt_key in enumerate(self.prompt_keys):
                    prompt_template, num_expected_vals = all_prompt_dict[prompt_key]
                    
                    # 提取输入字段
                    entry = {k: data_row[k] for k in self.dataset_config.input_columns if k in data_row}
                    
                    # 验证字段
                    if len(entry) != num_expected_vals or len(entry) != len(self.dataset_config.input_columns):
                        logger.error(f"字段验证失败，需要{num_expected_vals}个字段，实际{len(entry)}个")
                        continue
                    
                    # 格式化prompt
                    ordered_values = [entry[col] for col in self.dataset_config.input_columns]
                    prompt = prompt_template.format(*ordered_values)
                    
                    # 生成一次原始响应
                    raw_response = None
                    max_retries = 5
                    for retry in range(max_retries):
                        raw_response = self._call_llm(prompt)
                        if raw_response and raw_response != '<|wrong data|>':
                            break
                    
                    if not raw_response:
                        logger.warning("无法生成有效响应")
                        continue
                    
                    # 获取该prompt对应的处理器组和输出列组
                    processor_group = self.grouped_response_processors[idx]
                    output_column_group = self.grouped_output_columns[idx]
                    
                    # 对每个处理器和输出列进行处理
                    for processor, output_column in zip(processor_group, output_column_group):
                        try:
                            processed_response = processor(raw_response)
                            if processed_response is not None:
                                data_row[output_column] = processed_response
                        except Exception as e:
                            logger.debug(f"响应处理失败 (输出列{output_column}): {e}")
                            continue
                    
                    # 保存prompt（如果需要）
                    if (self.dataset_config.output_prompt_column and 
                        idx < len(self.dataset_config.output_prompt_column)):
                        data_row[self.dataset_config.output_prompt_column[idx]] = prompt
            
            elif len(self.prompt_keys) == 1:
                # 单个prompt，可能有多个输出列
                prompt_key = self.prompt_keys[0]
                prompt_template, num_expected_vals = all_prompt_dict[prompt_key]
                
                # 提取输入字段
                entry = {k: data_row[k] for k in self.dataset_config.input_columns if k in data_row}
                
                # 验证字段
                if len(entry) != num_expected_vals or len(entry) != len(self.dataset_config.input_columns):
                    logger.error(f"字段验证失败，需要{num_expected_vals}个字段，实际{len(entry)}个")
                    return data_row
                
                # 格式化prompt
                ordered_values = [entry[col] for col in self.dataset_config.input_columns]
                prompt = prompt_template.format(*ordered_values)
                
                # 生成一次原始响应
                raw_response = None
                max_retries = 5
                for retry in range(max_retries):
                    raw_response = self._call_llm(prompt)
                    if raw_response and raw_response != '<|wrong data|>':
                        break
                
                if not raw_response:
                    logger.warning("无法生成有效响应")
                    return data_row
                
                # 对每个输出列使用对应的处理器处理同一个原始响应
                for idx, output_column in enumerate(self.dataset_config.output_column):
                    processor = self.response_processors[idx]
                    
                    try:
                        processed_response = processor(raw_response)
                        if processed_response is not None:
                            data_row[output_column] = processed_response
                    except Exception as e:
                        logger.debug(f"响应处理失败 (输出列{output_column}): {e}")
                        continue
                
                # 保存prompt（如果需要）
                if self.dataset_config.output_prompt_column:
                    # 单个prompt_key时，OUTPUT_PROMPT_COLUMN只有1个，所有输出列共享同一个prompt
                    prompt_col = self.dataset_config.output_prompt_column[0]
                    data_row[prompt_col] = prompt
                            
            else:
                # 多个prompt，传统逻辑：每个prompt对应一个输出列
                for idx, prompt_key in enumerate(self.prompt_keys):
                    prompt_template, num_expected_vals = all_prompt_dict[prompt_key]
                    
                    # 提取输入字段
                    entry = {k: data_row[k] for k in self.dataset_config.input_columns if k in data_row}
                    
                    # 验证字段
                    if len(entry) != num_expected_vals or len(entry) != len(self.dataset_config.input_columns):
                        logger.error(f"字段验证失败，需要{num_expected_vals}个字段，实际{len(entry)}个")
                        continue
                    
                    # 格式化prompt
                    ordered_values = [entry[col] for col in self.dataset_config.input_columns]
                    prompt = prompt_template.format(*ordered_values)
                    
                    # 获取对应的输出解析器
                    processor = self.response_processors[idx]
                    
                    # 生成响应
                    responses = self._generate_responses(prompt, processor)
                    
                    # 保存结果
                    output_column = self.dataset_config.output_column[idx]
                    if responses:
                        data_row[output_column] = responses[0] if len(responses) == 1 else responses
                    
                    # 保存prompt（如果需要）
                    if (self.dataset_config.output_prompt_column and 
                        idx < len(self.dataset_config.output_prompt_column)):
                        data_row[self.dataset_config.output_prompt_column[idx]] = prompt
            
            return data_row
            
        except Exception as e:
            logger.error(f"处理条目失败: {e}", exc_info=True)
            return data_row

    def produce_data(self, data_rows: List[Dict], output_path: str, pbar: tqdm):
        """批量处理数据并直接写入文件 - 仿照第二段代码的produce_data结构
        
        Args:
            data_rows: 一批数据字典列表
            output_path: 输出文件路径
            pbar: 进度条对象
        """
        with ThreadPoolExecutor(max_workers=self.dataset_config.max_thread_num) as executor:
            futures = [executor.submit(self.process_entry, data_row) for data_row in data_rows]
            
            with open(output_path, "a", encoding="utf-8") as f:
                for future in as_completed(futures):
                    try:
                        pbar.update(1)
                        data_row = future.result()
                        if not data_row:
                            logger.warning('[ERR] 处理结果为空')
                            continue
                        
                        # 检查是否有生成的结果
                        has_results = False
                        for output_col in self.dataset_config.output_column:
                            if data_row.get(output_col):
                                has_results = True
                                break
                        
                        if not has_results:
                            logger.warning('[ERR] 未生成有效结果')
                            continue
                        
                        # 检查是否包含错误标记
                        if '<|wrong data|>' in str(data_row):
                            logger.warning('[ERR] 结果包含错误标记')
                            continue
                            
                        f.write(json.dumps(data_row, ensure_ascii=False) + "\n")
                        
                    except Exception as ex:
                        logger.error(f'[ERR] 处理批次失败: {ex}')
                        continue

    def load_jsonl(self, file_path: str, batch_size: int = 1000, max_rows: Optional[int] = None):
        """批次加载JSONL文件数据
        
        Args:
            file_path: JSONL文件路径
            batch_size: 每批次大小
            max_rows: 最大处理行数，None表示处理所有行
            
        Yields:
            每批次的数据字典列表
        """
        batch = []
        processed_rows = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 如果设置了max_rows且已达到限制，停止处理
                if max_rows is not None and processed_rows >= max_rows:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    batch.append(data)
                    processed_rows += 1
                    
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
                        
                except json.JSONDecodeError as e:
                    continue
                    
        if batch:  # 处理最后一批剩余数据
            yield batch

    def get_file_line_nums(self, file_path: str, max_rows: Optional[int] = None) -> int:
        """获取文件行数
        
        Args:
            file_path: 文件路径
            max_rows: 最大行数限制
            
        Returns:
            文件行数（考虑max_rows限制）
        """
        try:
            out = subprocess.getoutput("wc -l {}".format(file_path))
            all_nums = int(out.split()[0])
        except:
            # 如果wc命令失败，回退到Python方式
            with open(file_path, 'r', encoding='utf-8') as f:
                all_nums = sum(1 for _ in f)
        
        # 应用max_rows限制
        if max_rows is not None:
            all_nums = min(all_nums, max_rows)
        
        return all_nums

    def process_dataset(self):
        """处理整个数据集 - 仿照第二段代码的main_entry结构
        
        主要流程：
        1. 批次读取JSONL文件
        2. 批次处理并立即写入结果
        3. 使用追加模式写入
        4. 支持max_rows限制
        """
        config = self.dataset_config
        input_path = config.input_path
        output_path = config.output_path
        max_rows = getattr(config, 'max_rows', None)
        
        # 检查输入输出文件相同且设置了max_rows的情况
        if input_path == output_path and max_rows is not None:
            raise ValueError(
                "当输入输出文件相同时，不能设置max_rows限制，因为这会导致原文件中未处理的数据丢失。"
                "请使用不同的输出文件路径，或者将max_rows设置为None。"
            )
        
        # 只处理单个文件        
        if os.path.isfile(input_path) and input_path.lower().endswith('.jsonl'):
            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 获取文件总行数（考虑max_rows限制）
            in_all_nums = self.get_file_line_nums(input_path, max_rows)
            
            if max_rows is not None:
                logger.info(f'开始处理文件：{input_path}，限制处理行数：{max_rows}')
            else:
                logger.info(f'开始处理文件：{input_path}')
            
            # 如果输入输出是同一文件，使用临时文件
            if input_path == output_path:
                temp_output = output_path + '.tmp'
                actual_output = temp_output
            else:
                actual_output = output_path
                # 如果输出文件已存在，删除以避免重复追加
                if os.path.isfile(output_path):
                    os.remove(output_path)
            
            # 创建进度条
            pbar = tqdm(desc=f"proc->{os.path.basename(input_path)}", 
                       total=in_all_nums, ncols=150)
            
            try:
                # 批次处理文件（传递max_rows参数）
                for data_rows in self.load_jsonl(input_path, batch_size=config.batch_size, max_rows=max_rows):
                    if len(data_rows) == 0:
                        logger.warning('[ERR] JSON加载错误')
                        continue
                    self.produce_data(data_rows, actual_output, pbar)
            finally:
                pbar.close()
            
            # 如果使用了临时文件，最后替换原文件
            if input_path == output_path:
                os.replace(temp_output, output_path)
                
        else:
            raise ValueError("输入路径需要为jsonl文件")
        
        logger.info(f"处理完成: {output_path}")
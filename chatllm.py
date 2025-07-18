import json
import logging
import asyncio
import gc
from prompt import all_prompt_dict
from typing import Dict, List, Optional, Callable, Any
from openai import AsyncOpenAI
from dataset_config import DatasetConfig
import aiofiles
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)


class ChatLLM:
    """聊天LLM处理类"""
    
    def __init__(
        self,
        llm_url: str,
        prompt_keys: List[str],
        response_processor: Callable[[str], Any],
        generate_config: Dict,
        dataset_config: DatasetConfig,
        api_key: str = "test",
    ):
        self.llm_url = llm_url
        self.prompt_keys = prompt_keys
        self.response_processor = response_processor
        self.dataset_config = dataset_config
        self.generate_config = generate_config
        self.is_async_processor = asyncio.iscoroutinefunction(response_processor)
        
        # 关闭HTTP请求日志
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        logger.info(f"初始化ChatLLM: {llm_url}, Prompts: {prompt_keys}")
        
        self._validate_config()
        self.client = AsyncOpenAI(base_url=llm_url, api_key=api_key, max_retries=3, timeout=60.0)
        
        # 预缓存，提升性能
        self.input_columns_set = set(self.dataset_config.input_columns)

    def _validate_config(self):
        """验证配置"""
        missing_prompts = [key for key in self.prompt_keys if key not in all_prompt_dict]
        if missing_prompts:
            raise ValueError(f"Prompt不存在: {missing_prompts}")
        
        output_len, prompt_len = len(self.dataset_config.output_column), len(self.prompt_keys)
        if output_len not in (1, prompt_len):
            raise ValueError(f"输出列数量({output_len})必须为1或等于prompt数量({prompt_len})")

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """调用LLM"""
        try:
            messages = [
                {"role": "system", "content": "你叫理想同学，你是一个有用的助手。"},
                {"role": "user", "content": prompt}
            ]
            completion = await self.client.chat.completions.create(messages=messages, **self.generate_config)
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return None

    async def _process_response(self, raw_response: str) -> Any:
        """处理响应"""
        if self.is_async_processor:
            return await self.response_processor(raw_response)
        return await asyncio.get_event_loop().run_in_executor(None, self.response_processor, raw_response)

    async def _generate_responses(self, prompt: str) -> List[Any]:
        """生成响应"""
        responses = []
        for _ in range(self.dataset_config.num_responses):
            for retry in range(3):
                raw_response = await self._call_llm(prompt)
                if raw_response and raw_response != '<|wrong data|>':
                    processed = await self._process_response(raw_response)
                    if processed is not None:
                        responses.append(processed)
                        break
                if retry < 2:
                    await asyncio.sleep(0.5)
            else:
                responses.append(None)
        return responses

    async def _process_single_entry(self, data_row: Dict) -> Dict:
        """处理单条数据"""
        result_row = data_row.copy()
        
        # 快速字段检查
        if not self.input_columns_set.issubset(data_row.keys()):
            missing = self.input_columns_set - data_row.keys()
            logger.warning(f"缺少字段: {missing}")
            return result_row

        input_values = [data_row[col] for col in self.dataset_config.input_columns]
        all_responses, all_prompts = [], []
        
        # 处理每个prompt
        for prompt_key in self.prompt_keys:
            prompt_template, expected_args = all_prompt_dict[prompt_key]
            
            if len(input_values) != expected_args:
                logger.error(f'字段数量不匹配: 期望{expected_args}, 实际{len(input_values)}')
                all_responses.append(None)
                all_prompts.append("")
                continue
            
            prompt = prompt_template.format(*input_values)
            responses = await self._generate_responses(prompt)
            valid_responses = [r for r in responses if r is not None]
            
            all_prompts.append(prompt)
            all_responses.append(valid_responses[0] if len(valid_responses) == 1 else valid_responses or None)
        
        self._set_output(result_row, all_responses, all_prompts)
        return result_row

    def _set_output(self, data_row: Dict, responses: List, prompts: List):
        """设置输出结果"""
        # 设置响应
        output_cols = self.dataset_config.output_column
        if len(output_cols) == 1:
            data_row[output_cols[0]] = responses[0] if len(self.prompt_keys) == 1 else responses
        else:
            for i, col in enumerate(output_cols):
                data_row[col] = responses[i] if i < len(responses) else None
        
        # 设置prompt
        prompt_cols = self.dataset_config.output_prompt_column
        if prompt_cols:
            if len(prompt_cols) == 1:
                data_row[prompt_cols[0]] = prompts[0] if len(self.prompt_keys) == 1 else prompts
            else:
                for i, col in enumerate(prompt_cols):
                    data_row[col] = prompts[i] if i < len(prompts) else ""

    async def _load_data(self, file_path: str) -> List[Dict]:
        """一次性加载所有数据到内存"""
        data = []
        line_count = 0
        
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            async for line in f:
                if self.dataset_config.max_rows and line_count >= self.dataset_config.max_rows:
                    break
                
                if not line.strip():
                    continue
                
                try:
                    data.append(json.loads(line))
                    line_count += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"第{line_count+1}行JSON解析失败: {e}")
        
        logger.info(f"成功加载{len(data)}条数据到内存")
        return data

    async def _process_batch(self, batch_data: List[Dict], pbar, output_file):
        """处理一个批次的数据"""
        semaphore = asyncio.Semaphore(self.dataset_config.max_concurrent_tasks)
        
        async def process_task(data_row):
            async with semaphore:
                return await self._process_single_entry(data_row)
        
        # 并发处理当前批次
        tasks = [asyncio.create_task(process_task(data_row)) for data_row in batch_data]
        results = await asyncio.gather(*tasks)
        
        # 更新进度条
        pbar.update(len(results))
        
        # 写入结果
        write_lines = [json.dumps(result, ensure_ascii=False) for result in results]
        logger.info(f"开始写入{len(write_lines)}条数据")
        await output_file.write("\n".join(write_lines) + "\n")
        
        # 强制垃圾回收，释放内存
        del results, tasks, write_lines
        gc.collect()

    async def _process_dataset(self) -> int:
        """处理数据集核心逻辑"""
        # 加载所有数据
        all_data = await self._load_data(self.dataset_config.input_path)
        
        if not all_data:
            logger.warning("没有找到有效数据")
            return 0
        
        total_count = len(all_data)
        batch_size = self.dataset_config.batch_size
        
        logger.info(f"开始批量处理{total_count}条数据，batch_size={batch_size}")
        
        # 使用进度条
        with tqdm(total=total_count, desc="处理数据", unit="条") as pbar:
            async with aiofiles.open(self.dataset_config.output_path, "w", encoding="utf-8") as output_file:
                # 按批次处理数据
                for i in range(0, total_count, batch_size):
                    batch_data = all_data[i:i + batch_size]
                    await self._process_batch(batch_data, pbar, output_file)
        
        return total_count

    async def process_dataset(self):
        """处理数据集主函数"""
        logger.info(f"开始处理数据集:\n{self.dataset_config}")
        
        try:
            processed_count = await self._process_dataset()
            logger.info(f"处理完成，共处理{processed_count}条数据")
        finally:
            await self.client.close()
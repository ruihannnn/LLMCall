import os
import json
import pytest
from pathlib import Path
from dotenv import load_dotenv
import sys

# 添加项目根目录到路径，以便导入项目模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import init_chat_llm


class TestChatLLM:
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_env(self):
        """加载环境变量"""
        project_root = Path(__file__).parent.parent
        env_file = project_root / ".env.example"
        load_dotenv(env_file, override=True)
        
        # 设置测试专用的MAX_ROWS限制
        os.environ['MAX_ROWS'] = '5'
    
    def test_mode1_no_prompt(self, setup_env):
        """测试模式1：无prompt列输出"""
        # 设置测试专用环境变量
        os.environ.update({
            'PROMPT_KEY': 'test1',
            'OUTPUT_PROMPT_COLUMN': '',
            'OUTPUT_COLUMN': 'test_mode1_output_column_no_prompt',
            'RESPONSE_PROCESSOR': 'test_111_response_processor1'
        })
        
        # 初始化并运行
        chat_llm = init_chat_llm()
        chat_llm.process_dataset()
        
        # 验证输出
        with open(chat_llm.dataset_config.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                assert data.get('test_mode1_output_column_no_prompt') == '111'
    
    def test_mode1(self, setup_env):
        """测试模式1：有prompt列输出"""
        # 设置测试专用环境变量
        os.environ.update({
            'PROMPT_KEY': 'test1',
            'OUTPUT_PROMPT_COLUMN': 'test_mode1_prompt_column',
            'OUTPUT_COLUMN': 'test_mode1_output_column_with_prompt',
            'RESPONSE_PROCESSOR': 'test_111_response_processor1'
        })
        
        # 初始化并运行
        chat_llm = init_chat_llm()
        chat_llm.process_dataset()
        
        # 验证输出
        with open(chat_llm.dataset_config.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                assert data.get('test_mode1_output_column_with_prompt') == '111'
                assert data.get('test_mode1_prompt_column') is not None
                assert data.get('test_mode1_prompt_column').strip() != ''
    
    def test_mode2_no_prompt(self, setup_env):
        """测试模式2：单prompt多输出，无prompt列"""
        # 设置测试专用环境变量
        os.environ.update({
            'PROMPT_KEY': 'test1',
            'OUTPUT_PROMPT_COLUMN': '',
            'OUTPUT_COLUMN': 'test_mode2_output_column_no_prompt1,test_mode2_output_column_no_prompt2',
            'RESPONSE_PROCESSOR': 'test_111_response_processor1,test_111_response_processor2'
        })
        
        # 初始化并运行
        chat_llm = init_chat_llm()
        chat_llm.process_dataset()
        
        # 验证输出
        with open(chat_llm.dataset_config.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                assert data.get('test_mode2_output_column_no_prompt1') == '111'
                assert data.get('test_mode2_output_column_no_prompt2') == '222'
    
    def test_mode2(self, setup_env):
        """测试模式2：单prompt多输出，有prompt列"""
        # 设置测试专用环境变量
        os.environ.update({
            'PROMPT_KEY': 'test1',
            'OUTPUT_PROMPT_COLUMN': 'test_mode2_prompt_column',
            'OUTPUT_COLUMN': 'test_mode2_output_column_with_prompt1,test_mode2_output_column_with_prompt2',
            'RESPONSE_PROCESSOR': 'test_111_response_processor1,test_111_response_processor2'
        })
        
        # 初始化并运行
        chat_llm = init_chat_llm()
        chat_llm.process_dataset()
        
        # 验证输出
        with open(chat_llm.dataset_config.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                assert data.get('test_mode2_output_column_with_prompt1') == '111'
                assert data.get('test_mode2_output_column_with_prompt2') == '222'
                assert data.get('test_mode2_prompt_column') is not None
                assert data.get('test_mode2_prompt_column').strip() != ''
    
    def test_mode3_no_prompt(self, setup_env):
        """测试模式3：多prompt多输出一一对应，无prompt列"""
        # 设置测试专用环境变量
        os.environ.update({
            'PROMPT_KEY': 'test1,test2',
            'OUTPUT_PROMPT_COLUMN': '',
            'OUTPUT_COLUMN': 'test_mode3_output_column_no_prompt1,test_mode3_output_column_no_prompt2',
            'RESPONSE_PROCESSOR': 'test_111_response_processor1,test_111_response_processor2'
        })
        
        # 初始化并运行
        chat_llm = init_chat_llm()
        chat_llm.process_dataset()
        
        # 验证输出
        with open(chat_llm.dataset_config.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                assert data.get('test_mode3_output_column_no_prompt1') == '111'
                assert data.get('test_mode3_output_column_no_prompt2') == '222'
    
    def test_mode3(self, setup_env):
        """测试模式3：多prompt多输出一一对应，有prompt列"""
        # 设置测试专用环境变量
        os.environ.update({
            'PROMPT_KEY': 'test1,test2',
            'OUTPUT_PROMPT_COLUMN': 'test_mode3_prompt_column1,test_mode3_prompt_column2',
            'OUTPUT_COLUMN': 'test_mode3_output_column_with_prompt1,test_mode3_output_column_with_prompt2',
            'RESPONSE_PROCESSOR': 'test_111_response_processor1,test_111_response_processor2'
        })
        
        # 初始化并运行
        chat_llm = init_chat_llm()
        chat_llm.process_dataset()
        
        # 验证输出
        with open(chat_llm.dataset_config.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                assert data.get('test_mode3_output_column_with_prompt1') == '111'
                assert data.get('test_mode3_output_column_with_prompt2') == '222'
                assert data.get('test_mode3_prompt_column1') is not None
                assert data.get('test_mode3_prompt_column1').strip() != ''
                assert data.get('test_mode3_prompt_column2') is not None
                assert data.get('test_mode3_prompt_column2').strip() != ''
    
    def test_mode4_no_prompt(self, setup_env):
        """测试模式4：多prompt分组输出，无prompt列"""
        # 设置测试专用环境变量
        os.environ.update({
            'PROMPT_KEY': 'test1,test2',
            'OUTPUT_PROMPT_COLUMN': '',
            'OUTPUT_COLUMN': '[test_mode4_output_column_no_prompt1,test_mode4_output_column_no_prompt2],[test_mode4_output_column_no_prompt3]',
            'RESPONSE_PROCESSOR': '[test_111_response_processor1,test_111_response_processor2],[test_111_response_processor3]'
        })
        
        # 初始化并运行
        chat_llm = init_chat_llm()
        chat_llm.process_dataset()
        
        # 验证输出
        with open(chat_llm.dataset_config.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # 第一个prompt的两个输出
                assert data.get('test_mode4_output_column_no_prompt1') == '111'
                assert data.get('test_mode4_output_column_no_prompt2') == '222'
                # 第二个prompt的一个输出
                assert data.get('test_mode4_output_column_no_prompt3') == '333'
    
    def test_mode4(self, setup_env):
        """测试模式4：多prompt分组输出，有prompt列"""
        # 设置测试专用环境变量
        os.environ.update({
            'PROMPT_KEY': 'test1,test2',
            'OUTPUT_PROMPT_COLUMN': 'test_mode4_prompt_column1,test_mode4_prompt_column2',
            'OUTPUT_COLUMN': '[test_mode4_output_column_with_prompt1,test_mode4_output_column_with_prompt2],[test_mode4_output_column_with_prompt3]',
            'RESPONSE_PROCESSOR': '[test_111_response_processor1,test_111_response_processor2],[test_111_response_processor3]'
        })
        
        # 初始化并运行
        chat_llm = init_chat_llm()
        chat_llm.process_dataset()
        
        # 验证输出
        with open(chat_llm.dataset_config.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # 第一个prompt的两个输出
                assert data.get('test_mode4_output_column_with_prompt1') == '111'
                assert data.get('test_mode4_output_column_with_prompt2') == '222'
                # 第二个prompt的一个输出
                assert data.get('test_mode4_output_column_with_prompt3') == '333'
                # 验证prompt列
                assert data.get('test_mode4_prompt_column1') is not None
                assert data.get('test_mode4_prompt_column1').strip() != ''
                assert data.get('test_mode4_prompt_column2') is not None
                assert data.get('test_mode4_prompt_column2').strip() != ''
import os
import pytest
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv


@pytest.fixture(scope="module")
def llm_client():
    """初始化LLM客户端"""
    # 获取项目根目录路径
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env.example"
    
    # 加载.env.example文件
    load_dotenv(env_file)
    
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("LLM_URL")
    )
    return client


def test_llm_connection(llm_client):
    """测试LLM连通性"""
    try:
        response = llm_client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "deepseek-chat"),
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=10
        )
        
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content.strip()) > 0
        
    except Exception as e:
        pytest.fail(f"LLM连接失败: {e}")


def test_llm_config_loaded():
    """测试配置是否正确加载"""
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env.example"
    load_dotenv(env_file)

    assert os.getenv("LLM_URL") is not None, "LLM_URL未配置"
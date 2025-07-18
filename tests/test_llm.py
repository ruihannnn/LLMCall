#!/usr/bin/env python3
"""
简单的LLM连通性测试 - pytest版本

运行方式：
    pytest tests/test_llm.py -v
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI


def test_llm_connectivity():
    """测试LLM连通性"""
    
    async def run_test():
        # 加载.env配置
        env_path = Path(__file__).parent.parent / '.env'
        assert env_path.exists(), f"未找到配置文件: {env_path}"
        load_dotenv(env_path)
        
        # 读取配置
        llm_url = os.getenv('LLM_URL')
        api_key = os.getenv('API_KEY', 'test')
        model_name = os.getenv('MODEL_NAME', 'qwen')
        
        assert llm_url, "缺少LLM_URL配置"
        
        # 测试连接
        client = AsyncOpenAI(base_url=llm_url, api_key=api_key, timeout=30.0)
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=100,
                temperature=0.1
            )
            
            assert response.choices, "收到空回复"
            reply = response.choices[0].message.content.strip()
            assert reply, "回复内容为空"
            
            print(f"✅ LLM回复: {reply}")
            
        finally:
            await client.aclose()
    
    # 运行异步测试
    asyncio.run(run_test())
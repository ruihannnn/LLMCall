
# LLM数据蒸馏工具 v2

高效的大语言模型批量数据处理工具，支持多prompt模板和并发处理。

## 功能特性

- 🚀 **并发处理**：多线程并行，显著提升处理速度
- 📝 **多prompt支持**：同时使用多个prompt模板处理数据
- 🔧 **灵活响应处理**：支持多种后处理函数

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd llm_distill_tool_v2

# 安装依赖
pip install -r requirements.txt
```

### 2. 模型部署

在LPAI平台上部署LLM模型：

```bash
bash model_deployment/lizrun_lpai_sglang_api_qwen3_32b.sh
```

### 3. 获取模型服务地址

```bash
# 查看运行状态
lizrun lpai list

# 获取具体运行信息和模型IP
lizrun lpai list -j <训练组件名称>
```

在`.env.example`中配置`LLM_URL`字段：
```bash
LLM_URL=http://<模型IP>:1688/v1/
```

### 4. 测试模型连通性

```bash
pytest tests/test_llm_connection.py
```

如果测试通过，说明模型部署成功且可正常访问。

### 5. 配置文件设置

在`.env.example`中配置以下关键参数：

#### 基础配置
```bash

# LLM服务配置
LLM_URL=http://10.80.11.133:1688/v1/

# 数据文件路径
INPUT_PATH=/path/to/input.jsonl
OUTPUT_PATH=/path/to/output.jsonl

# 数据列配置
INPUT_COLUMNS=session,query
OUTPUT_COLUMN=answer
# 多个输出列：数量必须与REPONSE_PROCESSOR相同，将多种后处理结果分别输出到不同的列中
OUTPUT_COLUMN=answer1,answer2

# 可选：保存prompt文本(如果不指定OUTPUT_PROMPT_COLUMN，则不保存prompt文本)
OUTPUT_PROMPT_COLUMN=prompt

# Prompt模板和响应后处理函数配置
PROMPT_KEY=prompt_template
RESPONSE_PROCESSOR=simple_response_processor
# 多个处理器：对同一个响应进行多种后处理
RESPONSE_PROCESSOR=simple_response_processor,no_think_response_processor
```

#### 高级配置（多prompt场景）
```bash
# 多个prompt模板
PROMPT_KEY=prompt_template1,prompt_template2

# 对应的输出列
OUTPUT_COLUMN=answer1,answer2

# 可选：保存prompt文本
OUTPUT_PROMPT_COLUMN=prompt1,prompt2

# 响应处理器配置
# 单个处理器：对所有prompt响应使用相同处理器
RESPONSE_PROCESSOR=no_think_response_processor

# 多个处理器：为每个prompt响应指定不同处理器
RESPONSE_PROCESSOR=no_think_response_processor,json_load_response_processor
```

### 6. 运行

```bash
python main.py
```

## 扩展开发

### 添加新的Prompt模板

1. 在`prompt.py`中添加新模板：
```python
new_prompt_template = """
你的新prompt模板...
输入1: {0}
输入2: {1}
"""

all_prompt_dict = {
    'new_template': [new_prompt_template, 2],  # 2表示需要2个输入参数
    # 其他模板...
}
```

2. 在配置文件中使用：
```bash
PROMPT_KEY=new_template
```

### 添加新的响应处理器

1. 在`response_processor.py`中添加新函数：
```python
def custom_response_processor(response: str) -> Any:
    """自定义响应处理器"""
    # 你的处理逻辑
    return processed_result
```
**函数必须接收单个字符串参数**

2. 在配置文件中使用：
```bash
RESPONSE_PROCESSOR=custom_response_processor
```


## 项目结构
```
llm_distill_tool_v2/
├── .env.example              # 配置文件
├── main.py                   # 主程序
├── chat_llm.py              # LLM处理类
├── prompt.py                # Prompt模板
├── response_processor.py    # 响应处理器
├── tests/                   # 测试文件
└── model_deployment/        # 模型部署脚本
```

## 注意事项

### 部署环境
- 代码必须部署在数据卷的个人文件夹下，避免权限问题
- 首次使用需安装lizrun工具，参考[公司文档](https://li.feishu.cn/wiki/P3yKwND9Wiz2ylkTDkHciuBDnIc)

### 网络配置
- 如遇pip超时，请配置公司pip源，参考[配置文档](https://li.feishu.cn/wiki/FIU9weHr4iDZIzkioCocWd4Yn4b)
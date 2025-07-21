# LLMCall

高效的大语言模型批量数据处理工具，支持多prompt模板和并发处理。

<div align="center">
  <p>
    <img src="https://img.shields.io/badge/特性-并发处理-blue">
    <img src="https://img.shields.io/badge/模式-四种工作模式-green">
    <img src="https://img.shields.io/badge/扩展-高可扩展性-orange">
  </p>
</div>

## 功能特性

- 🚀 **并发处理**：多线程并行执行，显著提升批量数据处理速度
- 📝 **四种工作模式**：从简单到复杂，覆盖不同数据处理场景需求
- 🔧 **可扩展性强**：支持新增prompt模板与输出后处理逻辑，适配多样化业务

## 四种工作模式

### 模式对比概览

| 模式 | 核心特点 |
|------|----------|
| 模式一 | 单prompt，单输出解析器，单输出列 |
| 模式二 | 单prompt，多输出解析器，多输出列 |
| 模式三 | 多prompt，多输出解析器和输出列（一一对应） |
| 模式四 | 多prompt，多输出解析器和输出列（分组模式） |

### 详细说明

<details>
<summary>模式一：单prompt，单输出解析器，单输出列</summary>

**适用场景**：基础数据处理，一对一转换  
- 一个prompt模板生成一个响应  
- 使用一个输出解析器进行后处理  
- 输出到一个列  

**优势**：配置简单，处理速度最快  

**配置示例**：
```bash
PROMPT_KEY=prompt1
RESPONSE_PROCESSOR=processor1
OUTPUT_COLUMN=answer
OUTPUT_PROMPT_COLUMN=prompt
```
</details>

<details>
<summary>模式二：单prompt，多输出解析器，多输出列</summary>

**适用场景**：需要对同一响应进行多种格式化处理  
- 一个prompt模板生成一个响应  
- 使用多个输出解析器同时处理同一个响应  
- 输出到多个列（如：原始文本、JSON格式、清理后文本）  

**优势**：节省LLM调用成本，一次生成多种格式  

**配置示例**：
```bash
PROMPT_KEY=prompt1
RESPONSE_PROCESSOR=processor1,processor2,processor3
OUTPUT_COLUMN=answer1,answer2,answer3
OUTPUT_PROMPT_COLUMN=prompt
```
</details>

<details>
<summary>模式三：多prompt，多输出解析器和输出列（一一对应）</summary>

**适用场景**：需要生成多种不同类型的内容  
- 多个prompt模板分别生成不同的响应  
- 每个响应使用对应的输出解析器  
- 输出到对应的列  

**优势**：并行处理不同类型的任务，提高处理效率  

**配置示例**：
```bash
PROMPT_KEY=prompt1,prompt2,prompt3
RESPONSE_PROCESSOR=processor1,processor2,processor3
OUTPUT_COLUMN=answer1,answer2,answer3
OUTPUT_PROMPT_COLUMN=prompt1,prompt2,prompt3
```
</details>

<details>
<summary>模式四：多prompt，多输出解析器和输出列（分组模式）</summary>

**适用场景**：复杂的数据处理需求，最大灵活性  
- 多个prompt模板分别生成不同的响应  
- 每个响应可以使用多个输出解析器进行不同的后处理  
- 每个prompt对应一组输出列  

**优势**：最大的配置灵活性，适合复杂业务场景  

**配置示例**：
```bash
PROMPT_KEY=prompt1,prompt2,prompt3
RESPONSE_PROCESSOR=[processor1,processor2],[processor3],[processor4,processor5,processor6]
OUTPUT_COLUMN=[answer1,answer2],[answer3],[answer4,answer5,answer6]
OUTPUT_PROMPT_COLUMN=prompt1,prompt2,prompt3
```

**分组模式说明**：  
- 第1个prompt (`prompt1`) 使用2个输出解析器，输出到2个列  
- 第2个prompt (`prompt2`) 使用1个输出解析器，输出到1个列  
- 第3个prompt (`prompt3`) 使用3个输出解析器，输出到3个列  
</details>

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd llm_distill_tool_v2

# 安装依赖
pip install -r requirements.txt
```



### 2. 配置文件设置

根据您的需求选择工作模式，编辑 `.env.example` 文件：

```bash
# 基础配置
LLM_URL=<>
API_KEY=<>
MODEL_NAME=<>
INPUT_PATH=/path/to/input.jsonl
OUTPUT_PATH=/path/to/output.jsonl
INPUT_COLUMNS=session,query

# 根据选择的模式配置以下参数
PROMPT_KEY=...
RESPONSE_PROCESSOR=...
OUTPUT_COLUMN=...
```

### 3. 运行

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
    'new_prompt': [new_prompt_template, 2],  # 2表示需要2个输入参数
    # 其他模板...
}
```
<details>
<summary>⚠️ 重要提醒：如果prompt中包含大括号，需要转义大括号避免格式化错误</summary>
例如，

```python
prompt_with_json = """
回答格式：
```json
{
    "字段": "值"
}
输入：{0}
"""
```
❌错误写法：会导致 KeyError

```python
prompt_with_json = """
回答格式：
json{{
    "字段": "值"
}}
输入：{0}
"""
```
✅ 正确写法 - 使用双大括号转义
</details>


2. 在 `.env.example` 中使用：
```bash
PROMPT_KEY=new_prompt
```

### 添加新的输出解析器

1. 在`response_processor.py`中添加新函数：
```python
def new_processor(response: str) -> Any:
    """自定义响应输出解析器"""
    # 你的处理逻辑
    return processed_result
```

2. 在 `.env.example` 中使用：
```bash
RESPONSE_PROCESSOR=new_processor
```

## 项目结构

```
llm_distill_tool_v2/
├── .env.example              # 配置文件（含四种模式配置说明）
├── main.py                   # 主程序
├── chat_llm.py              # LLM处理类
├── dataset_config.py        # 数据集配置
├── prompt.py                # Prompt模板
├── response_processor.py    # 输出解析器
└── tests/                   # 测试文件
```

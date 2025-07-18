# LLM数据蒸馏工具 v2

高效的大语言模型批量数据处理工具，支持多prompt模板和并发处理。

## 功能特性

- 🚀 **并发处理**：多线程并行，显著提升处理速度
- 📝 **四种工作模式**：从简单到复杂，满足不同数据处理需求
- 🔧 **可扩展性强**：新增prompt模板，输出后处理

## 四种工作模式

### 模式一：单prompt，单回调，单输出列
**适用场景**：基础数据处理，一对一转换
- 一个prompt模板生成一个响应
- 使用一个输出解析器进行后处理
- 输出到一个列
- **优势**：配置简单，处理速度最快

**配置示例**：
```bash
PROMPT_KEY=prompt1
RESPONSE_PROCESSOR=processor1
OUTPUT_COLUMN=answer
OUTPUT_PROMPT_COLUMN=prompt
```

### 模式二：单prompt，多回调，多输出列
**适用场景**：需要对同一响应进行多种格式化处理
- 一个prompt模板生成一个响应
- 使用多个输出解析器同时处理同一个响应
- 输出到多个列（如：原始文本、JSON格式、清理后文本）
- **优势**：节省LLM调用成本，一次生成多种格式

**配置示例**：
```bash
PROMPT_KEY=prompt1
RESPONSE_PROCESSOR=processor1,processor2,processor3
OUTPUT_COLUMN=answer1,answer2,answer3
OUTPUT_PROMPT_COLUMN=prompt
```

### 模式三：多prompt，多回调和输出列（一一对应）
**适用场景**：需要生成多种不同类型的内容
- 多个prompt模板分别生成不同的响应
- 每个响应使用对应的输出解析器
- 输出到对应的列
- **优势**：并行处理不同类型的任务，提高处理效率

**配置示例**：
```bash
PROMPT_KEY=prompt1,prompt2,prompt3
RESPONSE_PROCESSOR=processor1,processor2,processor3
OUTPUT_COLUMN=answer1,answer2,answer3
OUTPUT_PROMPT_COLUMN=prompt1,prompt2,prompt3
```

### 模式四：多prompt，多回调和输出列（分组模式）
**适用场景**：复杂的数据处理需求，最大灵活性
- 多个prompt模板分别生成不同的响应
- 每个响应可以使用多个输出解析器进行不同的后处理
- 每个prompt对应一组输出列
- **优势**：最大的配置灵活性，适合复杂业务场景

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

### 4. 测试模型连通性

```bash
pytest tests/test_llm_connection.py
```

### 5. 配置文件设置

根据您的需求选择工作模式，编辑 `.env.example` 文件：

```bash
# 基础配置
LLM_URL=http://<模型IP>:1688/v1/
INPUT_PATH=/path/to/input.jsonl
OUTPUT_PATH=/path/to/output.jsonl
INPUT_COLUMNS=session,query

# 根据选择的模式配置以下参数
PROMPT_KEY=...
RESPONSE_PROCESSOR=...
OUTPUT_COLUMN=...
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
    'new_prompt': [new_prompt_template, 2],  # 2表示需要2个输入参数
    # 其他模板...
}
```

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
├── tests/                   # 测试文件
└── model_deployment/        # 模型部署脚本
```

## 注意事项

### 部署环境
- 代码必须部署在数据卷的个人文件夹下，避免权限问题
- 首次使用需安装lizrun工具，参考[公司文档](https://li.feishu.cn/wiki/P3yKwND9Wiz2ylkTDkHciuBDnIc)

### 网络配置
- 如遇pip超时，请配置公司pip源，参考[配置文档](https://li.feishu.cn/wiki/FIU9weHr4iDZIzkioCocWd4Yn4b)
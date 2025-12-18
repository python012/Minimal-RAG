# 项目重构说明 - 从 Azure OpenAI 迁移到阿里云通义千问

## 📋 重构概述

本次重构将整个项目从 Azure OpenAI 迁移到阿里云通义千问（Qwen），所有注释和文档字符串都已改为中文。

## 🔄 主要变更

### 1. API 提供商变更

**之前**: Azure OpenAI
**现在**: 阿里云通义千问 (Qwen)

### 2. 文件修改清单

#### ✅ data_loader.py
- 将 `AzureOpenAI` 替换为 `OpenAI`
- 更新所有 API 配置变量名
- 修改 `get_embedding()` 方法使用阿里云 API
- 所有 docstring 和注释改为中文

**主要变更**:
```python
# 之前
from openai import AzureOpenAI
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")

# 现在
from openai import OpenAI
ALIYUN_MODEL_API_KEY = os.getenv("ALIYUN_MODEL_API_KEY", "")
ALIYUN_BASE_URL = os.getenv("ALIYUN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
```

#### ✅ rag_engine.py
- 同样从 `AzureOpenAI` 迁移到 `OpenAI`
- 更新 `get_embedding()` 方法
- 更新 `generate_answer()` 方法
- 更新 `query()` 方法中的 API 调用
- 所有 docstring 和注释改为中文

**主要变更**:
```python
# Embedding 调用
response = client.embeddings.create(
    model=ALIYUN_EMBED_MODEL,
    input=text,
    dimensions=ALIYUN_EMBED_DIM,  # 新增维度参数
    encoding_format="float"        # 新增格式参数
)

# Chat 调用
completion = client.chat.completions.create(
    model=ALIYUN_CHAT_MODEL,      # 之前是 AZURE_DEPLOYMENT_NAME_CHAT
    messages=[...]
)
```

#### ✅ app.py
- 更新侧边栏显示的模型配置信息
- 更新页脚的服务提供商信息
- 环境变量名称对应更新

**主要变更**:
```python
# 之前
gen_model = os.getenv("AZURE_DEPLOYMENT_NAME_CHAT", "gpt-4-mini")
st.info(f"**服务提供商：** Azure OpenAI")

# 现在
gen_model = os.getenv("ALIYUN_CHAT_MODEL", "qwen-plus")
st.info(f"**服务提供商：** 阿里云通义千问")
```

#### ✅ .env.example
- 完全重写配置示例
- 添加详细的中文注释
- 提供阿里云 API Key 获取链接
- 说明不同模型的使用场景

#### ✅ Shell 脚本
- 创建了 `clear.sh`, `load-json.sh`, `stats.sh`
- 作为 Windows bat 文件的 macOS/Linux 版本

#### ✅ README.md
- 全新的中文文档
- 详细的使用说明
- 配置参数说明
- 常见问题解答

### 3. 新增环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `ALIYUN_MODEL_API_KEY` | 阿里云 API 密钥 | 必填 |
| `ALIYUN_BASE_URL` | API 端点 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `ALIYUN_CHAT_MODEL` | 对话模型 | `qwen-plus` |
| `ALIYUN_EMBED_MODEL` | 嵌入模型 | `text-embedding-v4` |
| `ALIYUN_EMBED_DIM` | 向量维度 | `1024` |

### 4. 移除的环境变量

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_DEPLOYMENT_NAME_CHAT`
- `AZURE_DEPLOYMENT_NAME_EMBED`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_API_TIMEOUT` (改为 `ALIYUN_API_TIMEOUT`)

## 🚀 迁移步骤

如果您要从旧版本迁移到新版本，请按照以下步骤操作：

### 1. 更新环境变量

```bash
# 1. 备份旧的 .env 文件
cp .env .env.backup

# 2. 从示例创建新的 .env
cp .env.example .env

# 3. 编辑 .env，填入您的阿里云 API Key
# 必填：ALIYUN_MODEL_API_KEY
```

### 2. 清空并重建向量数据库

```bash
# 激活虚拟环境
source venv/bin/activate  # macOS/Linux

# 清空数据库
python data_loader.py --clear

# 重新导入文档
python data_loader.py --input ./data/recipes/ --pattern *.json
```

**重要**: 由于嵌入模型从 `text-embedding-3-small` (1536维) 变更为 `text-embedding-v4` (1024维)，向量维度不同，必须清空并重建数据库。

### 3. 验证配置

```bash
# 查看数据库统计
python data_loader.py --stats

# 启动 Web 应用
streamlit run app.py
```

## 💡 API 差异说明

### Embedding API

**Azure OpenAI**:
```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
)
```

**阿里云 Qwen**:
```python
response = client.embeddings.create(
    model="text-embedding-v4",
    input=text,
    dimensions=1024,           # 必须指定维度
    encoding_format="float"    # 指定格式
)
```

### Chat API

两者的 Chat API 几乎完全兼容（都遵循 OpenAI 标准），只需要更改：
- 基础 URL
- 模型名称

## 📊 性能对比

| 项目 | Azure OpenAI | 阿里云 Qwen |
|------|--------------|-------------|
| Embedding 维度 | 1536 | 1024 |
| 响应速度 | 较快 | 快 |
| 价格 | 较贵 | 较便宜 |
| 中文支持 | 好 | 优秀 |

## 🔍 测试建议

1. **基础功能测试**
   ```bash
   # 测试数据导入
   python data_loader.py --input ./data/sample.txt
   
   # 测试统计功能
   python data_loader.py --stats
   ```

2. **Web 界面测试**
   - 启动应用: `streamlit run app.py`
   - 测试基本问答
   - 检查来源显示
   - 验证流式输出

3. **API 连接测试**
   - 在 `aliyun-demo/` 目录查看示例文件
   - 运行 `generate-demo.py` 测试对话 API
   - 运行 `embedding-demo.py` 测试嵌入 API

## 🐛 常见问题

### Q: 代码没有错误但运行失败
A: 检查是否已配置 `.env` 文件，确保 `ALIYUN_MODEL_API_KEY` 已正确设置

### Q: 找不到文档但之前可以
A: 需要清空并重建向量数据库（维度不同）

### Q: 想使用其他 Qwen 模型
A: 修改 `.env` 中的 `ALIYUN_CHAT_MODEL`，可选值：
   - `qwen-turbo` - 快速
   - `qwen-plus` - 平衡（推荐）
   - `qwen-max` - 最强
   - `qwen-flash` - 极速

## 📝 代码审查

所有修改的代码已通过：
- ✅ Python 语法检查 (py_compile)
- ✅ 类型提示保持一致
- ✅ 中文注释和 docstring
- ✅ 与示例文件 API 调用方式一致

## 🎉 完成

重构已完成！项目现在完全使用阿里云通义千问 API，所有文档和注释都已中文化。

---

最后更新时间: 2025-12-18

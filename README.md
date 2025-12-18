# Minimal RAG - 基于阿里云通义千问的检索增强生成系统

这是一个简单而强大的 RAG（检索增强生成）知识问答系统，使用阿里云通义千问模型提供智能对话功能。

## 📸 界面预览

![RAG 系统运行界面](RAG-2025-12-18%2019.44.57.png)

## ✨ 功能特性

- 🤖 **智能问答**：基于您的知识库文档进行精准回答
- 📚 **文档管理**：支持多种格式文档导入（TXT、MD、JSON）
- 🔍 **语义检索**：使用向量相似度进行智能文档检索
- 💬 **流式对话**：实时流式输出，体验流畅
- 🎯 **来源追溯**：每个答案都可以查看引用来源
- 🌐 **Web 界面**：基于 Streamlit 的友好用户界面

## 🛠 技术栈

- **LLM**: 阿里云通义千问（Qwen）
- **向量数据库**: ChromaDB
- **Web 框架**: Streamlit
- **向量化**: 阿里云 text-embedding-v4

## 📋 前置要求

- Python 3.8+
- 阿里云 API Key（[获取地址](https://dashscope.console.aliyun.com/apiKey)）

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
source venv/bin/activate  # macOS/Linux
# 或
venv\\Scripts\\activate  # Windows

# 安装依赖包
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制配置示例文件
cp .env.example .env

# 编辑 .env 文件，填入您的阿里云 API Key
# 必填项：ALIYUN_MODEL_API_KEY
```

### 3. 导入文档

```bash
# 导入单个文件
python data_loader.py --input ./data/sample.txt

# 导入整个目录
python data_loader.py --input ./data/ --pattern *.txt *.md

# 导入 JSON 食谱文件
python data_loader.py --input ./data/recipes/ --pattern *.json

# 清空数据库后重新导入
python data_loader.py --input ./data/ --clear
```

### 4. 启动 Web 界面

```bash
streamlit run app.py
```

然后在浏览器中访问 `http://localhost:8501`

## 📝 使用说明

### 数据加载器 (data_loader.py)

```bash
# 查看数据库统计
python data_loader.py --stats

# 清空数据库
python data_loader.py --clear

# 指定数据源类型
python data_loader.py --input ./data/ --source git
```

### Web 界面功能

1. **侧边栏设置**
   - 重新加载引擎：添加新文档后刷新
   - 数据库统计：查看文档数量
   - 搜索设置：调整检索参数
   - 清除历史：重置对话

2. **主聊天界面**
   - 输入问题获取答案
   - 展开查看来源文档
   - 查看相关度分数

## ⚙️ 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `ALIYUN_MODEL_API_KEY` | 阿里云 API 密钥 | 必填 |
| `ALIYUN_BASE_URL` | API 端点 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `ALIYUN_CHAT_MODEL` | 对话模型 | `qwen-plus` |
| `ALIYUN_EMBED_MODEL` | 嵌入模型 | `text-embedding-v4` |
| `ALIYUN_EMBED_DIM` | 向量维度 | `1024` |
| `CHROMA_DB_PATH` | 数据库路径 | `./vector_db` |

### 模型选择

- **qwen-turbo**: 快速响应，适合简单对话
- **qwen-plus**: 平衡性能和成本（推荐）
- **qwen-max**: 最强性能，适合复杂任务
- **qwen-flash**: 极速响应，适合高并发场景

## 🗂 项目结构

```
Minimal-RAG/
├── app.py                 # Streamlit Web 应用
├── data_loader.py         # 数据加载工具
├── rag_engine.py          # RAG 核心引擎
├── requirements.txt       # Python 依赖
├── .env.example          # 配置示例
├── clear.sh              # 清空数据库脚本（macOS/Linux）
├── load-json.sh          # 加载 JSON 脚本（macOS/Linux）
├── stats.sh              # 查看统计脚本（macOS/Linux）
└── vector_db/            # 向量数据库目录（自动创建）
```

## 🔧 批处理脚本

### macOS/Linux

```bash
# 清空数据库
./clear.sh

# 加载 JSON 资料库（存入向量数据库 ChromaDB）
./load-json.sh

# 查看统计信息
./stats.sh
```

### Windows

```cmd
# 清空数据库
clear.bat

# 加载 JSON 资料库（存入向量数据库 ChromaDB）
load-json.bat

# 查看统计信息
stats.bat
```

## 💡 使用技巧

1. **文档准备**
   - 使用 UTF-8 编码保存文档
   - 将相关文档放在同一目录
   - JSON 文件支持中文字段

2. **提问技巧**
   - 问题越具体，答案越精确
   - 可以要求引用具体来源
   - 调整相关度阈值过滤无关结果

3. **性能优化**
   - 定期清理无用文档
   - 合理设置检索数量（n_results）
   - 根据需求选择合适的模型

## 🐛 常见问题

### Q: API 密钥错误
A: 检查 `.env` 文件中的 `ALIYUN_MODEL_API_KEY` 是否正确配置

### Q: 找不到相关文档
A: 降低相关度阈值（min_relevance）或检查文档是否正确导入

### Q: 导入文档失败
A: 确保文件使用 UTF-8 编码，检查文件路径是否正确

## 📚 相关资源

- [阿里云通义千问文档](https://help.aliyun.com/zh/model-studio/getting-started/models)
- [ChromaDB 文档](https://docs.trychroma.com/)
- [Streamlit 文档](https://docs.streamlit.io/)

## 📄 许可证

本项目采用 MIT 许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

Made with ❤️ using Aliyun Qwen

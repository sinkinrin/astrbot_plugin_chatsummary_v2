# AstrBot LLM 消息总结插件

一个基于 AstrBot 的群聊/私聊消息总结插件，支持按命令即时生成摘要，也可定时自动归档到 Markdown；支持展开合并转发的聊天记录参与总结。

## 功能概览
- 群内命令 `/消息总结 [条数] [关注话题]`：拉取最近 N 条群聊记录并用 LLM 总结，可选指定关注话题。
- 私聊命令 `/群总结 [条数] [群号] [关注话题]`：在私聊中指定群号获取摘要（需在该群内）。
- 转发总结 `/转发总结 [关注话题]`：对合并转发的聊天记录进行总结。
- 提示词、最大记录数、Token 上限可在 AstrBot WebUI 配置。
- 自动总结：支持定时任务、分群配置、按消息数/时间窗口分段、过滤最低群等级，并将结果写入 `auto_summaries/*.md` 归档。

## 安装与部署
1. 将本插件目录放入 AstrBot 的 `plugins/`，保持 `main.py`、`metadata.yaml`、`_conf_schema.json` 在根目录。
2. 重启 AstrBot 或在 WebUI 重新加载插件。
3. 在 WebUI 的“插件配置”中找到 `astrbot_plugin_chatsummary`，按需调整提示词和自动总结开关。

## 使用

### 基本用法
- 群聊：`/消息总结 120`（输出以“合并转发”形式发送）
- 私聊：`/群总结 80 123456789`（输出以“合并转发”形式发送）
- 转发：直接转发聊天记录 + 命令 `/转发总结`

### 话题关注功能（新功能）
在命令后添加可选的话题参数，让 LLM 重点关注某类话题进行总结：

```
/消息总结 50 技术讨论
/消息总结 100 会议安排和待办事项
/群总结 30 123456789 项目进度
/转发总结 技术方案讨论
```

当指定话题时，插件会指导 LLM 优先提取和总结与该话题相关的内容。如果聊天记录中没有相关内容，会明确说明。

### 其他特性
- 如果发送合并转发的聊天记录，插件会自动展开转发节点参与总结。
- 自动总结：在配置中开启 `auto_summary.enabled` 后，可选 `broadcast=true` 让定时总结也以合并转发发送到目标群；默认只写入 `auto_summaries/*.md`。
- 条数需为整数，且受 `limits.max_chat_records` 约束，超过时会自动截断。

## 配置项
- `prompt`：LLM 系统提示词，默认已提供结构化总结模板，可在 WebUI 编辑（默认要求简短回复且不使用 Markdown）。
- `limits.max_chat_records`：拉取的最大聊天记录条数（默认 200）。
- `limits.max_tokens`：LLM 输出上限，用于控制回复长度。
- `auto_summary`：
  - `enabled`：是否开启定时总结。
  - `interval_minutes`：轮询间隔（分钟）。
  - `target_groups`：要自动总结的群号列表。
  - `min_member_level`：仅统计高于该等级的成员消息。
  - `summary_mode`：`message_count` 按消息数分段；`time_window` 按时间窗口分段。
  - `message_chunk_size` / `time_window_minutes`：对应分段参数。
  - `broadcast`：开启后，自动总结也会以合并转发推送到群。

自动总结结果会保存到插件目录的 `auto_summaries/` 下，文件名包含群号和时间，便于归档追溯。

## 说明
- 插件仅支持 aiocqhttp 适配器。
- 若提示未配置可用的模型，请检查 AstrBot Provider 设置。
- 如需自定义格式，可直接修改提示词或调整 `limits/max_tokens` 以控制长度。

## 参考
- [AstrBot 官方文档](https://astrbot.app)

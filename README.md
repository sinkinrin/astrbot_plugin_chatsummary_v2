# AstrBot LLM 消息总结插件

基于 LLM 的 AstrBot 群聊总结插件，支持手动总结、自动定时归档、合并转发总结、图片渲染输出，以及基于规则的“重要消息提醒”。

当前版本：`v1.4.0`

## 功能概览

- 手动群聊总结：`/消息总结 [条数] [关注话题]`
- 私聊指定群总结：`/群总结 [条数] [群号] [关注话题]`
- 合并转发总结：`/转发总结 [关注话题]`
- 模型列表查看：`/总结模型列表`
- 自动总结：定时拉取群消息，按消息数或时间窗口分段，总结后写入 Markdown 归档，可选推送回群
- 图片渲染：可将总结内容渲染为图片发送
- 重要消息提醒：监听指定群的消息，命中正则规则或“与我相关”条件时向指定 QQ 私聊提醒

## 适用范围

- 插件核心能力依赖 aiocqhttp / NapCat 侧的群历史消息、合并转发和 QQ 私聊发送能力。
- 手动总结、自动总结、转发总结都以 QQ 群消息场景为主。
- “重要消息提醒”当前明确面向 QQ 私聊提醒，`target_user_id` 填写的是 QQ 号。

## 安装与部署

1. 将插件目录放入 AstrBot 的 `plugins/` 目录。
2. 确保 `main.py`、`metadata.yaml`、`_conf_schema.json` 位于插件根目录。
3. 重启 AstrBot，或在 WebUI 中重新加载插件。
4. 在 WebUI 的“插件配置”中找到 `astrbot_plugin_chatsummary_v2`，按需填写提示词、限制项、自动总结和提醒规则。

### 可选依赖：图片渲染

如果启用 `render_as_image: true`，需要安装 `html2image`，并保证系统可用 Chromium / Chrome / Edge。

```bash
pip install html2image
```

Linux / Docker 示例：

```bash
apt-get update && apt-get install -y chromium chromium-driver
```

说明：

- Windows 通常可自动检测已安装的 Chrome / Edge
- 图片模式渲染失败时，会自动降级为合并转发文本发送

## 命令说明

### 1. 群聊总结

```text
/消息总结 <条数> [关注话题]
```

示例：

```text
/消息总结 50
/消息总结 100 技术讨论
/消息总结 120 会议安排和待办事项
```

行为：

- 拉取当前群最近 N 条聊天记录
- 可选指定“关注话题”，让模型优先聚焦某一类内容
- 使用全局 `provider_id`（如有配置）

### 2. 私聊指定群总结

```text
/群总结 <条数> <群号> [关注话题]
```

示例：

```text
/群总结 30 123456789
/群总结 50 123456789 项目进度
```

行为：

- 在私聊中指定群号获取摘要
- 插件会先确认发送者仍在该群中
- 使用全局 `provider_id`（如有配置）

### 3. 合并转发总结

```text
/转发总结 [关注话题]
```

示例：

```text
/转发总结
/转发总结 技术方案
```

行为：

- 需要把命令和合并转发消息一起发送
- 插件会展开转发节点内容，再交给 LLM 总结
- 使用全局 `provider_id`（如有配置）

### 4. 查看可用模型

```text
/总结模型列表
```

行为：

- 列出当前 AstrBot 中可用的 Provider 信息
- 可用于查询 `provider_id` 应该填写什么值

## 输出方式

### 默认输出

- 手动总结默认以合并转发形式发送
- 若合并转发发送失败，会降级为普通文本消息

### 图片输出

当 `render_as_image = true` 时：

- 手动总结会优先渲染成图片发送
- 自动总结在推送到群时也会优先发送图片
- 图片发送失败时自动降级为合并转发

## 自动总结

自动总结会：

- 定时轮询目标群
- 拉取最近消息
- 根据 `summary_mode` 按消息数或时间窗口分段
- 只对新消息做总结，避免重复总结相同内容
- 将结果写入 AstrBot 数据目录中的 Markdown 文件
- 可选把总结再次推送回群

归档目录：

```text
plugins_data/astrbot_plugin_chatsummary_v2/auto_summaries/
```

### 自动总结相关配置

- `auto_summary.enabled`：是否开启自动总结
- `auto_summary.interval_minutes`：轮询间隔（分钟）
- `auto_summary.target_groups`：需要自动总结的群号列表
- `auto_summary.summary_mode`：
  - `message_count`：按消息条数分段
  - `time_window`：按时间窗口分段
- `auto_summary.message_chunk_size`：按消息数分段时，每段包含的消息数
- `auto_summary.time_window_minutes`：按时间窗口分段时的窗口大小
- `auto_summary.min_messages`：少于该阈值时跳过本轮总结
- `auto_summary.broadcast`：是否把自动总结重新发送回群
- `auto_summary.provider_id`：自动总结专用 Provider；留空则使用默认模型

## 重要消息提醒

这是 `v1.4.0` 新增功能，用于监听指定群消息，并向指定 QQ 私聊发送提醒。

### 能做什么

- 监听 `watch_groups` 中的群消息
- 命中任一正则规则时触发提醒
- 或在开启 `mention_me` 后，检测：
  - `@目标用户`
  - 回复目标用户的消息
  - 文本中出现配置的称呼别名

提醒内容当前分两部分：

- 一条摘要文本：群名、发送者、时间、命中规则
- 一条原消息的合并转发

### 重要提醒

- `watch_groups` 为空时不会监听任何群
- `target_user_id` 为空时不会发送提醒
- 该能力当前按 QQ 私聊目标设计，`target_user_id` 填写 QQ 号

### 相关配置

- `important_message_reminder.enabled`：总开关
- `important_message_reminder.watch_groups`：监听群号列表
- `important_message_reminder.target_user_id`：接收提醒的 QQ 号
- `important_message_reminder.rules`：正则提醒规则列表
  - `pattern`：Python 正则表达式
  - `alias`：规则别名，可为空
- `important_message_reminder.mention_me.enabled`：开启“与我相关”提醒
- `important_message_reminder.mention_me.aliases`：我的称呼列表

## 配置项总览

### 基础配置

- `prompt`：LLM 系统提示词，支持 `\n` 分行
- `render_as_image`：是否把总结渲染成图片发送
- `provider_id`：手动总结命令使用的全局 Provider ID

### 限制项

- `limits.max_chat_records`：单次拉取的最大聊天记录数
- `limits.max_input_chars`：发给 LLM 的最大上下文字符数
- `limits.max_tokens`：LLM 输出 token 上限

### 自动总结配置

见上文“自动总结”章节。

### 重要消息提醒配置

见上文“重要消息提醒”章节。

## 使用建议

- 如果模型输出过长，优先收紧 `limits.max_input_chars` 和 `limits.max_tokens`
- 如果你有多个模型，可先执行 `/总结模型列表`，再配置 `provider_id`
- 如果只想归档不想打扰群聊，可将 `auto_summary.broadcast` 关闭
- 如果启用了图片模式，建议先确认服务器上 Chromium 可用
- 如果启用了重要消息提醒，务必显式配置 `watch_groups`，避免误以为默认监听全部群

## 说明

- 插件会按配置文件变更自动 reload，大多数配置修改后无需重启 AstrBot
- 若未配置可用模型，请先检查 AstrBot Provider 设置
- 图片渲染和提醒能力都依赖当前运行环境对 QQ / aiocqhttp 相关能力的支持
- 本项目参考了 laopanmemz 的消息总结插件，但因需求差异较大，采用了独立重构的 v2 实现

## 参考

- [AstrBot 官方文档](https://astrbot.app)
- [参考插件 astrbot_plugin_chatsummary](https://github.com/laopanmemz/astrbot_plugin_chatsummary)

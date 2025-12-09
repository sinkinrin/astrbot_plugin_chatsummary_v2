import asyncio
import contextlib
import copy
import hashlib
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.api import logger  # 使用 AstrBot 提供的 logger
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_platform_adapter import (
    AiocqhttpAdapter,
)
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

_TYPE_DEFAULTS = {
    "string": "",
    "text": "",
    "int": 0,
    "float": 0.0,
    "bool": False,
    "list": [],
    "object": {},
}

# 图片渲染配置
_IMAGE_CONFIG = {
    "width": 600,           # 图片宽度
    "padding": 30,          # 内边距
    "title_font_size": 24,  # 标题字号
    "content_font_size": 18,# 正文字号
    "line_spacing": 10,     # 行间距
    "bg_color": (245, 245, 250),      # 背景色 (浅灰紫)
    "card_color": (255, 255, 255),    # 卡片背景色 (白色)
    "title_color": (51, 51, 51),      # 标题颜色 (深灰)
    "content_color": (68, 68, 68),    # 正文颜色 (灰色)
    "accent_color": (102, 126, 234),  # 强调色 (紫色)
    "footer_color": (170, 170, 170),  # 页脚颜色 (浅灰)
}


@register(
    "astrbot_plugin_chatsummary_v2",
    "sinkinrin",
    "基于 LLM 的群聊总结与定时归档插件，支持图片渲染和指定关注话题",
    "1.2.1",
)
class ChatSummary(Star):
    CONFIG_NAMESPACE = "astrbot_plugin_chatsummary_v2"
    CONFIG_FILE = f"{CONFIG_NAMESPACE}_config.json"

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context, config)
        self._config_proxy = config or {}
        self._config_path = self._resolve_config_path()
        self._schema_defaults = self._load_schema_defaults()
        self.settings: dict[str, Any] = {}
        self.prompt: str = ""
        self._config_mtime: float | None = None
        self._reload_settings(force=True)

        astrbot_conf = self.context.get_config()
        wake = astrbot_conf.get("wake_prefix") or astrbot_conf.get("provider_settings", {}).get("wake_prefix", [])
        if isinstance(wake, str):
            wake = [wake]
        self.wake_prefix: List[str] = [str(prefix).strip() for prefix in wake or [] if str(prefix).strip()]

        self._aiocqhttp_client = None
        self._summary_storage = Path(__file__).with_name("auto_summaries")
        self._summary_storage.mkdir(parents=True, exist_ok=True)
        self._auto_summary_lock = asyncio.Lock()
        self._auto_summary_task: asyncio.Task | None = None
        # 实例唯一标识，用于调试多实例问题
        self._instance_id = str(uuid.uuid4())[:8]
        # 记录每个群上次总结的最后一条消息时间，用于判断是否有新消息
        self._last_summary_time: Dict[str | int, datetime] = {}
        # 记录上次总结的消息内容哈希，避免重复总结相同内容
        self._last_summary_hash: Dict[str | int, str] = {}
        
        # 直接在 __init__ 中启动后台任务（官方推荐方式）
        # 任务内部会等待平台适配器就绪
        self._auto_summary_task = asyncio.create_task(self._auto_summary_loop())
        logger.info("ChatSummary[%s] 初始化完成，配置路径：%s，自动总结任务已启动", self._instance_id, self._config_path)

    # ------------------------------------------------------------------
    # AstrBot 生命周期钩子（仅用于日志记录，不再重复启动任务）
    # ------------------------------------------------------------------
    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        """当 AstrBot 完全初始化后的回调"""
        logger.info("ChatSummary: on_astrbot_loaded 钩子被触发")
        # 任务已在 __init__ 中启动，这里仅记录状态
        if self._auto_summary_task:
            if self._auto_summary_task.done():
                logger.warning("Auto summary task 已结束，可能发生了异常")
            else:
                logger.debug("Auto summary task 正在运行中")

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _resolve_config_path(self) -> Path:
        path = getattr(self._config_proxy, "config_path", None)
        if path:
            return Path(path)
        return Path(get_astrbot_data_path()) / "config" / self.CONFIG_FILE

    def _load_schema_defaults(self) -> dict:
        schema_path = Path(__file__).with_name("_conf_schema.json")
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning("Schema file %s not found, fallback to empty defaults.", schema_path)
            return {}
        except json.JSONDecodeError as exc:
            logger.error("Schema file %s is invalid: %s", schema_path, exc)
            return {}
        return self._schema_to_defaults(schema)

    def _schema_to_defaults(self, schema: dict) -> dict:
        defaults: dict[str, Any] = {}
        for key, meta in schema.items():
            meta_type = meta.get("type", "string")
            if meta_type == "object":
                defaults[key] = self._schema_to_defaults(meta.get("items", {}))
            elif meta_type == "list":
                default_value = meta.get("default")
                if default_value is None:
                    default_value = []
                defaults[key] = copy.deepcopy(default_value)
            else:
                default_value = meta.get("default")
                if default_value is None:
                    default_value = copy.deepcopy(_TYPE_DEFAULTS.get(meta_type, ""))
                elif isinstance(default_value, (list, dict)):
                    default_value = copy.deepcopy(default_value)
                defaults[key] = default_value
        return defaults

    def _read_config_file(self) -> dict:
        try:
            with self._config_path.open(encoding="utf-8-sig") as fp:
                return json.load(fp)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as exc:
            logger.error("配置文件 %s 损坏：%s，已回退至默认值", self._config_path, exc)
            return {}

    def _merge_defaults(self, overrides: dict) -> dict:
        merged = copy.deepcopy(self._schema_defaults)
        for key, value in overrides.items():
            if isinstance(merged.get(key), dict) and isinstance(value, dict):
                merged[key] = self._merge_nested_dict(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _merge_nested_dict(self, base: dict, overrides: dict) -> dict:
        result = copy.deepcopy(base)
        for key, value in overrides.items():
            if isinstance(result.get(key), dict) and isinstance(value, dict):
                result[key] = self._merge_nested_dict(result[key], value)
            else:
                result[key] = value
        return result

    def _reload_settings(self, *, force: bool = False) -> dict:
        try:
            mtime = self._config_path.stat().st_mtime
        except FileNotFoundError:
            mtime = None
        if force or mtime != self._config_mtime:
            self._config_mtime = mtime
            loaded = self._read_config_file()
            merged = self._merge_defaults(loaded)
            prompt_value = merged.get("prompt", "")
            if isinstance(prompt_value, str):
                self.prompt = prompt_value.replace("\\n", "\n")
            else:
                self.prompt = str(prompt_value)
            self.settings = merged
        return self.settings

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------
    async def _collect_group_messages(
        self,
        client,
        group_id: str | int,
        *,
        count: int,
    ) -> Tuple[str, List[dict]]:
        payloads = {
            "group_id": self._normalize_group_id(group_id),
            "message_seq": 0,
            "count": max(1, count),
            # 注意：部分 CQHTTP 实现不支持此参数，消息顺序取决于实现
        }
        history = await client.api.call_action("get_group_msg_history", **payloads)
        login_info = await client.api.call_action("get_login_info")
        my_id = str(login_info.get("user_id", ""))
        messages = history.get("messages", []) or []

        chat_lines: List[str] = []
        structured: List[dict] = []
        for msg in messages:
            sender = msg.get("sender", {}) or {}
            sender_id = str(sender.get("user_id", ""))
            if sender_id == my_id:
                continue

            nickname = sender.get("card") or sender.get("nickname") or "未知用户"
            msg_time = datetime.fromtimestamp(msg.get("time", 0))
            message_text = await self._flatten_message_parts(msg.get("message", []) or [], client)

            if not message_text:
                continue
            if any(message_text.startswith(prefix) for prefix in self.wake_prefix):
                continue

            line = f"[{msg_time}]「{nickname}」: {message_text}"
            chat_lines.append(line)
            structured.append(
                {
                    "time": msg_time,
                    "nickname": nickname,
                    "user_id": sender_id,
                    "text": message_text,
                },
            )

        return "\n".join(chat_lines), structured

    async def _flatten_message_parts(self, parts: Sequence[dict], client=None) -> str:
        buffers: List[str] = []
        for part in parts:
            p_type = part.get("type")
            data = part.get("data", {}) or {}
            if p_type == "text":
                buffers.append(data.get("text", "").strip())
            elif p_type == "json":
                snippet = self._extract_json_desc(data.get("data"))
                if snippet:
                    buffers.append(f"[卡片]{snippet}")
            elif p_type == "face":
                buffers.append("[表情]")
            elif p_type == "image":
                url = data.get("url") or data.get("file")
                if url:
                    buffers.append(f"[图片]{url}")
                else:
                    buffers.append("[图片]")
            elif p_type == "reply":
                buffers.append("[回复消息]")
            elif p_type == "record":
                buffers.append("[语音]")
            elif p_type == "video":
                buffers.append("[视频]")
            elif p_type == "forward":
                forward_id = data.get("id") or data.get("resid")
                forward_text = ""
                if client and forward_id:
                    forward_text = await self._fetch_forward_messages(client, forward_id)
                buffers.append(forward_text or "[合并转发]")
        return " ".join(token for token in buffers if token).strip()

    def _extract_json_desc(self, raw: Any) -> str:
        if not raw:
            return ""
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return ""
        return (
            parsed.get("meta", {})
            .get("news", {})
            .get("desc", "")
            .strip()
        )

    async def _fetch_forward_messages(self, client, forward_id: str) -> str:
        """Expand forward (合并转发) messages into readable lines."""
        try:
            resp = await client.api.call_action("get_forward_msg", id=forward_id)
        except Exception as exc:
            logger.warning("获取转发记录失败: %s", exc)
            return ""

        nodes = resp.get("messages") or resp.get("data", {}).get("messages") or []
        lines: List[str] = []
        for node in nodes:
            sender = node.get("sender", {}) or {}
            nickname = sender.get("card") or sender.get("nickname") or "未知用户"
            msg_time = datetime.fromtimestamp(node.get("time", 0))
            content = node.get("content") or node.get("message") or []
            if not isinstance(content, list):
                continue
            text = await self._flatten_message_parts(content, client)
            if not text:
                continue
            lines.append(f"[{msg_time}]「{nickname}」: {text}")
        return "\n".join(lines)

    def _normalize_group_id(self, group_id: str | int) -> int | str:
        try:
            return int(group_id)
        except (TypeError, ValueError):
            return str(group_id)

    def _split_text_by_sections(self, text: str, max_len: int = 2000) -> List[str]:
        """按照内容的大点/段落智能分割文本。
        
        分割策略（优先级从高到低）：
        1. 按数字编号开头的大点分割（如1. 2. 3. 或 一、二、三、）
        2. 按【】标题分割
        3. 按双换行分割
        4. 如果单个分段超过 max_len，再按字符切分
        
        Args:
            text: 要分割的文本
            max_len: 每个分段的最大字符数
        
        Returns:
            分割后的文本列表
        """
        text = (text or "").strip()
        if not text:
            return []
        
        # 策略 1: 尝试按数字编号大点分割 (1. 2. 3. 或 一、二、三、 或 （1）（2）)
        # 匹配行首的编号模式
        section_pattern = re.compile(
            r'^(?=(?:\d+[.\u3001\uff0e]|[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+[\u3001\uff0e.]|[\uff08\(]\d+[\uff09\)]|[\u3010\u300a].+?[\u3011\u300b]))',
            re.MULTILINE
        )
        
        sections = self._split_by_pattern(text, section_pattern)
        if len(sections) > 1:
            return self._ensure_max_len(sections, max_len)
        
        # 策略 2: 按【】标题分割
        bracket_pattern = re.compile(r'^(?=\u3010)', re.MULTILINE)
        sections = self._split_by_pattern(text, bracket_pattern)
        if len(sections) > 1:
            return self._ensure_max_len(sections, max_len)
        
        # 策略 3: 按双换行分割
        sections = [s.strip() for s in re.split(r'\n\s*\n', text) if s.strip()]
        if len(sections) > 1:
            return self._ensure_max_len(sections, max_len)
        
        # 策略 4: 按单换行分割（适用于列表形式）
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) > 1:
            # 尝试合并短行，避免过多消息
            merged = self._merge_short_lines(lines, max_len // 2)
            return self._ensure_max_len(merged, max_len)
        
        # 最后回退：按字符长度切分
        return self._split_by_length(text, max_len)
    
    def _split_by_pattern(self, text: str, pattern: re.Pattern) -> List[str]:
        """根据正则模式分割文本。"""
        positions = [m.start() for m in pattern.finditer(text)]
        if not positions:
            return [text.strip()] if text.strip() else []
        
        # 确保从头开始
        if positions[0] != 0:
            positions.insert(0, 0)
        
        sections: List[str] = []
        for i, start in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(text)
            section = text[start:end].strip()
            if section:
                sections.append(section)
        return sections
    
    def _merge_short_lines(self, lines: List[str], target_len: int) -> List[str]:
        """合并较短的行，避免每行一条消息。"""
        if not lines:
            return []
        
        merged: List[str] = []
        current = lines[0]
        
        for line in lines[1:]:
            # 如果当前行以编号开头，可能是新的大点
            is_new_point = bool(re.match(
                r'^(?:\d+[.\u3001]|[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+[\u3001.]|[\uff08\(]\d+[\uff09\)]|[\u3010\u300a])',
                line
            ))
            
            if is_new_point or len(current) + len(line) + 1 > target_len:
                if current.strip():
                    merged.append(current.strip())
                current = line
            else:
                current = current + '\n' + line
        
        if current.strip():
            merged.append(current.strip())
        return merged
    
    def _ensure_max_len(self, sections: List[str], max_len: int) -> List[str]:
        """确保每个分段不超过最大长度，超过则再次切分。"""
        result: List[str] = []
        for section in sections:
            if len(section) <= max_len:
                result.append(section)
            else:
                result.extend(self._split_by_length(section, max_len))
        return result
    
    def _split_by_length(self, text: str, max_len: int) -> List[str]:
        """按字符长度切分，尽量在换行符处断开。"""
        text = text.strip()
        if not text:
            return []
        if len(text) <= max_len:
            return [text]
        
        chunks: List[str] = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break
            
            # 尝试在 max_len 附近找换行符
            cut_pos = text.rfind('\n', 0, max_len)
            if cut_pos == -1 or cut_pos < max_len // 2:
                # 没找到合适的换行符，直接截断
                cut_pos = max_len
            
            chunks.append(text[:cut_pos].strip())
            text = text[cut_pos:].strip()
        
        return chunks

    async def _send_group_forward(
        self,
        client,
        group_id: str | int,
        title: str,
        summary_text: str,
        outline_text: str = "",
    ) -> bool:
        """Send merged forward message to a group with summary + outline.
        
        发送策略：
        1. 尝试发送带 message segment 的合并转发
        2. 失败则尝试纯文本 content 的合并转发
        3. 再失败则降级为普通群消息
        
        Returns:
            bool: 是否成功发送
        """
        try:
            login_info = await client.api.call_action("get_login_info")
            self_id = str(login_info.get("user_id", ""))
        except Exception as exc:
            logger.error("获取 bot 信息失败：%s", exc)
            return False
        
        nodes = self._build_forward_nodes(
            title=title, 
            self_id=self_id, 
            summary_text=summary_text, 
            outline_text=outline_text
        )
        
        if not nodes:
            logger.warning("构建转发节点为空，跳过发送")
            return False
        
        normalized_group_id = self._normalize_group_id(group_id)
        logger.debug("准备发送合并转发到群 %s，节点数=%d", group_id, len(nodes))

        # 策略 1: 尝试发送带 message segment 的合并转发
        try:
            resp = await client.api.call_action(
                "send_group_forward_msg",
                group_id=normalized_group_id,
                messages=nodes,
            )
            if isinstance(resp, dict) and resp.get("status") == "failed":
                raise RuntimeError(f"API 返回失败: {resp}")
            logger.info("合并转发发送成功（message segment 模式）")
            return True
        except Exception as exc:
            logger.warning("发送合并转发失败（message segment 模式）：%s", exc)

        # 策略 2: 尝试纯文本 content 的合并转发
        plain_nodes = self._build_forward_nodes(
            title=title,
            self_id=self_id,
            summary_text=summary_text,
            outline_text=outline_text,
            as_plain=True,
        )
        try:
            resp = await client.api.call_action(
                "send_group_forward_msg",
                group_id=normalized_group_id,
                messages=plain_nodes,
            )
            if isinstance(resp, dict) and resp.get("status") == "failed":
                raise RuntimeError(f"API 返回失败: {resp}")
            logger.info("合并转发发送成功（纯文本模式）")
            return True
        except Exception as exc:
            logger.warning("发送合并转发失败（纯文本模式）：%s", exc)

        # 策略 3: 降级为普通群消息
        logger.warning("合并转发均失败，降级为普通文本消息")
        text = f"📝 {title}\n\n{summary_text.strip()}"
        if outline_text:
            text += f"\n\n📌 聊天要点\n{outline_text.strip()}"
        
        try:
            await client.api.call_action(
                "send_group_msg",
                group_id=normalized_group_id,
                message=text[:4000],
            )
            logger.info("已降级为普通文本消息发送")
            return True
        except Exception as exc:
            logger.error("普通文本消息发送也失败：%s", exc)
            return False

    def _extract_forward_ids_from_event(self, event: AstrMessageEvent) -> List[str]:
        """Try to grab forward (合并转发) ids from incoming message payload."""
        forward_ids: List[str] = []
        candidates: List[Sequence[dict] | None] = []

        raw_event = getattr(event, "raw_event", None)
        if isinstance(raw_event, dict):
            candidates.append(raw_event.get("message") or raw_event.get("original_message"))

        message_attr = getattr(event, "message", None)
        if isinstance(message_attr, list):
            candidates.append(message_attr)

        for parts in candidates:
            if not parts:
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if part.get("type") != "forward":
                    continue
                data = part.get("data", {}) or {}
                forward_id = data.get("id") or data.get("resid")
                if forward_id:
                    forward_ids.append(str(forward_id))
        return forward_ids

    def _build_forward_nodes(
        self,
        *,
        title: str,
        self_id: str,
        summary_text: str,
        outline_text: str | None = None,
        as_plain: bool = False,
    ) -> List[dict]:
        """Build forward nodes with cqhttp message segments or plain string content.
        
        每个大点/段落作为一条单独的消息，以合并转发的形式发送。
        """
        nodes: List[dict] = []

        def _node(name: str, chunk: str) -> dict:
            chunk = chunk.strip()
            if as_plain:
                return {"type": "node", "data": {"name": name, "uin": self_id, "content": chunk}}
            return {
                "type": "node",
                "data": {
                    "name": name,
                    "uin": self_id,
                    "content": [
                        {"type": "text", "data": {"text": chunk}},
                    ],
                },
            }

        # 按大点分割总结内容，每个大点一条消息
        summary_sections = self._split_text_by_sections(summary_text)
        for section in summary_sections:
            if section.strip():
                nodes.append(_node(title, section))

        # 如果有聊天要点，同样按大点分割
        if outline_text:
            outline_sections = self._split_text_by_sections(outline_text)
            for section in outline_sections:
                if section.strip():
                    nodes.append(_node("聊天要点", section))
        
        return nodes

    async def _send_forward_summary(self, event: AstrMessageEvent, summary_text: str, outline_text: str = ""):
        """Send summary as a merged forward message; fallback to plain text on failure.
        
        每个大点作为单独一条消息，以合并转发形式发送。
        """
        try:
            ai_event = self._ensure_aiocqhttp_event(event)
        except TypeError:
            return event.plain_result(summary_text)

        client = ai_event.bot
        try:
            login_info = await client.api.call_action("get_login_info")
            self_id = str(login_info.get("user_id", ""))
        except Exception as exc:
            logger.warning("获取 bot 身份失败，改用普通文本: %s", exc)
            return event.plain_result(summary_text)

        nodes = self._build_forward_nodes(
            title="群聊总结", 
            self_id=self_id, 
            summary_text=summary_text, 
            outline_text=outline_text
        )
        if not nodes:
            return event.plain_result("(暂无内容)")
        
        # 确定发送目标（群聊或私聊）
        group_id = getattr(event, "get_group_id", lambda: None)()
        user_id = getattr(event, "get_sender_id", lambda: None)() or getattr(event, "get_user_id", lambda: None)()
        
        is_group = bool(group_id)
        target_id = self._normalize_group_id(group_id) if is_group else user_id
        action_name = "send_group_forward_msg" if is_group else "send_private_forward_msg"
        id_param = "group_id" if is_group else "user_id"
        
        logger.debug("准备发送合并转发: %s=%s, 节点数=%d", id_param, target_id, len(nodes))

        async def _send(nodes_payload: List[dict]) -> dict:
            return await client.api.call_action(
                action_name,
                **{id_param: target_id, "messages": nodes_payload},
            )

        # 策略 1: message segment 模式
        try:
            resp = await _send(nodes)
            if isinstance(resp, dict) and resp.get("status") == "failed":
                raise RuntimeError(f"API 返回失败: {resp}")
            logger.info("合并转发总结发送成功（message segment 模式）")
            return None
        except Exception as exc:
            logger.warning("发送合并转发总结失败（message segment 模式）：%s", exc)

        # 策略 2: 纯文本 content 模式
        plain_nodes = self._build_forward_nodes(
            title="群聊总结",
            self_id=self_id,
            summary_text=summary_text,
            outline_text=outline_text,
            as_plain=True,
        )
        try:
            resp = await _send(plain_nodes)
            if isinstance(resp, dict) and resp.get("status") == "failed":
                raise RuntimeError(f"API 返回失败: {resp}")
            logger.info("合并转发总结发送成功（纯文本模式）")
            return None
        except Exception as exc:
            logger.warning("发送合并转发总结失败（纯文本模式）：%s", exc)

        # 策略 3: 降级为普通文本
        logger.warning("合并转发均失败，降级为普通文本")
        text = f"📝 群聊总结\n\n{summary_text.strip()}"
        if outline_text:
            text = f"{text}\n\n📌 聊天要点\n{outline_text.strip()}"
        return event.plain_result(text[:4000])

    async def _send_image_summary(self, event: AstrMessageEvent, summary_text: str, title: str = "群聊总结"):
        """将总结内容渲染为图片并发送。
        
        使用 Pillow 库将文字渲染为图片。
        
        Args:
            event: 消息事件
            summary_text: 总结文本
            title: 标题
        
        Returns:
            MessageResult 或 None，如果渲染失败返回 False 表示需要降级
        """
        try:
            # 延迟导入 Pillow
            try:
                from PIL import Image, ImageDraw, ImageFont
            except ImportError:
                logger.error("图片渲染需要 Pillow 库，请安装: pip install Pillow")
                return False
            
            cfg = _IMAGE_CONFIG
            width = cfg["width"]
            padding = cfg["padding"]
            content_width = width - 2 * padding
            
            # 尝试加载字体，如果失败则使用默认字体
            title_font = None
            content_font = None
            footer_font = None
            try:
                # 尝试常见的中文字体路径
                font_paths = [
                    "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
                    "C:/Windows/Fonts/simhei.ttf",  # 黑体
                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux 文泉驿
                    "/System/Library/Fonts/PingFang.ttc",  # macOS 苹方
                ]
                font_path = None
                for fp in font_paths:
                    if Path(fp).exists():
                        font_path = fp
                        break
                
                if font_path:
                    title_font = ImageFont.truetype(font_path, cfg["title_font_size"])
                    content_font = ImageFont.truetype(font_path, cfg["content_font_size"])
                    footer_font = ImageFont.truetype(font_path, 12)
            except Exception as font_err:
                logger.warning("加载字体失败: %s，使用默认字体", font_err)
            
            # 如果没有成功加载字体，使用默认字体
            if not title_font:
                title_font = ImageFont.load_default()
                content_font = ImageFont.load_default()
                footer_font = ImageFont.load_default()
            
            # 文本自动换行处理
            def wrap_text(text: str, font, max_width: int) -> list:
                """将文本按宽度自动换行"""
                lines = []
                for paragraph in text.split('\n'):
                    if not paragraph.strip():
                        lines.append('')
                        continue
                    
                    current_line = ''
                    for char in paragraph:
                        test_line = current_line + char
                        try:
                            bbox = font.getbbox(test_line)
                            text_width = bbox[2] - bbox[0]
                        except:
                            text_width = len(test_line) * cfg["content_font_size"]
                        
                        if text_width <= max_width:
                            current_line = test_line
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = char
                    
                    if current_line:
                        lines.append(current_line)
                
                return lines
            
            # 准备文本内容
            content_lines = wrap_text(summary_text.strip(), content_font, content_width - 20)
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            footer_text = "由 AstrBot 群聊总结插件生成"
            
            # 计算高度
            try:
                title_bbox = title_font.getbbox(title)
                title_height = title_bbox[3] - title_bbox[1]
                line_height = content_font.getbbox("测试")[3] + cfg["line_spacing"]
            except:
                title_height = cfg["title_font_size"] + 5
                line_height = cfg["content_font_size"] + cfg["line_spacing"]
            
            # 总高度 = 上边距 + 标题区域 + 分割线 + 内容区域 + 页脚 + 下边距
            header_height = title_height + 30  # 标题 + 时间
            content_height = len(content_lines) * line_height + 20
            footer_height = 40
            total_height = padding + header_height + 20 + content_height + footer_height + padding
            
            # 创建图片
            img = Image.new('RGB', (width, total_height), cfg["bg_color"])
            draw = ImageDraw.Draw(img)
            
            # 绘制卡片背景（圆角矩形效果用普通矩形替代）
            card_margin = 10
            draw.rectangle(
                [card_margin, card_margin, width - card_margin, total_height - card_margin],
                fill=cfg["card_color"],
                outline=cfg["accent_color"],
                width=2
            )
            
            y = padding + 10
            
            # 绘制标题
            draw.text((padding + 10, y), f"📝 {title}", font=title_font, fill=cfg["title_color"])
            
            # 绘制时间（右对齐）
            try:
                time_bbox = footer_font.getbbox(time_str)
                time_width = time_bbox[2] - time_bbox[0]
            except:
                time_width = len(time_str) * 8
            draw.text((width - padding - time_width - 10, y + 5), time_str, font=footer_font, fill=cfg["footer_color"])
            
            y += header_height
            
            # 绘制分割线
            draw.line([(padding + 10, y), (width - padding - 10, y)], fill=(240, 240, 240), width=2)
            y += 20
            
            # 绘制内容
            for line in content_lines:
                draw.text((padding + 10, y), line, font=content_font, fill=cfg["content_color"])
                y += line_height
            
            # 绘制页脚分割线
            y += 10
            draw.line([(padding + 10, y), (width - padding - 10, y)], fill=(240, 240, 240), width=1)
            y += 10
            
            # 绘制页脚（居中）
            try:
                footer_bbox = footer_font.getbbox(footer_text)
                footer_width = footer_bbox[2] - footer_bbox[0]
            except:
                footer_width = len(footer_text) * 8
            footer_x = (width - footer_width) // 2
            draw.text((footer_x, y), footer_text, font=footer_font, fill=cfg["footer_color"])
            
            # 保存图片到内存并转为 base64
            import base64
            from io import BytesIO
            
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # 同时保存到本地作为备份
            image_dir = self._summary_storage / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            image_filename = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
            image_path = image_dir / image_filename
            img.save(str(image_path), "PNG", quality=95)
            
            logger.info("总结图片生成成功: %s", image_path)
            # 使用 base64 格式发送图片
            return event.image_result(f"base64://{image_base64}")
                
        except Exception as exc:
            logger.error("图片渲染失败: %s，将降级为合并转发", exc)
            return False  # 返回 False 表示需要降级

    async def _send_summary(self, event: AstrMessageEvent, summary_text: str, outline_text: str = "", title: str = "群聊总结"):
        """发送总结内容，根据配置选择图片或文本模式。
        
        Args:
            event: 消息事件
            summary_text: 总结文本
            outline_text: 聊天要点（可选）
            title: 标题
        
        Returns:
            MessageResult 或 None
        """
        # 检查是否启用图片渲染
        render_as_image = self.settings.get("render_as_image", False)
        
        if render_as_image:
            # 图片模式：将总结和要点合并渲染
            full_text = summary_text.strip()
            if outline_text:
                full_text = f"{full_text}\n\n📌 聊天要点\n{outline_text.strip()}"
            result = await self._send_image_summary(event, full_text, title)
            # 如果图片渲染成功，返回结果；如果返回 False 表示失败，降级为合并转发
            if result is not False:
                return result
            logger.info("图片渲染失败，降级为合并转发模式")
        
        # 文本模式或图片渲染失败降级：使用合并转发
        return await self._send_forward_summary(event, summary_text, outline_text)

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------
    def _build_topic_instruction(self, base_instruction: str, topic: str | None = None) -> str:
        """根据用户指定的话题构建增强的指令。
        
        Args:
            base_instruction: 基础指令文本
            topic: 用户指定的关注话题（可选）
        
        Returns:
            组合后的指令文本
        """
        if not topic or not topic.strip():
            return base_instruction
        
        topic = topic.strip()
        topic_instruction = (
            f"【重点关注话题】请特别关注与「{topic}」相关的讨论内容，"
            f"优先提取和总结与此话题相关的信息、结论和待办事项。"
            f"如果聊天记录中没有与该话题相关的内容，请明确说明。\n\n"
        )
        return topic_instruction + base_instruction

    async def _summarize_text(
        self,
        chat_text: str,
        *,
        extra_instruction: str = "",
        umo: str | None = None,
        max_tokens: int = 0,
        provider_id: str | None = None,
    ) -> str:
        # 优先使用指定的 provider_id
        provider = None
        if provider_id:
            try:
                provider = self.context.get_provider_by_id(provider_id)
                if provider:
                    logger.debug("使用指定的 LLM Provider: %s", provider_id)
            except Exception as e:
                logger.warning("获取指定的 Provider '%s' 失败: %s，将使用默认 Provider", provider_id, e)
        
        # 如果没有指定或获取失败，使用默认 provider
        if not provider:
            provider = self.context.get_using_provider(umo=umo)
        
        if not provider:
            return "当前未配置可用的 LLM Provider，无法生成总结。"

        contexts = [
            {
                "role": "user",
                "content": f"{chat_text}\n\n[SummarizationInstruction]{extra_instruction or '请输出结构化的重点总结，保持简短优美，不要使用 Markdown。'}",
            },
        ]
        kwargs: Dict[str, Any] = {}
        if max_tokens and max_tokens > 0:
            kwargs["max_tokens"] = max_tokens

        try:
            logger.info("LLM[%s] 调用开始, prompt长度=%d", self._instance_id, len(chat_text))
            response = await provider.text_chat(
                prompt=self.prompt,
                contexts=contexts,
                **kwargs,
            )
            logger.info("LLM[%s] 调用完成", self._instance_id)
        except Exception as exc:
            logger.error("LLM 调用失败: %s", exc)
            return "LLM 调用失败，请检查模型配置后重试。"
        return response.completion_text

    def _apply_token_budget(self, text: str, token_limit: int) -> str:
        if token_limit <= 0:
            return text
        approx_chars = max(200, token_limit * 4)
        if len(text) <= approx_chars:
            return text
        return text[-approx_chars:]

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    @filter.command("消息总结")
    async def summary(self, event: AstrMessageEvent, count: int | None = None, topic: str | None = None):
        """群聊场景触发消息总结
        
        用法:
            /消息总结 <数量> [关注话题]
        
        示例:
            /消息总结 50
            /消息总结 50 技术讨论
            /消息总结 100 会议安排和待办事项
        """
        if count is None:
            yield event.plain_result(
                "未传入要总结的聊天记录数量\n"
                "请按「/消息总结 20」格式发送\n"
                "可选：添加关注话题，如「/消息总结 50 技术讨论」"
            )
            event.stop_event()
            return

        self._reload_settings()
        limit = max(1, self.settings.get("limits", {}).get("max_chat_records", 200))
        count_value = max(1, min(int(count), limit))
        if count > limit:
            yield event.plain_result(f"单次最多支持 {limit} 条记录，已自动按上限 {limit} 条处理~")

        ai_event = self._ensure_aiocqhttp_event(event)
        chat_text, _ = await self._collect_group_messages(
            ai_event.bot,
            event.get_group_id(),
            count=count_value,
        )
        if not chat_text:
            yield event.plain_result("未找到可供总结的群聊记录~")
            return

        base_instruction = "请突出关键议题、明确结论和 TODO，并附上时间范围；回复保持简短优美，不要使用 Markdown。"
        instruction = self._build_topic_instruction(base_instruction, topic)
        
        summary_text = await self._summarize_text(
            chat_text,
            extra_instruction=instruction,
            umo=event.unified_msg_origin,
            max_tokens=self.settings.get("limits", {}).get("max_tokens", 2000),
        )
        result = await self._send_summary(event, summary_text)
        if result:
            yield result

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE)
    @filter.command("群总结")
    async def private_summary(
        self,
        event: AstrMessageEvent,
        count: int | None = None,
        group_id: int | None = None,
        topic: str | None = None,
    ):
        """私聊指定群号进行消息总结
        
        用法:
            /群总结 <数量> <群号> [关注话题]
        
        示例:
            /群总结 30 123456789
            /群总结 50 123456789 项目进度
            /群总结 100 123456789 技术方案讨论
        """
        if count is None:
            yield event.plain_result(
                "未传入要总结的聊天记录数量\n"
                "请按照「/群总结 30 群号」格式发送~\n"
                "可选：添加关注话题，如「/群总结 50 群号 项目进度」"
            )
            event.stop_event()
            return
        if group_id is None:
            yield event.plain_result(
                "未传入要总结的群号\n"
                "请按照「/群总结 30 群号」格式发送~\n"
                "可选：添加关注话题，如「/群总结 50 群号 项目进度」"
            )
            event.stop_event()
            return

        self._reload_settings()
        limit = max(1, self.settings.get("limits", {}).get("max_chat_records", 200))
        count_value = max(1, min(int(count), limit))
        if count > limit:
            yield event.plain_result(f"单次最多支持 {limit} 条记录，已自动按上限 {limit} 条处理~")

        ai_event = self._ensure_aiocqhttp_event(event)
        client = ai_event.bot
        if not await self._user_in_group(client, group_id, event.get_sender_id()):
            yield event.plain_result("未能确认你在该群内，无法获取群聊摘要。")
            event.stop_event()
            return

        chat_text, _ = await self._collect_group_messages(
            client,
            group_id,
            count=count_value,
        )
        if not chat_text:
            yield event.plain_result("未找到可供总结的群聊记录~")
            return

        base_instruction = "请突出关键议题、结论、TODO，并注明对应的群成员；回复保持简短优美，不要使用 Markdown。"
        instruction = self._build_topic_instruction(base_instruction, topic)
        
        summary_text = await self._summarize_text(
            chat_text,
            extra_instruction=instruction,
            umo=None,
            max_tokens=self.settings.get("limits", {}).get("max_tokens", 2000),
        )
        result = await self._send_summary(event, summary_text)
        if result:
            yield result

    @filter.command("转发总结")
    async def forward_summary(self, event: AstrMessageEvent, topic: str | None = None):
        """对用户发送的合并转发聊天记录进行总结
        
        用法:
            /转发总结 [关注话题]
        
        示例:
            /转发总结
            /转发总结 技术方案
            /转发总结 会议纪要和待办
        
        注意：需要将合并转发的聊天记录与指令一起发送
        """
        ai_event = self._ensure_aiocqhttp_event(event)
        forward_ids = self._extract_forward_ids_from_event(ai_event)
        if not forward_ids:
            yield event.plain_result(
                "未发现转发记录，请将合并转发的聊天记录与指令一起发送。\n"
                "可选：添加关注话题，如「/转发总结 技术讨论」"
            )
            return

        texts: List[str] = []
        for fid in forward_ids:
            text = await self._fetch_forward_messages(ai_event.bot, fid)
            if text:
                texts.append(text)

        if not texts:
            yield event.plain_result("未能读取转发内容，请确认转发消息可访问。")
            return

        chat_text = "\n".join(texts)
        base_instruction = (
            "请根据转发的聊天记录进行总结，突出结论、TODO、时间范围和相关参与者；"
            "回复保持简短优美，不要使用 Markdown。"
        )
        instruction = self._build_topic_instruction(base_instruction, topic)
        
        summary_text = await self._summarize_text(
            chat_text,
            extra_instruction=instruction,
            umo=event.unified_msg_origin,
            max_tokens=self.settings.get("limits", {}).get("max_tokens", 2000),
        )
        result = await self._send_summary(event, summary_text)
        if result:
            yield result

    @filter.command("总结模型列表")
    async def list_providers(self, event: AstrMessageEvent):
        """列出所有可用的 LLM 提供商，用于配置 provider_id
        
        用法:
            /总结模型列表
        """
        try:
            providers = self.context.get_all_providers()
            if not providers:
                yield event.plain_result("当前没有可用的 LLM 提供商。")
                return
            
            lines = ["📋 可用的 LLM 提供商列表：", ""]
            for p in providers:
                # 尝试从 meta 属性获取 ID（AstrBot Provider 通常使用 meta.id）
                p_id = None
                meta = getattr(p, 'meta', None)
                if meta:
                    # meta 可能是对象或字典
                    if hasattr(meta, 'id'):
                        p_id = meta.id
                    elif isinstance(meta, dict):
                        p_id = meta.get('id')
                
                # 获取模型名称
                p_model = getattr(p, 'model_name', None) or getattr(p, 'model', None)
                
                # 尝试获取类型
                p_type = None
                if meta:
                    if hasattr(meta, 'type'):
                        p_type = meta.type
                    elif isinstance(meta, dict):
                        p_type = meta.get('type')
                
                # 构建显示行
                line = f"• ID: {p_id or '(未知)'}"
                if p_model:
                    line += f"\n  模型: {p_model}"
                if p_type:
                    line += f"\n  类型: {p_type}"
                
                # 如果没有 ID，显示 meta 的内容帮助调试
                if not p_id and meta:
                    if hasattr(meta, '__dict__'):
                        meta_attrs = {k: v for k, v in meta.__dict__.items() if not k.startswith('_') and isinstance(v, (str, int, bool))}
                        line += f"\n  [调试] meta 属性: {meta_attrs}"
                    elif isinstance(meta, dict):
                        line += f"\n  [调试] meta 内容: {meta}"
                
                lines.append(line)
            
            lines.append("")
            lines.append("💡 将 ID 填入自动总结配置的 provider_id 字段即可指定使用该模型")
            
            yield event.plain_result("\n".join(lines))
        except Exception as exc:
            logger.error("获取 Provider 列表失败: %s", exc)
            yield event.plain_result(f"获取 Provider 列表失败：{exc}")

    # ------------------------------------------------------------------
    # Auto summary
    # ------------------------------------------------------------------
    async def _auto_summary_loop(self):
        """Auto summary 后台循环任务"""
        logger.info("Auto summary loop[%s] 开始运行", self._instance_id)
        
        # 启动时等待一段时间，让 AstrBot 和平台适配器完成初始化
        startup_delay = 30  # 等待 30 秒
        logger.info("Auto summary: 等待 %s 秒让系统完成初始化...", startup_delay)
        await asyncio.sleep(startup_delay)
        logger.info("Auto summary: 初始化等待完成，开始正常运行")
        
        # 默认间隔时间（分钟），在循环外初始化以避免未定义错误
        interval = 60
        
        while True:
            try:
                settings = self._reload_settings()
                auto_cfg = settings.get("auto_summary", {}) or {}
                interval = max(1, int(auto_cfg.get("interval_minutes", 60)))
                
                if not auto_cfg.get("enabled"):
                    logger.debug("Auto summary 未开启，%s 分钟后再次检查", interval)
                    await asyncio.sleep(interval * 60)
                    continue
                
                # 检查是否有可用的客户端
                client = self._get_aiocqhttp_client()
                if client is None:
                    logger.warning("Auto summary: 等待 aiocqhttp 客户端就绪，60 秒后重试")
                    await asyncio.sleep(60)
                    continue
                
                logger.info("Auto summary[%s]: 开始执行自动总结任务...", self._instance_id)
                async with self._auto_summary_lock:
                    await self._execute_auto_summary(auto_cfg, settings)
                logger.info("Auto summary[%s]: 本轮任务完成，%s 分钟后执行下一轮", self._instance_id, interval)
                
                # 成功执行后等待下一轮
                await asyncio.sleep(interval * 60)
                
            except asyncio.CancelledError:
                logger.info("Auto summary loop 被取消")
                raise
            except Exception:
                logger.exception("自动群聊总结执行失败")
                # 发生异常时也等待一段时间后重试
                await asyncio.sleep(interval * 60)

    async def _execute_auto_summary(self, auto_cfg: dict, settings: dict):
        target_groups = self._normalize_target_groups(auto_cfg.get("target_groups"))
        logger.info(
            "自动总结任务启动: enabled=%s, groups=%s, interval=%s分钟",
            auto_cfg.get("enabled"),
            target_groups,
            auto_cfg.get("interval_minutes"),
        )
        if not target_groups:
            logger.warning("自动总结已启用，但未配置目标群，请在配置中添加 target_groups。")
            return

        client = self._get_aiocqhttp_client()
        if client is None:
            logger.error("自动总结需要 aiocqhttp 适配器，但当前未发现可用实例。")
            return

        max_records = max(1, settings.get("limits", {}).get("max_chat_records", 200))
        max_tokens = settings.get("limits", {}).get("max_tokens", 2000)
        summary_mode = auto_cfg.get("summary_mode", "message_count")
        chunk_size = max(1, int(auto_cfg.get("message_chunk_size", 30)))
        window_minutes = max(1, int(auto_cfg.get("time_window_minutes", 15)))
        broadcast_value = auto_cfg.get("broadcast", True)
        # 支持布尔值和字符串值
        if isinstance(broadcast_value, bool):
            broadcast = broadcast_value
        else:
            broadcast = str(broadcast_value).lower() in {"1", "true", "yes", "on"}
        min_messages = max(1, int(auto_cfg.get("min_messages", 5)))

        instruction = (
            f"请基于已按{'消息数量' if summary_mode == 'message_count' else '时间窗口'}分段的记录进行总结，"
            "每个分段输出关键议题、重要发言人、时间范围以及需要跟进的事项。"
            "最后给出全局重点和 TODO，整体内容要突出重点，保持简短优美，不要使用 Markdown。"
        )

        for group_id in target_groups:
            try:
                chat_text, structured = await self._collect_group_messages(
                    client,
                    group_id,
                    count=max_records,
                )
            except Exception as exc:
                logger.error("拉取群 %s 聊天记录失败：%s", group_id, exc)
                continue

            if not structured:
                logger.info("群 %s 无可总结的消息。", group_id)
                continue

            # 检查是否有新消息（相比上次总结）
            last_msg_time = structured[-1]["time"] if structured else None
            last_summary_time = self._last_summary_time.get(group_id)
            
            if last_summary_time and last_msg_time:
                # 过滤掉上次总结之前的消息，只保留新消息
                new_messages = [msg for msg in structured if msg["time"] > last_summary_time]
                if not new_messages:
                    logger.info(
                        "群 %s 自上次总结(%s)以来无新消息，跳过本轮总结。",
                        group_id,
                        last_summary_time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    continue
                
                # 检查新消息数量是否达到最小阈值
                if len(new_messages) < min_messages:
                    logger.info(
                        "群 %s 新消息数量(%d)少于最小阈值(%d)，跳过本轮总结。",
                        group_id,
                        len(new_messages),
                        min_messages,
                    )
                    continue
                
                logger.info(
                    "群 %s 发现 %d 条新消息（上次总结: %s）",
                    group_id,
                    len(new_messages),
                    last_summary_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                # 使用新消息进行总结，但保留一些上下文
                # 如果新消息太少，使用全部消息以提供上下文
                if len(new_messages) < 10 and len(structured) > len(new_messages):
                    logger.debug("新消息较少，使用全部 %d 条消息以提供上下文", len(structured))
                else:
                    structured = new_messages
                    chat_text = "\n".join(
                        f"[{msg['time']}]「{msg['nickname']}」: {msg['text']}"
                        for msg in structured
                    )
            else:
                # 首次运行，检查消息数量是否达到最小阈值
                if len(structured) < min_messages:
                    logger.info(
                        "群 %s 消息数量(%d)少于最小阈值(%d)，跳过本轮总结。",
                        group_id,
                        len(structured),
                        min_messages,
                    )
                    continue
            
            # 计算内容哈希，避免重复总结相同内容
            content_hash = self._compute_content_hash(structured)
            if content_hash == self._last_summary_hash.get(group_id):
                logger.info("群 %s 消息内容与上次相同，跳过重复总结。", group_id)
                continue

            segments = self._segment_messages(structured, summary_mode, chunk_size, window_minutes)
            outline_text = self._render_segments(segments)
            llm_context = self._apply_token_budget(outline_text or chat_text, max_tokens)
            # 获取配置的 provider_id
            configured_provider_id = auto_cfg.get("provider_id", "") or ""
            summary_text = await self._summarize_text(
                llm_context or chat_text,
                extra_instruction=instruction,
                max_tokens=max_tokens,
                provider_id=configured_provider_id if configured_provider_id.strip() else None,
            )
            logger.info(
                "群 %s 总结完成，记录数=%s，模式=%s，写入中...",
                group_id,
                len(structured),
                summary_mode,
            )
            group_info = await self._safe_group_info(client, group_id)
            file_path = self._persist_summary_file(
                group_id=group_id,
                group_name=group_info.get("group_name") if isinstance(group_info, dict) else "",
                summary_text=summary_text,
                outline_text=outline_text or chat_text,
                messages=structured,
                summary_mode=summary_mode,
            )
            logger.info("自动总结已输出：%s", file_path)

            # 更新上次总结时间和内容哈希
            if structured:
                self._last_summary_time[group_id] = structured[-1]["time"]
                self._last_summary_hash[group_id] = content_hash
                logger.debug("更新群 %s 的上次总结时间为: %s", group_id, self._last_summary_time[group_id])

            if broadcast:
                title = f"群自动总结 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                # 自动总结不发送 outline，避免消息过长
                success = await self._send_group_forward(
                    client, 
                    group_id, 
                    title, 
                    summary_text,
                    ""  # 自动总结不附带原始聊天要点，仅发送 AI 总结
                )
                if success:
                    logger.info("自动总结已成功推送到群 %s", group_id)
                else:
                    logger.error("自动总结推送群 %s 失败", group_id)
            else:
                logger.info("自动总结未推送到群 %s（broadcast=false）", group_id)

    def _segment_messages(
        self,
        messages: List[dict],
        mode: str,
        chunk_size: int,
        window_minutes: int,
    ) -> List[dict]:
        if mode != "time_window":
            return self._segment_by_size(messages, chunk_size)
        return self._segment_by_time(messages, window_minutes)

    def _segment_by_size(self, messages: List[dict], chunk_size: int) -> List[dict]:
        segments: List[dict] = []
        for idx in range(0, len(messages), chunk_size):
            chunk = messages[idx : idx + chunk_size]
            if not chunk:
                continue
            segments.append(
                {
                    "messages": chunk,
                    "start": chunk[0]["time"],
                    "end": chunk[-1]["time"],
                },
            )
        return segments

    def _segment_by_time(self, messages: List[dict], window_minutes: int) -> List[dict]:
        segments: List[dict] = []
        current: List[dict] = []
        window_seconds = window_minutes * 60
        window_start: datetime | None = None

        for msg in messages:
            timestamp = msg["time"]
            if not current:
                current = [msg]
                window_start = timestamp
                continue

            assert window_start is not None
            delta = (timestamp - window_start).total_seconds()
            if delta <= window_seconds:
                current.append(msg)
            else:
                segments.append(
                    {
                        "messages": current,
                        "start": current[0]["time"],
                        "end": current[-1]["time"],
                    },
                )
                current = [msg]
                window_start = timestamp

        if current:
            segments.append(
                {
                    "messages": current,
                    "start": current[0]["time"],
                    "end": current[-1]["time"],
                },
            )
        return segments

    def _render_segments(self, segments: List[dict]) -> str:
        lines: List[str] = []
        for idx, segment in enumerate(segments, 1):
            start = segment["start"].strftime("%Y-%m-%d %H:%M:%S")
            end = segment["end"].strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"[Segment {idx}] {start} - {end} | 消息 {len(segment['messages'])}")
            for msg in segment["messages"]:
                speaker = msg["nickname"]
                timestamp = msg["time"].strftime("%H:%M:%S")
                lines.append(f"- ({timestamp}) {speaker}: {msg['text']}")
        return "\n".join(lines)

    def _persist_summary_file(
        self,
        *,
        group_id: str | int,
        group_name: str | None,
        summary_text: str,
        outline_text: str,
        messages: List[dict],
        summary_mode: str,
    ) -> Path:
        timestamp = datetime.now()
        file_name = f"{self._sanitize_group_id(group_id)}_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        file_path = self._summary_storage / file_name
        first_time = messages[0]["time"].strftime("%Y-%m-%d %H:%M:%S")
        last_time = messages[-1]["time"].strftime("%Y-%m-%d %H:%M:%S")
        content = [
            "# 群自动总结",
            f"- 群号: {group_id}",
            f"- 群名: {group_name or '未知'}",
            f"- 生成时间: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 消息范围: {first_time} ~ {last_time}",
            f"- 采样模式: {'按消息数量分段' if summary_mode == 'message_count' else '按时间窗口分段'}",
            "",
            "## AI 总结",
            summary_text.strip() or "（暂无内容）",
            "",
            "## 会话提要",
            outline_text.strip() or "（暂无记录）",
        ]
        file_path.write_text("\n".join(content), encoding="utf-8")
        return file_path

    def _sanitize_group_id(self, group_id: str | int) -> str:
        return re.sub(r"[^0-9A-Za-z_-]", "_", str(group_id))

    def _compute_content_hash(self, messages: List[dict]) -> str:
        """计算消息内容的哈希值，用于检测内容是否有变化。"""
        content = "".join(
            f"{msg['time'].isoformat()}:{msg['user_id']}:{msg['text']}"
            for msg in messages
        )
        return hashlib.md5(content.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _ensure_aiocqhttp_event(self, event: AstrMessageEvent) -> AiocqhttpMessageEvent:
        if not isinstance(event, AiocqhttpMessageEvent):
            raise TypeError("当前插件仅支持 aiocqhttp 平台。")
        return event

    async def _user_in_group(self, client, group_id: int | str, user_id: str) -> bool:
        try:
            normalized_user = int(user_id)
        except (TypeError, ValueError):
            normalized_user = user_id
        try:
            await client.api.call_action(
                "get_group_member_info",
                group_id=self._normalize_group_id(group_id),
                user_id=normalized_user,
            )
            return True
        except Exception as exc:
            logger.warning("校验群成员身份失败：%s", exc)
            return False

    def _normalize_target_groups(self, groups: Iterable[Any] | None) -> List[str | int]:
        if not groups:
            return []
        if isinstance(groups, str):
            # Support comma/semicolon/whitespace separated single string from config UI
            raw_items = re.split(r"[，,;\\s]+", groups)
            groups = [item for item in raw_items if item]
        result: List[str | int] = []
        seen: set = set()  # 去重
        for item in groups:
            if item is None:
                continue
            text = str(item).strip()
            if not text:
                continue
            try:
                val = int(text)
            except ValueError:
                val = text
            # 去重：避免同一个群被处理多次
            if val not in seen:
                seen.add(val)
                result.append(val)
        return result

    def _get_aiocqhttp_client(self):
        """Get the aiocqhttp client, trying multiple methods."""
        # 如果已缓存且有效，直接返回
        if self._aiocqhttp_client:
            return self._aiocqhttp_client

        # 方法 1: 通过 context.get_platform
        try:
            platform = self.context.get_platform(filter.PlatformAdapterType.AIOCQHTTP)
            if platform and isinstance(platform, AiocqhttpAdapter):
                self._aiocqhttp_client = platform.get_client()
                if self._aiocqhttp_client:
                    logger.debug("通过 get_platform 获取到 aiocqhttp client")
                    return self._aiocqhttp_client
        except Exception as e:
            logger.debug("通过 get_platform 获取 client 失败: %s", e)

        # 方法 2: 遍历 platform_insts
        try:
            if hasattr(self.context, 'platform_manager') and self.context.platform_manager:
                for inst in self.context.platform_manager.platform_insts:
                    if isinstance(inst, AiocqhttpAdapter):
                        self._aiocqhttp_client = inst.get_client()
                        if self._aiocqhttp_client:
                            logger.debug("通过 platform_insts 获取到 aiocqhttp client")
                            return self._aiocqhttp_client
        except Exception as e:
            logger.debug("通过 platform_insts 获取 client 失败: %s", e)
        
        # 方法 3: 尝试通过 platforms 属性
        try:
            if hasattr(self.context, 'platforms'):
                for platform in self.context.platforms:
                    if isinstance(platform, AiocqhttpAdapter):
                        self._aiocqhttp_client = platform.get_client()
                        if self._aiocqhttp_client:
                            logger.debug("通过 platforms 获取到 aiocqhttp client")
                            return self._aiocqhttp_client
        except Exception as e:
            logger.debug("通过 platforms 获取 client 失败: %s", e)

        return self._aiocqhttp_client

    async def _safe_group_info(self, client, group_id: str | int) -> dict:
        try:
            return await client.api.call_action(
                "get_group_info",
                group_id=self._normalize_group_id(group_id),
            )
        except Exception:
            return {}

    async def terminate(self):
        if self._auto_summary_task:
            self._auto_summary_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._auto_summary_task
            self._auto_summary_task = None

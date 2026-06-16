# Changelog

## 1.7.1

### Changed

- Decoupled message-cache writes from `auto_summary.enabled`: as long as `cache_enabled=true` and the group is in `target_groups`, messages are persisted to SQLite even when auto summary is disabled. The `enabled` flag now only controls whether the scheduled auto summary runs.

### Fixed

- Manual summary cache path now over-fetches from the local cache (1.5x, capped at 3000) and filters noise before checking the count, so it returns the most recent `count` meaningful messages and falls back to OneBot only when the cache is genuinely insufficient.

## 1.7.0

### Added

- Manual summary commands (`/消息总结`, `/群总结`) now read the local SQLite cache first and fall back to OneBot history only when the cache is insufficient.
- Added `_NOISE_TOKEN_PATTERN` to drop noise-only messages (pure image / face / voice / video / reply / forward placeholders) before summarizing, covering both the OneBot and cache outline placeholder formats.
- Added a history-fetch log recording requested vs. actually returned message counts.

### Changed

- Manual summary now over-fetches history (1.5x, capped at 3000) so the target message count can still be met after noise filtering.

## 1.6.0

### Added

- Added `privacy.redact_sensitive_info` to control whether chat text is redacted before being sent to the LLM.
- Added `summary_include_time` to control whether summary points should include message times.
- Added regression tests for privacy redaction and summary time instructions.

## 1.5.0

### Added

- Added an event-driven local SQLite cache for auto summaries.
- Added `auto_summary.cache_enabled`, `auto_summary.max_records`, `auto_summary.history_backfill_count`, and `auto_summary.cache_retention_days`.
- Added persistent per-group summary cursors based on `timestamp + message_id`.
- Added regression tests for message caching, plugin loading, history backfill, and auto-summary state handling.

### Changed

- Auto summary now listens to target group messages and summarizes local unsummarized messages first.
- `get_group_msg_history` is no longer the primary auto-summary data source.
- Auto-summary history backfill is limited by `auto_summary.history_backfill_count`.
- Auto-summary history backfill no longer expands merged-forward messages.
- `limits.max_chat_records` now mainly applies to manual summary commands.

### Fixed

- Prevented bot self-messages from being cached and summarized again.
- Prevented LLM failure messages from being archived or advancing summary state.
- Prevented old messages from re-entering summaries after persistent cursor filtering.
- Fixed numeric `message_id` cursor ordering for same-second messages.
- Fixed local module loading to avoid importing another `message_cache` module from `sys.path`.
- Ensured SQLite connections are explicitly closed after each operation.

## 1.4.0

### Added

- Added important message reminder support for watched groups.
- Added regex-based reminder rules with optional aliases.
- Added mention-related reminder matching:
  - direct `@target_user_id`
  - replies to the target user
  - configured name aliases
- Added private reminder delivery with a short text summary and original-message forward node.
- Added tests for reminder matching and handler behavior.

### Fixed

- Hardened important message reminder handling for empty configs, invalid regex patterns, and missing target settings.
- Stopped summary commands from falling through to normal chat handling.

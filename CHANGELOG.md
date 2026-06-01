# Changelog

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

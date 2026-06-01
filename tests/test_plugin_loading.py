import subprocess
import sys
from pathlib import Path


def test_main_loads_when_plugin_directory_is_not_on_sys_path():
    plugin_dir = Path(__file__).resolve().parents[1]
    script = f"""
import importlib.util
import sys
from pathlib import Path

plugin_dir = Path(r"{plugin_dir}")
sys.path = [p for p in sys.path if p not in ("", str(plugin_dir))]
spec = importlib.util.spec_from_file_location("isolated_chatsummary_main", plugin_dir / "main.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.ChatSummary.__name__)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=plugin_dir.parent,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "ChatSummary" in result.stdout


def test_main_loads_local_message_cache_when_global_name_collides(tmp_path):
    plugin_dir = Path(__file__).resolve().parents[1]
    collision_dir = tmp_path / "collision"
    collision_dir.mkdir()
    (collision_dir / "message_cache.py").write_text(
        "raise RuntimeError('wrong message_cache imported')\n",
        encoding="utf-8",
    )
    script = f"""
import importlib.util
import sys
from pathlib import Path

plugin_dir = Path(r"{plugin_dir}")
collision_dir = Path(r"{collision_dir}")
sys.path.insert(0, str(collision_dir))
sys.path = [p for p in sys.path if p not in ("", str(plugin_dir))]
spec = importlib.util.spec_from_file_location("isolated_chatsummary_main_collision", plugin_dir / "main.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.MessageCache.__module__)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=plugin_dir.parent,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "_chatsummary_message_cache" in result.stdout

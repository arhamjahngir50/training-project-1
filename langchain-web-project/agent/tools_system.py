import os
import subprocess
import shutil
import platform
import psutil
import glob
import signal
from datetime import datetime
from pathlib import Path

from langchain.tools import tool


def resolve_path(path: str) -> str:
    path = path.strip()
    if path.startswith("Desktop/") or path == "Desktop":
        path = str(Path.home() / path)
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    return os.path.abspath(path)


@tool
def create_folder(path: str) -> str:
    """Create a folder (and any parent folders) at the given path."""
    try:
        p = resolve_path(path)
        os.makedirs(p, exist_ok=True)
        return f"✅ Folder created: {p}"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def delete_file_or_folder(path: str) -> str:
    """Delete a file or folder (recursively if folder)."""
    try:
        p = resolve_path(path)
        if os.path.isdir(p):
            shutil.rmtree(p)
            return f"✅ Folder deleted: {p}"
        elif os.path.isfile(p):
            os.remove(p)
            return f"✅ File deleted: {p}"
        else:
            return f"⚠️ Path not found: {p}"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def list_files(path: str = ".") -> str:
    """List files and folders in a directory."""
    try:
        p = resolve_path(path)
        items = sorted(os.listdir(p))
        if not items:
            return f"📂 '{p}' is empty."
        result = [f"📂 Contents of {p}:\n"]
        for item in items:
            full = os.path.join(p, item)
            kind = "📁" if os.path.isdir(full) else "📄"
            size = os.path.getsize(full) if os.path.isfile(full) else ""
            size_str = f"  ({size:,} bytes)" if size != "" else ""
            result.append(f"  {kind} {item}{size_str}")
        return "\n".join(result)
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def read_file(path: str) -> str:
    """Read and return the contents of a text file."""
    try:
        p = resolve_path(path)
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        lines = content.splitlines()
        if len(lines) > 200:
            content = "\n".join(lines[:200]) + f"\n\n... (truncated, {len(lines)} total lines)"
        return f"📄 Contents of {p}:\n\n{content}"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write text content to a file (creates or overwrites)."""
    try:
        p = resolve_path(path)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return f"✅ File written: {p} ({len(content)} chars)"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def append_to_file(path: str, content: str) -> str:
    """Append text to an existing file (or create it)."""
    try:
        p = resolve_path(path)
        with open(p, "a", encoding="utf-8") as f:
            f.write(content)
        return f"✅ Content appended to {p}"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def copy_file_or_folder(source: str, destination: str) -> str:
    """Copy a file or folder to a new location."""
    try:
        src = resolve_path(source)
        dst = resolve_path(destination)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
            return f"✅ Folder copied: {src} → {dst}"
        else:
            shutil.copy2(src, dst)
            return f"✅ File copied: {src} → {dst}"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def move_file_or_folder(source: str, destination: str) -> str:
    """Move or rename a file or folder."""
    try:
        src = resolve_path(source)
        dst = resolve_path(destination)
        shutil.move(src, dst)
        return f"✅ Moved: {src} → {dst}"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def find_files(pattern: str, search_path: str = "~") -> str:
    """Search for files matching a glob pattern (e.g. '*.pdf') in a directory."""
    try:
        base = resolve_path(search_path)
        matches = glob.glob(os.path.join(base, "**", pattern), recursive=True)
        if not matches:
            return f"🔍 No files found matching '{pattern}' in {base}"
        result = [f"🔍 Found {len(matches)} file(s) matching '{pattern}':\n"]
        for m in matches[:50]:
            result.append(f"  {m}")
        if len(matches) > 50:
            result.append(f"  ... and {len(matches)-50} more")
        return "\n".join(result)
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def run_shell_command(command: str) -> str:
    """Run any shell command and return its output."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True,
            text=True, timeout=30, cwd=os.path.expanduser("~")
        )
        output = result.stdout.strip()
        error = result.stderr.strip()
        parts = []
        if output:
            parts.append(f"STDOUT:\n{output}")
        if error:
            parts.append(f"STDERR:\n{error}")
        parts.append(f"Return code: {result.returncode}")
        return "\n".join(parts) if parts else "(no output)"
    except subprocess.TimeoutExpired:
        return "❌ Command timed out (30s limit)"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def open_application(app_name: str) -> str:
    """Open an application by name (e.g. 'firefox', 'gedit', 'nautilus')."""
    try:
        subprocess.Popen([app_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"✅ Launched: {app_name}"
    except FileNotFoundError:
        return f"❌ Application not found: {app_name}"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def open_file_with_default_app(path: str) -> str:
    """Open a file with its default application using xdg-open."""
    try:
        p = resolve_path(path)
        subprocess.Popen(["xdg-open", p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"✅ Opened: {p}"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def get_system_info() -> str:
    """Get detailed system information: OS, CPU, RAM, disk, uptime."""
    try:
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        cpu_percent = psutil.cpu_percent(interval=1)
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time
        info = {
            "OS": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "Hostname": platform.node(),
            "CPU": f"{psutil.cpu_count()} cores @ {cpu_percent}% usage",
            "RAM": f"{mem.used / 1e9:.1f} GB / {mem.total / 1e9:.1f} GB ({mem.percent}%)",
            "Disk": f"{disk.used / 1e9:.1f} GB / {disk.total / 1e9:.1f} GB ({disk.percent}%)",
            "Uptime": str(uptime).split(".")[0],
        }
        return "💻 System Info:\n" + "\n".join(f"  {k}: {v}" for k, v in info.items())
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def list_running_processes(filter_name: str = "") -> str:
    """List running processes. Optionally filter by name substring."""
    try:
        procs = []
        for p in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                if filter_name.lower() in p.info["name"].lower():
                    mem_mb = p.info["memory_info"].rss / 1e6
                    procs.append((p.info["pid"], p.info["name"], mem_mb))
            except Exception:
                pass
        if not procs:
            return "No matching processes found."
        procs.sort(key=lambda x: x[2], reverse=True)
        lines = [f"{'PID':>7}  {'MEM (MB)':>10}  NAME", "-" * 35]
        for pid, name, mem in procs[:30]:
            lines.append(f"{pid:>7}  {mem:>10.1f}  {name}")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def kill_process(pid_or_name: str) -> str:
    """Kill a process by PID (number) or name."""
    try:
        try:
            pid = int(pid_or_name)
            os.kill(pid, signal.SIGTERM)
            return f"✅ Sent SIGTERM to PID {pid}"
        except ValueError:
            pass
        killed = []
        for p in psutil.process_iter(["pid", "name"]):
            if pid_or_name.lower() in p.info["name"].lower():
                p.terminate()
                killed.append(f"{p.info['name']} (PID {p.info['pid']})")
        return f"✅ Terminated: {', '.join(killed)}" if killed else f"⚠️ No process matching '{pid_or_name}'"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def check_internet() -> str:
    """Check internet connectivity."""
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "3", "8.8.8.8"],
            capture_output=True, text=True, timeout=5
        )
        return "🌐 Internet: Connected" if result.returncode == 0 else "❌ Internet: Not connected"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def get_current_datetime() -> str:
    """Get the current date and time."""
    return f"🕐 {datetime.now().strftime('%A, %B %d, %Y — %H:%M:%S')}"


@tool
def take_screenshot(save_path: str = "~/Desktop/screenshot.png") -> str:
    """Take a screenshot. Requires scrot: sudo apt install scrot"""
    try:
        p = resolve_path(save_path)
        for cmd in [["scrot", p], ["gnome-screenshot", "-f", p]]:
            if subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0:
                return f"✅ Screenshot saved: {p}"
        return "❌ Install scrot: sudo apt install scrot"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def set_volume(level: int) -> str:
    """Set system volume 0-100."""
    try:
        level = max(0, min(100, level))
        subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"],
            capture_output=True
        )
        return f"🔊 Volume set to {level}%"
    except Exception as e:
        return f"❌ Error: {e}"


SYSTEM_TOOLS = [
    create_folder, delete_file_or_folder, list_files, read_file,
    write_file, append_to_file, copy_file_or_folder, move_file_or_folder,
    find_files, run_shell_command, open_application, open_file_with_default_app,
    get_system_info, list_running_processes, kill_process, check_internet,
    get_current_datetime, take_screenshot, set_volume,
]
"""
Extended tools: clipboard, git, zip/unzip, desktop notifications,
environment variables, scheduled tasks (cron), disk usage, network info.
"""
import os
import subprocess
import shutil
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime

from langchain_classic.tools import tool
from .tools_system import resolve_path


# ── Clipboard ────────────────────────────────────────────────────────────────

@tool
def clipboard_read() -> str:
    """Read the current contents of the system clipboard."""
    try:
        result = subprocess.run(
            ["xclip", "-selection", "clipboard", "-o"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return f"📋 Clipboard: {result.stdout}"
        # fallback: xsel
        result = subprocess.run(
            ["xsel", "--clipboard", "--output"],
            capture_output=True, text=True, timeout=5
        )
        return f"📋 Clipboard: {result.stdout}" if result.returncode == 0 else "❌ Install xclip: sudo apt install xclip"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def clipboard_write(text: str) -> str:
    """Write text to the system clipboard."""
    try:
        for cmd in [
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
        ]:
            try:
                proc = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=5)
                if proc.returncode == 0:
                    return f"✅ Copied to clipboard ({len(text)} chars)"
            except FileNotFoundError:
                continue
        return "❌ Install xclip: sudo apt install xclip"
    except Exception as e:
        return f"❌ Error: {e}"


# ── Desktop Notifications ────────────────────────────────────────────────────

@tool
def notify(title: str, message: str, urgency: str = "normal") -> str:
    """
    Send a desktop notification.
    urgency: 'low' | 'normal' | 'critical'
    Requires: sudo apt install libnotify-bin
    """
    try:
        subprocess.Popen(
            ["notify-send", "-u", urgency, title, message],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return f"🔔 Notification sent: [{urgency}] {title} — {message}"
    except FileNotFoundError:
        return "❌ Install notify-send: sudo apt install libnotify-bin"
    except Exception as e:
        return f"❌ Error: {e}"


# ── Zip / Archive ─────────────────────────────────────────────────────────────

@tool
def zip_files(source_path: str, output_zip: str) -> str:
    """
    Zip a file or folder.
    source_path: path to file or folder to compress
    output_zip: path for the resulting .zip file
    """
    try:
        src = resolve_path(source_path)
        out = resolve_path(output_zip)
        if not out.endswith(".zip"):
            out += ".zip"
        if os.path.isdir(src):
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(src):
                    for file in files:
                        fp = os.path.join(root, file)
                        zf.write(fp, os.path.relpath(fp, os.path.dirname(src)))
        else:
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(src, os.path.basename(src))
        size = os.path.getsize(out)
        return f"✅ Zipped to: {out} ({size:,} bytes)"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def unzip_file(zip_path: str, destination: str = "") -> str:
    """
    Unzip a .zip file.
    destination: folder to extract into (defaults to same folder as zip)
    """
    try:
        zp = resolve_path(zip_path)
        dst = resolve_path(destination) if destination else os.path.dirname(zp)
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(dst)
        return f"✅ Extracted to: {dst}"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def create_tarball(source_path: str, output_tar: str, compress: str = "gz") -> str:
    """
    Create a .tar.gz or .tar.bz2 archive.
    compress: 'gz' (default) or 'bz2'
    """
    try:
        src = resolve_path(source_path)
        out = resolve_path(output_tar)
        mode = f"w:{compress}"
        if not out.endswith(f".tar.{compress}"):
            out += f".tar.{compress}"
        with tarfile.open(out, mode) as tf:
            tf.add(src, arcname=os.path.basename(src))
        size = os.path.getsize(out)
        return f"✅ Archive created: {out} ({size:,} bytes)"
    except Exception as e:
        return f"❌ Error: {e}"


# ── Git ───────────────────────────────────────────────────────────────────────

@tool
def git_status(repo_path: str = ".") -> str:
    """Show git status of a repository."""
    try:
        p = resolve_path(repo_path)
        result = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=p, capture_output=True, text=True, timeout=10
        )
        return result.stdout if result.returncode == 0 else f"❌ {result.stderr.strip()}"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def git_log(repo_path: str = ".", count: int = 10) -> str:
    """Show recent git commit history (last N commits)."""
    try:
        p = resolve_path(repo_path)
        result = subprocess.run(
            ["git", "log", f"-{count}", "--oneline", "--decorate"],
            cwd=p, capture_output=True, text=True, timeout=10
        )
        return result.stdout if result.returncode == 0 else f"❌ {result.stderr.strip()}"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def git_commit(repo_path: str, message: str, add_all: bool = True) -> str:
    """Stage and commit changes in a git repo."""
    try:
        p = resolve_path(repo_path)
        if add_all:
            subprocess.run(["git", "add", "-A"], cwd=p, capture_output=True, timeout=10)
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=p, capture_output=True, text=True, timeout=15
        )
        return result.stdout if result.returncode == 0 else f"❌ {result.stderr.strip()}"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def git_clone(url: str, destination: str = "~") -> str:
    """Clone a git repository to a local path."""
    try:
        dst = resolve_path(destination)
        result = subprocess.run(
            ["git", "clone", url], cwd=dst,
            capture_output=True, text=True, timeout=60
        )
        return result.stdout + result.stderr if result.returncode == 0 else f"❌ {result.stderr.strip()}"
    except Exception as e:
        return f"❌ Error: {e}"


# ── Environment Variables ─────────────────────────────────────────────────────

@tool
def get_env_var(name: str) -> str:
    """Get the value of an environment variable."""
    val = os.environ.get(name)
    return f"{name}={val}" if val is not None else f"⚠️ '{name}' is not set"


@tool
def list_env_vars(prefix: str = "") -> str:
    """List environment variables, optionally filtered by prefix."""
    items = {k: v for k, v in os.environ.items() if k.startswith(prefix.upper())}
    if not items:
        return f"No env vars matching prefix '{prefix}'"
    return "\n".join(f"{k}={v}" for k, v in sorted(items.items()))


# ── Disk Usage ────────────────────────────────────────────────────────────────

@tool
def disk_usage(path: str = "~") -> str:
    """Show disk usage of a directory (human-readable, top 10 subdirs by size)."""
    try:
        p = resolve_path(path)
        result = subprocess.run(
            ["du", "-sh", "--max-depth=1", p],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout if result.returncode == 0 else f"❌ {result.stderr.strip()}"
    except Exception as e:
        return f"❌ Error: {e}"


# ── Network Info ──────────────────────────────────────────────────────────────

@tool
def network_info() -> str:
    """Show network interfaces, IP addresses, and connection stats."""
    try:
        result = subprocess.run(
            ["ip", "addr", "show"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout if result.returncode == 0 else "❌ Could not get network info"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def ping_host(host: str, count: int = 4) -> str:
    """Ping a hostname or IP address."""
    try:
        result = subprocess.run(
            ["ping", "-c", str(count), host],
            capture_output=True, text=True, timeout=15
        )
        return result.stdout if result.returncode == 0 else f"❌ {result.stderr.strip()}"
    except Exception as e:
        return f"❌ Error: {e}"


# ── Cron / Scheduled Tasks ────────────────────────────────────────────────────

@tool
def list_cron_jobs() -> str:
    """List the current user's cron jobs."""
    try:
        result = subprocess.run(
            ["crontab", "-l"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return f"⏰ Cron jobs:\n{result.stdout}" if result.stdout.strip() else "⏰ No cron jobs scheduled."
        return "⚠️ No crontab for this user."
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def add_cron_job(schedule: str, command: str) -> str:
    """
    Add a cron job for the current user.
    schedule: cron expression e.g. '0 9 * * *' (daily at 9am)
    command: shell command to run
    """
    try:
        existing = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        current = existing.stdout if existing.returncode == 0 else ""
        new_line = f"{schedule} {command}\n"
        new_crontab = current.rstrip("\n") + "\n" + new_line
        proc = subprocess.run(
            ["crontab", "-"],
            input=new_crontab, capture_output=True, text=True
        )
        return f"✅ Cron job added: {new_line.strip()}" if proc.returncode == 0 else f"❌ {proc.stderr}"
    except Exception as e:
        return f"❌ Error: {e}"


EXTENDED_TOOLS = [
    clipboard_read, clipboard_write,
    notify,
    zip_files, unzip_file, create_tarball,
    git_status, git_log, git_commit, git_clone,
    get_env_var, list_env_vars,
    disk_usage, network_info, ping_host,
    list_cron_jobs, add_cron_job,
]
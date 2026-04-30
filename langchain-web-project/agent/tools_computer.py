"""Computer control tools: mouse and keyboard actions via pyautogui."""
import re
from typing import Optional

from langchain_classic.tools import tool

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.05
except ImportError:
    pyautogui = None


def _validate_pyautogui() -> Optional[str]:
    if pyautogui is None:
        return "❌ pyautogui is not installed. Install with `pip install pyautogui`."
    return None


@tool
def move_mouse(x: int, y: int, duration: float = 0.0) -> str:
    """Move the mouse pointer to the given screen coordinates."""
    err = _validate_pyautogui()
    if err:
        return err
    try:
        pyautogui.moveTo(x, y, duration=duration)
        return f"✅ Mouse moved to ({x}, {y})"
    except Exception as e:
        return f"❌ Error moving mouse: {e}"


@tool
def click(x: Optional[int] = None, y: Optional[int] = None, button: str = "left", clicks: int = 1, interval: float = 0.0) -> str:
    """Click at the specified coordinates or at the current cursor position."""
    err = _validate_pyautogui()
    if err:
        return err
    try:
        if x is not None and y is not None:
            pyautogui.click(x=x, y=y, clicks=clicks, interval=interval, button=button)
            return f"✅ Clicked {button} button at ({x}, {y})"
        pyautogui.click(clicks=clicks, interval=interval, button=button)
        return f"✅ Clicked {button} button at current position"
    except Exception as e:
        return f"❌ Error clicking: {e}"


@tool
def type_text(text: str, interval: float = 0.05) -> str:
    """Type text using the keyboard."""
    err = _validate_pyautogui()
    if err:
        return err
    try:
        pyautogui.write(text, interval=interval)
        return f"✅ Typed text ({len(text)} chars)"
    except Exception as e:
        return f"❌ Error typing text: {e}"


@tool
def scroll(clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> str:
    """Scroll the mouse wheel. Positive scrolls up, negative scrolls down."""
    err = _validate_pyautogui()
    if err:
        return err
    try:
        if x is not None and y is not None:
            pyautogui.scroll(clicks, x=x, y=y)
            return f"✅ Scrolled {clicks} clicks at ({x}, {y})"
        pyautogui.scroll(clicks)
        return f"✅ Scrolled {clicks} clicks at current position"
    except Exception as e:
        return f"❌ Error scrolling: {e}"


@tool
def hotkey(keys: str) -> str:
    """Press a sequence of keys as a hotkey combination."""
    err = _validate_pyautogui()
    if err:
        return err
    try:
        key_list = [k.strip() for k in re.split(r"[ ,]+", keys) if k.strip()]
        if not key_list:
            return "❌ No keys provided for hotkey."
        pyautogui.hotkey(*key_list)
        return f"✅ Pressed hotkey: {' + '.join(key_list)}"
    except Exception as e:
        return f"❌ Error pressing hotkey: {e}"


@tool
def drag(start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5, button: str = "left") -> str:
    """Drag the mouse from one coordinate to another."""
    err = _validate_pyautogui()
    if err:
        return err
    try:
        pyautogui.moveTo(start_x, start_y)
        pyautogui.dragTo(end_x, end_y, duration=duration, button=button)
        return f"✅ Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})"
    except Exception as e:
        return f"❌ Error dragging mouse: {e}"


COMPUTER_TOOLS = [
    move_mouse,
    click,
    type_text,
    scroll,
    hotkey,
    drag,
]

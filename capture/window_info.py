import sys
import ctypes
import os

def get_active_window_info() -> tuple[str | None, str | None]:
    """
    Returns the active window's title and process (application) name.
    Supports Windows via ctypes API calls. Falls back to default strings on other systems.
    """
    if sys.platform != "win32":
        return "Unknown Title", "Unknown App"

    try:
        # Get handle to foreground window
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if not hwnd:
            return None, None

        # 1. Retrieve the window title
        title = None
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buffer = ctypes.create_unicode_buffer(length + 1)
            if ctypes.windll.user32.GetWindowTextW(hwnd, buffer, length + 1) > 0:
                title = buffer.value

        # 2. Retrieve the process ID
        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        
        app_name = None
        if pid.value:
            # PROCESS_QUERY_LIMITED_INFORMATION (0x1000) is preferred as it works for protected/elevation contexts
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h_process = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
            if h_process:
                try:
                    size = ctypes.c_ulong(512)
                    buf = ctypes.create_unicode_buffer(size.value)
                    if ctypes.windll.kernel32.QueryFullProcessImageNameW(h_process, 0, buf, ctypes.byref(size)):
                        app_name = os.path.basename(buf.value)
                finally:
                    # Always release process handle
                    ctypes.windll.kernel32.CloseHandle(h_process)

        return title, app_name

    except Exception as e:
        # Prevent any win32 ctypes exception from crashing the pipeline
        print(f"Warning: Failed to retrieve active window info: {e}")
        return None, None

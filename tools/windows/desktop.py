"""Native FlowHMR window using the local WebView2 runtime."""

import argparse
import ctypes
from ctypes import wintypes
from html import escape
from http.client import HTTPConnection, HTTPException
import json
import logging
import os
from pathlib import Path
from secrets import token_urlsafe
import socket
import subprocess
import sys
import threading
from urllib.parse import urlsplit
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

import webview


ROOT = Path(__file__).resolve().parents[2]
TITLE = "FlowHMR"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from desktop_downloads import configure_downloads
from desktop_messages import MESSAGES, model_archive_filter, read_language, save_language
from desktop_update import DesktopUpdate


class _JobBasicLimits(ctypes.Structure):
    _fields_ = [
        ("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
        ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
        ("max_working_set", ctypes.c_size_t), ("active_processes", wintypes.DWORD),
        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
        ("scheduling", wintypes.DWORD),
    ]


class _JobExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", _JobBasicLimits), ("io_counters", ctypes.c_uint64 * 6),
        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t),
    ]


class DesktopBackend:
    """Own the inference processes launched by this window, including their workers."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.closed = threading.Event()
        self._lock = threading.Lock()
        self.name = f"Local\\FlowHMR-backend-{uuid4().hex}"
        self.character_key = token_urlsafe(32)
        kernel.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = (
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
        )
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        handle = kernel.CreateJobObjectW(None, self.name)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _JobExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            kernel.CloseHandle(handle)
            raise error
        self._handle = handle

    def close(self):
        with self._lock:
            self.closed.set()
            if self._handle and self.kernel.CloseHandle(self._handle):
                self._handle = None
                logging.info("Closed desktop backend process group")
            elif self._handle:
                logging.error("Could not close desktop backend job: %s", ctypes.WinError(ctypes.get_last_error()))

    def run_helper(self, args, *, timeout, **options):
        # These helpers do not launch the server. The starter joins before spawning it.
        with self._lock:
            if self.closed.is_set():
                return
            process = subprocess.Popen(args, creationflags=subprocess.CREATE_NO_WINDOW, **options)
            child = self.kernel.OpenProcess(0x0101, False, process.pid)  # SET_QUOTA | TERMINATE
            try:
                if not child or not self.kernel.AssignProcessToJobObject(self._handle, child):
                    error = ctypes.WinError(ctypes.get_last_error())
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                        raise error
            finally:
                if child:
                    self.kernel.CloseHandle(child)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        if code and not self.closed.is_set():
            raise subprocess.CalledProcessError(code, args)


def status_page(state):
    language = read_language()
    heading, detail = MESSAGES[language][state]
    return f"""<!doctype html><html lang="{language}"><meta charset="utf-8">
<style>
body {{margin:0;display:grid;place-items:center;min-height:100vh;background:#252525;color:#ededed;
       font:14px 'Segoe UI','Microsoft YaHei',sans-serif}}
main {{width:min(480px,85%);padding:24px;background:#303030;border:1px solid #454545;border-radius:5px}}
h1 {{font-size:18px;margin:0 0 12px}} p {{margin:0;color:#bbb;line-height:1.6}}
</style><main><h1>{escape(heading)}</h1><p>{escape(detail)}</p></main></html>"""


def open_models_folder():
    directory = ROOT / "models"
    directory.mkdir(exist_ok=True)
    os.startfile(directory)


def open_runtime_folder():
    directory = ROOT / "runtime/environments"
    directory.mkdir(parents=True, exist_ok=True)
    os.startfile(directory)


def create_file_picker(window, port, backend):
    """Open one native picker at a time for the local FlowHMR page."""
    origin = f"http://127.0.0.1:{port}"
    last_directory = Path.home() / "Downloads"
    picker_lock = threading.Lock()

    def is_flowhmr_page():
        if backend.closed.is_set():
            return False
        current = urlsplit(window.get_current_url() or "")
        return f"{current.scheme}://{current.netloc}" == origin

    def choose(dialog, file_types=()):
        nonlocal last_directory
        if not is_flowhmr_page() or not picker_lock.acquire(blocking=False):
            return None
        try:
            selection = window.create_file_dialog(dialog, directory=os.fspath(last_directory),
                                                  allow_multiple=False, file_types=file_types)
            if selection and is_flowhmr_page():
                path = Path(selection[0])
                last_directory = path if dialog == webview.FileDialog.FOLDER else path.parent
                return os.fspath(path)
            return None
        finally:
            picker_lock.release()

    return choose


def expose_model_pickers(window, port, backend, choose):
    origin = f"http://127.0.0.1:{port}"

    def choose_model_folder():
        return choose(webview.FileDialog.FOLDER)

    def choose_model_archive():
        return choose(webview.FileDialog.OPEN, (model_archive_filter(),))

    def choose_character_file():
        path = choose(webview.FileDialog.OPEN, ("PMX VRM FBX (*.pmx;*.vrm;*.fbx)",))
        if not path:
            return None
        payload = json.dumps({"file": path}).encode("utf-8")
        req = Request(origin + "/api/desktop/character-file", data=payload,
                      headers={"Content-Type": "application/json", "X-FlowHMR-Desktop-Key": backend.character_key})
        try:
            with urlopen(req, timeout=10) as result:
                return json.load(result)
        except HTTPError as error:
            # A 403 means the desktop key and the running backend are out of sync: a new
            # window attached to a backend that an older window started, so the key baked
            # into the backend's environment no longer matches backend.character_key.
            detail = error.read().decode("utf-8", "replace")[:200]
            logging.error("character-file registration failed: HTTP %s %s", error.code, detail)
            if error.code == 403:
                logging.warning("Desktop key rejected; restart FlowHMR so the backend picks up the current key.")
            return {"error": "character_file", "status": error.code, "detail": detail}
        except (OSError, ValueError):
            logging.exception("character-file registration failed")
            return {"error": "character_file"}

    window.expose(choose_model_folder, choose_model_archive, choose_character_file)


def expose_language_preference(window, port, backend):
    def is_flowhmr_page():
        current = urlsplit(window.get_current_url() or "")
        return not backend.closed.is_set() and (current.scheme, current.netloc) == ("http", f"127.0.0.1:{port}")

    def get_ui_language():
        return read_language() if is_flowhmr_page() else None

    def set_ui_language(language):
        if not is_flowhmr_page():
            return False
        try:
            return save_language(language)
        except OSError:
            logging.exception("Could not persist the launcher language")
            return False

    window.expose(get_ui_language, set_ui_language)


def expose_mcp(window, port, backend):
    def request(path, body=None):
        current = urlsplit(window.get_current_url() or "")
        if backend.closed.is_set() or (current.scheme, current.netloc) != ("http", f"127.0.0.1:{port}"):
            return None
        connection = HTTPConnection("127.0.0.1", port, timeout=1210 if body and body.get("action") == "stop" else 30)
        headers = {"X-FlowHMR-Desktop-Key": backend.character_key, "Content-Type": "application/json"}
        try:
            if body is None:
                connection.request("GET", path, headers=headers)
            else:
                connection.request("POST", path, json.dumps(body).encode("utf-8"), headers=headers)
            with connection.getresponse() as response:
                data = response.read()
                if response.status != 200 and not response.getheader("Content-Type", "").startswith("application/json"):
                    return {"error": f"HTTP {response.status}"}
                return json.loads(data)
        except (OSError, HTTPException, ValueError):
            logging.exception("Could not access the desktop MCP service")
            return None
        finally:
            connection.close()

    def mcp_settings(body=None):
        if body is None or isinstance(body, dict):
            return request("/api/mcp", body)

    def mcp_activity():
        return request("/api/mcp/activity")

    window.expose(mcp_settings, mcp_activity)


def expose_updates(window, port, backend, trial_ready, choose):
    updater = DesktopUpdate(ROOT)

    def local_page():
        current = urlsplit(window.get_current_url() or "")
        return not backend.closed.is_set() and (current.scheme, current.netloc) == ("http", f"127.0.0.1:{port}")

    def get_update_state():
        return updater.snapshot() if local_page() else None

    def check_updates(manual=False):
        return updater.check(manual is True) if local_page() else None

    def download_update():
        return updater.download() if local_page() else None

    def stop_update():
        return updater.cancel() if local_page() else None

    def manual_update():
        if not local_page() or not updater.current:
            return None
        path = choose(webview.FileDialog.OPEN, ("FlowHMR ZIP (*.zip)",))
        return updater.stage_local(path) if path else updater.snapshot()

    def select_update_version(version):
        return updater.select_version(version) if local_page() else None

    def confirm_update_boot():
        if trial_ready is None or trial_ready.is_set() or not local_page():
            return
        (updater.work / "applying/healthy").touch(exist_ok=True)
        trial_ready.set()

    window.expose(get_update_state, check_updates, download_update, stop_update, manual_update,
                  select_update_version, confirm_update_boot)


def connect(window, port, backend, *, trial=False):
    if backend.closed.is_set():
        return
    script = ROOT / "tools/windows/start-web.ps1"
    launcher_log = ROOT / "logs/desktop-launcher.log"
    try:
        logging.info("Connecting to local FlowHMR at port %d", port)
        # A newly launched server can inherit PowerShell's handles. A pipe would keep
        # communicate() waiting for EOF even after the launcher reported ready.
        with launcher_log.open("ab") as log:
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
                 "-ServerOnly", "-Port", str(port), "-DesktopJobName", backend.name],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "FLOWHMR_DESKTOP_CHARACTER_KEY": backend.character_key},
                creationflags=subprocess.CREATE_NO_WINDOW, timeout=310,
            )
        if backend.closed.is_set():
            return
        if result.returncode == 20:
            logging.warning("The previous FlowHMR service could not be replaced. See desktop-launcher.log.")
            if not trial:
                window.load_html(status_page("server_busy"))
            return False
        if result.returncode:
            raise RuntimeError(f"FlowHMR backend exited with code {result.returncode}")
        window.load_url(f"http://127.0.0.1:{port}/")
        logging.info("Loaded local FlowHMR at port %d", port)
        return True
    except Exception:
        if backend.closed.is_set():
            return
        logging.exception("Failed to connect to FlowHMR")
        if not trial:
            window.load_html(status_page("failed"))
        return False


def expose_runtime_restart(window, port, backend):
    restart_lock = threading.Lock()
    url = f"http://127.0.0.1:{port}"

    def reload_backend():
        try:
            if backend.closed.is_set():
                return
            window.load_html(status_page("switching"))
            with (ROOT / "logs/desktop-launcher.log").open("ab") as log:
                backend.run_helper(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                    str(ROOT / "tools/windows/stop-web.ps1")], cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=25)
            connect(window, port, backend)
        except (OSError, subprocess.SubprocessError):
            if not backend.closed.is_set():
                logging.exception("Could not restart FlowHMR for runtime switch")
                window.load_url(url + "/")
        finally:
            restart_lock.release()

    def restart_runtime():
        if backend.closed.is_set():
            return False
        current = urlsplit(window.get_current_url() or "")
        if f"{current.scheme}://{current.netloc}" != url:
            return False
        try:
            with urlopen(url + "/api/runtime", timeout=5) as response:
                pending = json.load(response).get("pending_restart", False)
        except (OSError, ValueError):
            return False
        if not pending or backend.closed.is_set() or not restart_lock.acquire(blocking=False):
            return False
        threading.Thread(target=reload_backend, name="flowhmr-runtime-restart", daemon=True).start()
        return True

    window.expose(restart_runtime)


def desktop_port(requested):
    # Keep the same origin for saved UI settings, but allow the portable package
    # to run beside a source checkout using the usual port.
    if requested != 60550 or not (ROOT / "runtime/python.exe").is_file():
        return requested
    marker = ROOT / ".cache/desktop-port.txt"
    try:
        previous = int(marker.read_text(encoding="ascii"))
    except (OSError, ValueError):
        previous = requested
    for port in dict.fromkeys((previous, requested, 0)):
        if not 0 <= port <= 65535:
            continue
        try:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", port))
                port = listener.getsockname()[1]
        except OSError:
            try:
                with urlopen(f"http://127.0.0.1:{port}/api/info", timeout=.8) as response:
                    directory = json.load(response)["model_files"]["directory"]
                if Path(directory).resolve() != (ROOT / "models").resolve():
                    continue
            except (OSError, ValueError, KeyError, TypeError):
                continue
        marker.parent.mkdir(exist_ok=True)
        marker.write_text(str(port), encoding="ascii")
        return port
    return requested


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    port = desktop_port(args.port)

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetConsoleWindow.restype = wintypes.HWND
    inherited_console = bool(kernel.GetConsoleWindow())
    if inherited_console:
        kernel.FreeConsole()
    kernel.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    instance = kernel.CreateMutexW(None, False, f"Local\\FlowHMR-desktop-{port}")
    if not instance:
        raise ctypes.WinError(ctypes.get_last_error())
    already_open = ctypes.get_last_error() == 183
    backend = None
    try:
        if already_open:
            user32 = ctypes.windll.user32
            user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
            user32.FindWindowW.restype = wintypes.HWND
            user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
            user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
            handle = user32.FindWindowW(None, TITLE)
            if handle:
                user32.ShowWindow(handle, 9)
                user32.SetForegroundWindow(handle)
            return

        log_dir = ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        logging.basicConfig(filename=log_dir / "desktop.log", encoding="utf-8", level=logging.INFO,
                            format="%(asctime)s %(message)s")
        logging.info("Desktop startup: inherited console=%s", inherited_console)
        backend = DesktopBackend(kernel)
        webview.settings["ALLOW_DOWNLOADS"] = True
        webview.settings["ALLOW_FILE_URLS"] = False
        webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
        webview.settings["SHOW_DEFAULT_MENUS"] = False
        fixed_webview = ROOT / "runtime/webview2"
        if (fixed_webview / "msedgewebview2.exe").is_file():
            webview.settings["WEBVIEW2_RUNTIME_PATH"] = os.fspath(fixed_webview)
        window = webview.create_window(
            TITLE, html=status_page("starting"),
            width=1440, height=900, min_size=(960, 640), background_color="#252525", text_select=True,
        )
        window.expose(open_models_folder)
        window.expose(open_runtime_folder)
        choose_file = create_file_picker(window, port, backend)
        expose_model_pickers(window, port, backend, choose_file)
        expose_language_preference(window, port, backend)
        expose_mcp(window, port, backend)
        expose_runtime_restart(window, port, backend)
        trial_ready = threading.Event() if os.environ.get("FLOWHMR_UPDATE_TRIAL") == "1" else None
        trial_failed = threading.Event()
        expose_updates(window, port, backend, trial_ready, choose_file)

        def start_workspace():
            opened = connect(window, port, backend, trial=trial_ready is not None)
            if trial_ready is not None and (not opened or not trial_ready.wait(45)) and not backend.closed.is_set():
                logging.error("Updated FlowHMR did not open the local workspace. Restoring the previous version.")
                trial_failed.set()
                window.destroy()

        window.events.before_show += lambda: configure_downloads(window, port, ROOT / ".cache/desktop-export-directory.txt")
        window.events.shown += lambda: threading.Thread(target=start_workspace, daemon=True).start()
        window.events.closed += backend.close
        webview.start(gui="edgechromium", private_mode=False,
                      storage_path=os.fspath(ROOT / ".cache/webview2"),
                      icon=os.fspath(ROOT / "assets/flowhmr.ico"))
        if trial_failed.is_set():
            raise SystemExit(1)
    finally:
        if backend is not None:
            backend.close()
        kernel.CloseHandle(instance)


if __name__ == "__main__":
    main()

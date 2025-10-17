import asyncio
import os
import sys
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()
ACTIVE = False  # simplest: single session at a time

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Room Scenario Game</title>
  <link rel="stylesheet" href="https://unpkg.com/xterm@5.3.0/css/xterm.css" />
  <style>
    html, body { height: 100%; margin: 0; background: #1e1e1e; color: #ddd; font-family: monospace; }
    #wrap { display: flex; flex-direction: column; height: 100%; }
    #header { padding: 8px; background: #2d2d2d; }
    #terminal { flex: 1; padding: 8px; }
    .note { color: #aaa; font-size: 12px; }
    /* Floating "Jump to prompt" button */
    #jump {
      position: fixed; right: 12px; bottom: 12px;
      padding: 6px 10px; background: #3a3a3a; color: #fff;
      border: 1px solid #555; border-radius: 4px; cursor: pointer;
      opacity: .85; z-index: 10; display: none;
    }
    #jump:hover { opacity: 1; }
  </style>
</head>
<body>
<div id="wrap">
  <div id="header">
    <div>Room Scenario Game (web)</div>
    <div class="note">Type responses and press Enter. Backspace works; no advanced line editing.</div>
  </div>
  <div id="terminal"></div>
</div>
<button id="jump" title="Go to the current prompt">Jump to prompt ⬇</button>

<script src="https://unpkg.com/xterm@5.3.0/lib/xterm.js"></script>
<script>
(async () => {
  const term = new Terminal({ cursorBlink: true, convertEol: true, cols: 100, rows: 30 });
  term.open(document.getElementById('terminal'));
  term.focus();

  const proto = (location.protocol === 'https:') ? 'wss' : 'ws';
  const ws = new WebSocket(proto + '://' + location.host + '/ws');

  const jumpBtn = document.getElementById('jump');
  jumpBtn.addEventListener('click', () => {
    term.scrollToBottom();
    term.focus();
    jumpBtn.style.display = 'none';
  });

  // After initial output starts, snap to the top so the intro is visible.
  let didSnapTop = false;
  function snapIntroSoon() {
    if (didSnapTop) return;
    didSnapTop = true;
    setTimeout(() => {
      term.scrollToTop();
      jumpBtn.style.display = 'block';
    }, 700);
  }

  ws.onopen = () => {
    term.write("Connected. Launching game...\\r\\n");
    snapIntroSoon();
  };

  ws.onmessage = (ev) => {
    term.write(ev.data);
  };

  ws.onclose = () => {
    term.write("\\r\\n[Connection closed]\\r\\n");
  };

  let buffer = "";
  term.onKey(e => {
    const ev = e.domEvent;
    const key = e.key;

    if (ev.key === 'Enter') {
      try { ws.send(buffer); } catch (e) {}
      term.write('\\r\\n');
      buffer = '';
    } else if (ev.key === 'Backspace') {
      if (buffer.length > 0) {
        buffer = buffer.slice(0, -1);
        term.write('\\b \\b');
      }
    } else if (key && key.length === 1) {
      buffer += key;
      term.write(key);
    }
  });
})();
</script>
</body>
</html>
"""

@app.get("/")
async def index():
    return HTMLResponse(INDEX_HTML)

import asyncio, os, sys
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    global ACTIVE
    if ACTIVE:
        await ws.send_text("Another session is running. Please try again later.\r\n")
        await ws.close()
        return
    ACTIVE = True

    try:
        if os.name != "nt":
            # POSIX: run under a PTY (best for interactive programs)
            import termios

            master_fd, slave_fd = os.openpty()

            # Turn off echo so we don't get double-echo in the browser
            attrs = termios.tcgetattr(slave_fd)
            attrs[3] &= ~termios.ECHO
            termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)

            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-u", "tom_test.py",
                stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            os.close(slave_fd)  # parent only needs the master side

            async def pty_to_ws():
                try:
                    while True:
                        # Read from PTY in a thread to avoid blocking the loop
                        data = await asyncio.to_thread(os.read, master_fd, 1024)
                        if not data:
                            break
                        await ws.send_text(data.decode("utf-8", "ignore"))
                finally:
                    try:
                        os.close(master_fd)
                    except Exception:
                        pass

            async def ws_to_pty():
                try:
                    while True:
                        msg = await ws.receive_text()
                        # CRLF tends to be safest for line input on TTYs
                        await asyncio.to_thread(os.write, master_fd, (msg + "\n").encode())
                except WebSocketDisconnect:
                    pass
                except Exception:
                    pass

            t1 = asyncio.create_task(pty_to_ws())
            t2 = asyncio.create_task(ws_to_pty())
            await asyncio.wait([t1, t2], return_when=asyncio.FIRST_COMPLETED)

            # Clean up the child
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3)
                except asyncio.TimeoutError:
                    proc.kill()
        else:
            # Windows fallback using pipes (works, but PTY would be better via pywinpty/conpty)
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-u", "tom_test.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            async def pipe_stdout():
                try:
                    # Read line-by-line so prompts and responses flush promptly
                    while True:
                        line = await proc.stdout.readline()
                        if not line:
                            break
                        await ws.send_text(line.decode("utf-8", "ignore"))
                except Exception:
                    pass

            async def pipe_input():
                try:
                    while True:
                        msg = await ws.receive_text()
                        proc.stdin.write((msg + "\n").encode())
                        await proc.stdin.drain()
                except WebSocketDisconnect:
                    pass
                except Exception:
                    pass
                finally:
                    try:
                        if proc.returncode is None:
                            proc.terminate()
                    except Exception:
                        pass

            t1 = asyncio.create_task(pipe_stdout())
            t2 = asyncio.create_task(pipe_input())
            await asyncio.wait([t1, t2], return_when=asyncio.FIRST_COMPLETED)

            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                proc.kill()
    finally:
        ACTIVE = False
        try:
            await ws.close()
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
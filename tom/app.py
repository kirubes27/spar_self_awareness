import asyncio
import os
import sys
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import textwrap
import re

WRAP_COLS = 130

def wrap90_line(line: str, width: int = WRAP_COLS) -> str:
    # leave ASCII rulers alone (====, ----, ****)
    if re.fullmatch(r"\s*[=\-\*_]{4,}\s*", line):
        return line

    # preserve trailing spaces
    trailing_spaces = len(line) - len(line.rstrip(" "))
    
    # preserve leading spaces
    stripped = line.lstrip(" ")
    indent_len = len(line) - len(stripped) - trailing_spaces
    indent = " " * indent_len
    w = max(10, width - indent_len)

    parts = textwrap.wrap(
        stripped.rstrip(" "),
        width=w,
        break_long_words=False,
        break_on_hyphens=False,
        expand_tabs=False,
    )
    if not parts:
        return indent + " " * trailing_spaces
    
    # Add trailing spaces back to the last line
    result = "\n".join(indent + p for p in parts)
    if trailing_spaces > 0:
        result += " " * trailing_spaces
    return result

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
    html, body { height: 100%; margin: 0; background: #1e1e1e; color: #ddd; font-family: monospace; overflow: hidden; }
    #wrap { display: flex; flex-direction: column; height: 100%; }
    #header { padding: 8px; background: #2d2d2d; }
    #terminal { flex: 1; padding: 8px; }
    .note { color: #aaa; font-size: 12px; }
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
  (() => {
    const hostDiv = document.getElementById('terminal');
    
    if (typeof Terminal === 'undefined') {
      hostDiv.innerHTML = '<div style="color:#f66">Failed to load xterm.js from CDN.</div>';
      return;
    }

    const COLS = 140;
    const term = new Terminal({
      cursorBlink: true,
      convertEol: true,
      cols: COLS,
      rows: 30,
      scrollback: 5000
    });

    term.open(hostDiv);
    term.focus();

    const proto = (location.protocol === 'https:') ? 'wss' : 'ws';
    const ws = new WebSocket(proto + '://' + location.host + '/ws?cols=' + COLS);

    const jumpBtn = document.getElementById('jump');
    jumpBtn.addEventListener('click', () => {
      term.scrollToBottom();
      term.focus();
      jumpBtn.style.display = 'none';
    });

    let didSnapTop = false;
    
    function snapIntroTop() {
      if (didSnapTop) return;
      didSnapTop = true;
      term.scrollToTop();
      
      // Use requestAnimationFrame to ensure page scroll happens after rendering
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          if (document.scrollingElement) {
            document.scrollingElement.scrollTop = 0;
          }
          window.scrollTo(0, 0);
        });
      });
      
      jumpBtn.style.display = 'block';
    }

    ws.addEventListener('open', () => {
      term.writeln('Connected. Launching game...');
    });

    ws.addEventListener('message', (ev) => {
      term.write(ev.data);
      
      // Detect when game is waiting for input (reached the action prompt)
      if (!didSnapTop && ev.data.includes('ACTION PHASE')) {
        // Small delay to ensure all related text has been written
        setTimeout(snapIntroTop, 50);
      }
    });

    let buffer = '';
    
    term.onKey(({ key, domEvent }) => {
      const ev = domEvent;
      
      if (ev.key === 'Enter') {
        try {
          ws.send(buffer);
        } catch {}
        term.writeln('');
        buffer = '';
      } else if (ev.key === 'Backspace') {
        if (buffer.length > 0) {
          buffer = buffer.slice(0, -1);
          term.write('\b \b');
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

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    qcols = ws.query_params.get("cols")
    wrap_cols = WRAP_COLS
    try:
        if qcols is not None:
            wrap_cols = max(20, min(300, int(qcols)))  # clamp
    except Exception:
      pass
    global ACTIVE
    if ACTIVE:
        await ws.send_text(wrap90_line("Another session is running. Please try again later.", width=wrap_cols) + "\n")
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
                # Incremental word-wrapping of the current logical line
                line_accum = ""   # current line content (no trailing '\n')
                rendered = ""     # what we've already sent for this line after wrapping
                try:
                    while True:
                        data = await asyncio.to_thread(os.read, master_fd, 4096)
                        if not data:
                            break

                        s = data.decode("utf-8", "replace")
                        s = s.replace("\r\n", "\n").replace("\r", "\n")

                        while s:
                            nl = s.find("\n")
                            if nl == -1:
                                # No newline: extend the logical line
                                line_accum += s
                                wrapped = wrap90_line(line_accum, width=wrap_cols)
                                # send only the new suffix
                                diff = wrapped[len(rendered):]
                                if diff:
                                    await ws.send_text(diff)
                                    rendered = wrapped
                                s = ""
                            else:
                                # Complete a logical line
                                segment = s[:nl]
                                s = s[nl+1:]
                                line_accum += segment
                                wrapped = wrap90_line(line_accum, width=wrap_cols)
                                diff = wrapped[len(rendered):]
                                if diff:
                                    await ws.send_text(diff)
                                await ws.send_text("\n")
                                line_accum = ""
                                rendered = ""
                finally:
                    try:
                        os.close(master_fd)
                    except Exception:
                        pass

            async def ws_to_pty():
                try:
                    while True:
                        msg = await ws.receive_text()
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
            # Windows fallback using pipes (PTY via conpty/pywinpty would be better)
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-u", "tom_test.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            async def pipe_stdout():
                line_accum = ""
                rendered = ""
                try:
                    while True:
                        chunk = await proc.stdout.read(4096)
                        if not chunk:
                            break

                        s = chunk.decode("utf-8", "replace")
                        s = s.replace("\r\n", "\n").replace("\r", "\n")

                        while s:
                            nl = s.find("\n")
                            if nl == -1:
                                line_accum += s
                                wrapped = wrap90_line(line_accum, width=wrap_cols)
                                diff = wrapped[len(rendered):]
                                if diff:
                                    await ws.send_text(diff)
                                    rendered = wrapped
                                s = ""
                            else:
                                segment = s[:nl]
                                s = s[nl+1:]
                                line_accum += segment
                                wrapped = wrap90_line(line_accum, width=wrap_cols)
                                diff = wrapped[len(rendered):]
                                if diff:
                                    await ws.send_text(diff)
                                await ws.send_text("\n")
                                line_accum = ""
                                rendered = ""
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
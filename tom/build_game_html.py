"""
Build script to create standalone HTML game file.
Uses separate script tags to avoid quote escaping issues.

Usage:
    python build_game_html.py scenarios_tmp.json game.html
"""

import json
import sys
import base64
from pathlib import Path

def read_python_module(filename):
    """Read a Python file and return its contents."""
    with open(filename, 'r') as f:
        return f.read()

def extract_game_logic(tom_test_content):
    """
    Extract the core game logic from tom_test.py.
    Excludes the if __name__ == "__main__" section.
    """
    lines = tom_test_content.split('\n')
    result_lines = []

    for line in lines:
        # Stop at the if __name__ == "__main__" section
        if line.strip().startswith('if __name__'):
            break
        result_lines.append(line)

    return '\n'.join(result_lines)

def build_html(scenario_file, output_file='game.html'):
    """Build the standalone HTML file."""

    print(f"Reading scenario file: {scenario_file}")
    with open(scenario_file, 'r') as f:
        scenarios_data = json.load(f)

    print("Reading Python modules...")
    tom_helpers_content = read_python_module('tom_helpers.py')
    game_ui_content = read_python_module('game_ui.py')
    tom_test_content = read_python_module('tom_test.py')

    # Extract just the game logic parts we need
    game_logic = extract_game_logic(tom_test_content)

    # Encode everything as base64 to completely avoid escaping
    scenarios_b64 = base64.b64encode(json.dumps(scenarios_data).encode('utf-8')).decode('ascii')
    tom_helpers_b64 = base64.b64encode(tom_helpers_content.encode('utf-8')).decode('ascii')
    game_ui_b64 = base64.b64encode(game_ui_content.encode('utf-8')).decode('ascii')
    game_logic_b64 = base64.b64encode(game_logic.encode('utf-8')).decode('ascii')

    # Create a single Python script that decodes and writes files, then imports
    loader_script = f'''
import json
import base64

# Decode embedded modules
scenarios_data = json.loads(base64.b64decode("{scenarios_b64}").decode('utf-8'))
tom_helpers_code = base64.b64decode("{tom_helpers_b64}").decode('utf-8')
game_ui_code = base64.b64decode("{game_ui_b64}").decode('utf-8')
game_logic_code = base64.b64decode("{game_logic_b64}").decode('utf-8')

# Write them as actual .py files so Python can import them normally
with open('tom_helpers.py', 'w') as f:
    f.write(tom_helpers_code)

with open('game_ui.py', 'w') as f:
    f.write(game_ui_code)

with open('tom_test.py', 'w') as f:
    f.write(game_logic_code)

with open('scenarios.json', 'w') as f:
    json.dump(scenarios_data, f)

# Hide loading message
from js import document
document.getElementById('loading').style.display = 'none'

# Now import normally - this will work because they're real files
from tom_test import play_game_cli
from game_ui import BrowserInterface

# Run the game - top-level await works here
await play_game_cli(scenario_file='scenarios.json', human_player=True, ui=BrowserInterface())
'''

    # Build HTML with the loader script
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Room Scenario Game</title>
    <link rel="stylesheet" href="https://pyscript.net/releases/2024.1.1/core.css">
    <script type="module" src="https://pyscript.net/releases/2024.1.1/core.js"></script>
    <style>
        body {{
            font-family: 'Courier New', monospace;
            background-color: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            max-width: 900px;
            margin: 0 auto;
        }}
        .container {{
            background-color: #252526;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .separator {{
            border-top: 2px solid #3e3e42;
            margin: 20px 0;
        }}
        .thick-separator {{
            border-top: 3px double #3e3e42;
            margin: 30px 0;
        }}
        .header {{
            color: #4ec9b0;
            font-weight: bold;
            margin: 15px 0;
        }}
        .score {{
            color: #569cd6;
            font-weight: bold;
        }}
        .scenario-text {{
            line-height: 1.6;
            margin: 15px 0;
            white-space: pre-wrap;
        }}
        .input-group {{
            margin: 20px 0;
        }}
        input[type="text"] {{
            width: 100%;
            padding: 10px;
            background-color: #3c3c3c;
            border: 1px solid #555;
            color: #d4d4d4;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            border-radius: 4px;
        }}
        input[type="text"]:focus {{
            outline: none;
            border-color: #007acc;
        }}
        button {{
            background-color: #0e639c;
            color: white;
            border: none;
            padding: 10px 20px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            cursor: pointer;
            border-radius: 4px;
            margin: 5px;
        }}
        button:hover {{
            background-color: #1177bb;
        }}
        button:disabled {{
            background-color: #555;
            cursor: not-allowed;
        }}
        .error {{
            color: #f48771;
        }}
        .success {{
            color: #4ec9b0;
        }}
        .action-output {{
            color: #ce9178;
        }}
        .answer-output {{
            color: #dcdcaa;
        }}
        .game-over {{
            color: #4ec9b0;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }}
        #loading {{
            text-align: center;
            padding: 50px;
            color: #4ec9b0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div id="loading">
            <h2>Room Scenario Game</h2>
            <p>Loading game (this takes a few seconds)...</p>
        </div>
        <div id="gameContent"></div>
    </div>

    <script type="py">
{loader_script}
    </script>
</body>
</html>
'''

    # Write output file
    print(f"Writing HTML file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    file_size_kb = len(html_content) / 1024
    print(f"Successfully created {output_file}")
    print(f"  File size: {file_size_kb:.1f} KB")
    print(f"\nShare this file with friends!")
    print(f"   They just need to open it in any web browser.")
    print(f"   No installation required!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_game_html.py <scenario_file.json> [output_file.html]")
        print("\nExample:")
        print("  python build_game_html.py scenarios_tmp.json game.html")
        sys.exit(1)

    scenario_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'game.html'

    if not Path(scenario_file).exists():
        print(f"Error: Scenario file '{scenario_file}' not found")
        sys.exit(1)

    build_html(scenario_file, output_file)

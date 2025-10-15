"""
UI abstraction layer for the Room Scenario Game.
Provides CLI and Browser interfaces with async support.
"""

from abc import ABC, abstractmethod
from typing import Optional
import asyncio

class GameUI(ABC):
    """Abstract base class for game UI."""
    
    @abstractmethod
    def display(self, text: str, style: Optional[str] = None):
        """Display text to the user. Style can be: 'header', 'score', 'error', 'success', 'action', 'answer'"""
        pass
    
    @abstractmethod
    async def get_input(self, prompt: str) -> str:
        """Get text input from user."""
        pass
    
    @abstractmethod
    async def wait_for_continue(self):
        """Wait for user to continue (press enter/click button)."""
        pass
    
    @abstractmethod
    def show_separator(self, thick: bool = False):
        """Show a visual separator."""
        pass


class CLIInterface(GameUI):
    """Command-line interface implementation."""
    
    def display(self, text: str, style: Optional[str] = None):
        print(text)
    
    async def get_input(self, prompt: str) -> str:
        # In CLI, we need to run input() in an executor to make it async
        # Use asyncio.to_thread if available (Python 3.9+), otherwise run_in_executor
        try:
            return await asyncio.to_thread(input, prompt)
        except AttributeError:
            # Fallback for older Python or Pyodide
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, input, prompt)
    
    async def wait_for_continue(self):
        try:
            await asyncio.to_thread(input, "\n[Press Enter to continue]")
        except AttributeError:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, input, "\n[Press Enter to continue]")
    
    def show_separator(self, thick: bool = False):
        if thick:
            print("=" * 70)
        else:
            print("***********************************")


class BrowserInterface(GameUI):
    """Browser interface implementation for PyScript."""
    
    def __init__(self):
        from js import document, window
        from pyodide.ffi import create_proxy
        self.document = document
        self.window = window
        self.output_div = document.getElementById('gameContent')
        self.create_proxy = create_proxy
    
    def display(self, text: str, style: Optional[str] = None):
        """Display text with optional styling."""
        style_class = {
            'header': 'header',
            'score': 'score',
            'error': 'error',
            'success': 'success',
            'action': 'action-output',
            'answer': 'answer-output'
        }.get(style, '')
        
        div = self.document.createElement('div')
        if style_class:
            div.className = style_class
        div.innerHTML = text.replace('\n', '<br>')
        self.output_div.appendChild(div)
        self.window.scrollTo(0, self.document.body.scrollHeight)
    
    async def get_input(self, prompt: str) -> str:
        """Get input from user via HTML input field."""
        # Create a future to wait for input
        future = asyncio.Future()
        
        # Create input group
        input_group = self.document.createElement('div')
        input_group.className = 'input-group'
        input_group.id = 'currentInputGroup'
        
        # Create input field
        input_field = self.document.createElement('input')
        input_field.type = 'text'
        input_field.placeholder = prompt
        input_field.id = 'actionInput'
        
        # Create submit button
        submit_btn = self.document.createElement('button')
        submit_btn.textContent = 'Submit'
        submit_btn.id = 'submitBtn'
        
        input_group.appendChild(input_field)
        input_group.appendChild(submit_btn)
        self.output_div.appendChild(input_group)
        
        # Set up event handlers
        def on_submit(event):
            value = input_field.value.strip()
            input_group.remove()
            if not future.done():
                future.set_result(value)
        
        submit_proxy = self.create_proxy(on_submit)
        submit_btn.addEventListener('click', submit_proxy)
        
        def on_enter(event):
            if event.key == 'Enter':
                on_submit(event)
        
        enter_proxy = self.create_proxy(on_enter)
        input_field.addEventListener('keypress', enter_proxy)
        
        input_field.focus()
        
        # Wait for the future to complete
        return await future
    
    async def wait_for_continue(self):
        """Wait for user to click continue button."""
        # Create a future to wait for click
        future = asyncio.Future()
        
        input_group = self.document.createElement('div')
        input_group.className = 'input-group'
        input_group.id = 'currentInputGroup'
        
        continue_btn = self.document.createElement('button')
        continue_btn.textContent = 'Press Enter to continue'
        continue_btn.id = 'continueBtn'
        
        input_group.appendChild(continue_btn)
        self.output_div.appendChild(input_group)
        
        def on_continue(event):
            input_group.remove()
            self.document.removeEventListener('keypress', enter_proxy)
            if not future.done():
                future.set_result(None)
        
        continue_proxy = self.create_proxy(on_continue)
        continue_btn.addEventListener('click', continue_proxy)
        
        def on_enter(event):
            if event.key == 'Enter':
                on_continue(event)
        
        enter_proxy = self.create_proxy(on_enter)
        self.document.addEventListener('keypress', enter_proxy)
        
        continue_btn.focus()
        
        # Wait for the future to complete
        await future
    
    def show_separator(self, thick: bool = False):
        """Show a visual separator."""
        div = self.document.createElement('div')
        div.className = 'thick-separator' if thick else 'separator'
        self.output_div.appendChild(div)
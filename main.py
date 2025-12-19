#!/usr/bin/env python3
"""
Turkish AI Terminal Assistant - Main Application
A command-line interface for an AI assistant with Turkish language support
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from typing import Optional, Dict, List, Callable
from pathlib import Path


class CommandHandler:
    """Handles command execution and routing"""
    
    def __init__(self):
        self.commands: Dict[str, Callable] = {}
        self._register_default_commands()
    
    def register(self, name: str, func: Callable, help_text: str = ""):
        """Register a new command"""
        self.commands[name] = {
            'function': func,
            'help': help_text
        }
    
    def _register_default_commands(self):
        """Register built-in commands"""
        self.register('help', self.cmd_help, 'Yardım menüsünü göster / Show help menu')
        self.register('clear', self.cmd_clear, 'Ekranı temizle / Clear screen')
        self.register('exit', self.cmd_exit, 'Çıkış / Exit application')
        self.register('quit', self.cmd_exit, 'Çıkış / Exit application')
        self.register('time', self.cmd_time, 'Geçerli zaman / Show current time')
        self.register('date', self.cmd_date, 'Geçerli tarih / Show current date')
        self.register('echo', self.cmd_echo, 'Metni tekrarla / Echo text')
        self.register('version', self.cmd_version, 'Sürüm bilgisi / Show version')
    
    def execute(self, command: str, args: List[str]) -> bool:
        """Execute a command with arguments"""
        if command not in self.commands:
            print(f"❌ Bilinmeyen komut: '{command}'")
            print(f"   Unknown command: '{command}'")
            print("💡 'help' yazarak yardım alabilirsiniz / Type 'help' for assistance")
            return False
        
        try:
            self.commands[command]['function'](args)
            return True
        except Exception as e:
            print(f"❌ Komut yürütme hatası / Command execution error: {e}")
            return False
    
    # Built-in command implementations
    def cmd_help(self, args: List[str]):
        """Display help menu"""
        print("\n" + "="*70)
        print("📚 TURKISH AI TERMINAL ASSISTANT - YARDIM / HELP")
        print("="*70)
        print("\n📋 Kullanılabilir Komutlar / Available Commands:\n")
        
        for cmd_name, cmd_info in sorted(self.commands.items()):
            print(f"  {cmd_name:<15} - {cmd_info['help']}")
        
        print("\n" + "="*70)
        print("💡 Komut Formatı / Command Format: command [arguments]")
        print("="*70 + "\n")
    
    def cmd_clear(self, args: List[str]):
        """Clear the terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
        print("Turkish AI Terminal Assistant started!")
        print("Başlamak için 'help' yazın / Type 'help' to get started")
    
    def cmd_exit(self, args: List[str]):
        """Exit the application"""
        print("\n👋 Hoşça kalın / Goodbye!")
        print("   Turkish AI Terminal Assistant kapanıyor / Shutting down...")
        sys.exit(0)
    
    def cmd_time(self, args: List[str]):
        """Show current time"""
        current_time = datetime.utcnow().strftime("%H:%M:%S")
        print(f"⏰ Geçerli Zaman (UTC): {current_time}")
        print(f"   Current Time (UTC): {current_time}")
    
    def cmd_date(self, args: List[str]):
        """Show current date"""
        current_date = datetime.utcnow().strftime("%Y-%m-%d")
        print(f"📅 Geçerli Tarih (UTC): {current_date}")
        print(f"   Current Date (UTC): {current_date}")
    
    def cmd_echo(self, args: List[str]):
        """Echo the provided text"""
        if not args:
            print("⚠️  Lütfen metin sağlayın / Please provide text")
            return
        text = ' '.join(args)
        print(f"🔊 {text}")
    
    def cmd_version(self, args: List[str]):
        """Show version information"""
        print("\n" + "="*70)
        print("📱 Turkish AI Terminal Assistant")
        print("   Sürüm / Version: 1.0.0")
        print("   Tarih / Date: 2025-12-19")
        print("   Geliştirici / Developer: Haidra-hub")
        print("="*70 + "\n")


class TerminalUI:
    """Manages the terminal user interface"""
    
    def __init__(self):
        self.command_handler = CommandHandler()
        self.running = True
        self.history: List[str] = []
    
    def print_welcome(self):
        """Print welcome banner"""
        print("\n" + "="*70)
        print("🤖 TURKISH AI TERMINAL ASSISTANT")
        print("   Türkçe AI Terminal Asistanı")
        print("="*70)
        print("✨ Hoşgeldiniz / Welcome!")
        print("📖 Yardım için 'help' yazın / Type 'help' for assistance")
        print("🚪 Çıkmak için 'exit' yazın / Type 'exit' to quit")
        print("="*70 + "\n")
    
    def get_prompt(self) -> str:
        """Get the formatted prompt string"""
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        return f"🤖 [{timestamp}] > "
    
    def parse_input(self, user_input: str) -> tuple[Optional[str], List[str]]:
        """Parse user input into command and arguments"""
        parts = user_input.strip().split()
        if not parts:
            return None, []
        return parts[0].lower(), parts[1:]
    
    def add_to_history(self, command: str):
        """Add command to history"""
        self.history.append(f"{datetime.utcnow().isoformat()} - {command}")
    
    def save_history(self, filepath: str = ".assistant_history"):
        """Save command history to file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for entry in self.history:
                    f.write(entry + '\n')
        except Exception as e:
            print(f"⚠️  Tarih kaydedilirken hata / Error saving history: {e}")
    
    def load_history(self, filepath: str = ".assistant_history"):
        """Load command history from file"""
        try:
            if Path(filepath).exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.history = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"⚠️  Tarih yüklenirken hata / Error loading history: {e}")
    
    def run(self):
        """Main application loop"""
        self.load_history()
        self.print_welcome()
        
        try:
            while self.running:
                try:
                    # Get user input
                    user_input = input(self.get_prompt()).strip()
                    
                    if not user_input:
                        continue
                    
                    # Parse and execute command
                    command, args = self.parse_input(user_input)
                    
                    if command:
                        self.add_to_history(user_input)
                        self.command_handler.execute(command, args)
                
                except KeyboardInterrupt:
                    print("\n\n⚠️  Ctrl+C basıldı / Ctrl+C pressed")
                    print("   Çıkmak için 'exit' yazın / Type 'exit' to quit")
                    continue
                except EOFError:
                    print("\n")
                    self.command_handler.cmd_exit([])
        
        finally:
            self.save_history()


def main():
    """Application entry point"""
    try:
        app = TerminalUI()
        app.run()
    except Exception as e:
        print(f"❌ Uygulama hatası / Application error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

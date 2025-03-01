#!/usr/bin/env python3
# debatebrawl_cli.py

import os
import sys
import json
import random
import re
import time
import textwrap
from typing import List, Dict, Any, Optional
import requests
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.table import Table
from rich.progress import Progress
from rich.rule import Rule

# ASCII art banner for DebateBrawl
DEBATE_BRAWL_BANNER = r"""
[bold cyan]
██████╗ ███████╗██████╗  █████╗ ████████╗███████╗
██╔══██╗██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██╔════╝
██║  ██║█████╗  ██████╔╝███████║   ██║   █████╗  
██║  ██║██╔══╝  ██╔══██╗██╔══██║   ██║   ██╔══╝  
██████╔╝███████╗██████╔╝██║  ██║   ██║   ███████╗
╚═════╝ ╚══════╝╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝
                                                                                          
██████╗ ██████╗  █████╗ ██╗    ██╗██╗     
██╔══██╗██╔══██╗██╔══██╗██║    ██║██║     
██████╔╝██████╔╝███████║██║ █╗ ██║██║     
██╔══██╗██╔══██╗██╔══██║██║███╗██║██║     
██████╔╝██║  ██║██║  ██║╚███╔███╔╝███████╗
╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚══════╝
[/bold cyan]
"""

class DebateBrawlCLI:
    def __init__(self):
        self.console = Console()
        self.base_url = "http://localhost:8000/api"
        self.user_id = None
        self.current_debate_id = None
    
    def display_welcome(self):
        """Display welcome screen with ASCII art banner"""
        os.system('clear' if os.name == 'posix' else 'cls')
        self.console.print(DEBATE_BRAWL_BANNER)
        self.console.print(Panel("[bold cyan]Welcome to DebateBrawl CLI[/bold cyan]", 
                                 subtitle="AI-Powered Debate Platform with Genetic Algorithms and Adversarial Search"))
        self.console.print("🎯 [cyan]Press Enter to continue[/cyan]")
        input()
    
    def login(self):
        """Simple login functionality"""
        os.system('clear' if os.name == 'posix' else 'cls')
        self.console.print(DEBATE_BRAWL_BANNER)
        
        self.console.print("[bold cyan]Login to DebateBrawl[/bold cyan]")
        
        username = Prompt.ask("[cyan]Enter username[/cyan]", default="debater")
        # Generate a deterministic user ID from username
        self.user_id = f"user_{hash(username) % 10000:04d}"
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Logging in...", total=100)
            for i in range(101):
                time.sleep(0.01)
                progress.update(task, completed=i)
        
        self.console.print(f"[green]Login successful. User ID: {self.user_id}[/green]")
        time.sleep(1.5)
    
    def get_topics(self) -> List[str]:
        """Get debate topics from the API"""
        try:
            response = requests.post(
                f"{self.base_url}/get_topics",
                json={"user_id": self.user_id}
            )
            if response.status_code == 200:
                return response.json()["topics"]
            else:
                self.console.print(f"[red]Error fetching topics: {response.status_code}[/red]")
                return []
        except Exception as e:
            self.console.print(f"[red]Error connecting to server: {str(e)}[/red]")
            return []
    
    def start_debate(self, topic: str, position: str) -> bool:
        """Start a new debate"""
        try:
            response = requests.post(
                f"{self.base_url}/start_debate",
                json={
                    "user_id": self.user_id,
                    "topic": topic,
                    "position": position
                }
            )
            if response.status_code == 200:
                data = response.json()
                self.current_debate_id = data["debate_id"]
                return True
            else:
                self.console.print(f"[red]Error starting debate: {response.status_code}[/red]")
                return False
        except Exception as e:
            self.console.print(f"[red]Error connecting to server: {str(e)}[/red]")
            return False

    def submit_argument(self, argument: str) -> Dict[str, Any]:
        """Submit an argument to the current debate"""
        if not self.current_debate_id:
            self.console.print("[red]No active debate[/red]")
            return {}
        
        try:
            response = requests.post(
                f"{self.base_url}/submit_argument",
                json={
                    "user_id": self.user_id,
                    "debate_id": self.current_debate_id,
                    "argument": argument
                }
            )
            if response.status_code == 200:
                return response.json()
            else:
                self.console.print(f"[red]Error submitting argument: {response.status_code}[/red]")
                return {}
        except Exception as e:
            self.console.print(f"[red]Error connecting to server: {str(e)}[/red]")
            return {}
    
    def get_debate_state(self) -> Dict[str, Any]:
        """Get the current state of the debate"""
        if not self.current_debate_id:
            self.console.print("[red]No active debate[/red]")
            return {}
        
        try:
            response = requests.get(
                f"{self.base_url}/debate_state/{self.current_debate_id}"
            )
            if response.status_code == 200:
                return response.json()
            else:
                self.console.print(f"[red]Error getting debate state: {response.status_code}[/red]")
                return {}
        except Exception as e:
            self.console.print(f"[red]Error connecting to server: {str(e)}[/red]")
            return {}
    
    def display_debate_header(self, topic: str, position: str, round_num: int, max_rounds: int):
        """Display debate header information"""
        self.console.print(Panel(
            f"[bold cyan]Topic:[/bold cyan] {topic}\n"
            f"[bold cyan]Your Position:[/bold cyan] {position}\n"
            f"[bold cyan]Round:[/bold cyan] {round_num}/{max_rounds}",
            title="[bold cyan]Current Debate[/bold cyan]"
        ))
    
    def display_scores(self, user_score: float, ai_score: float):
        """Display current scores"""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Player", style="cyan")
        table.add_column("Score", justify="right")
        
        table.add_row("You", f"{user_score:.1f}")
        table.add_row("AI", f"{ai_score:.1f}")
        
        self.console.print(table)
    
    def display_suggestions(self, suggestions: List[str]):
        """Display argument suggestions"""
        if not suggestions:
            return
            
        panels = []
        for i, suggestion in enumerate(suggestions):
            # Format long suggestions properly with wrapping
            wrapped_text = textwrap.fill(suggestion, width=80)
            panels.append(f"[cyan]{i+1}.[/cyan] {wrapped_text}")
        
        # Join suggestions with newlines
        content = "\n\n".join(panels)
        
        self.console.print(Panel(
            content,
            title="[bold cyan]Argument Suggestions[/bold cyan]",
            expand=False
        ))
    
    def display_ai_response(self, ai_argument: str):
        """Display AI's response"""
        wrapped_text = textwrap.fill(ai_argument, width=80)
        self.console.print(Panel(
            wrapped_text,
            title="[bold cyan]AI's Response[/bold cyan]",
            border_style="cyan"
        ))
    
    def display_strategy_info(self, ga_strategy, as_prediction, debate_data=None):
        """Display strategy information with GA and AS metrics"""
        ga_content = textwrap.fill(ga_strategy, width=80) if ga_strategy else "No strategy available"
        as_content = as_prediction if as_prediction else "No prediction available"
        
        # Extract GA/AS influence metrics if available in debate_data
        metrics_content = ""
        if debate_data and "evolution_stats" in debate_data:
            evolution = debate_data.get("evolution_stats", {})
            pre = evolution.get("pre_strategy", {})
            post = evolution.get("post_strategy", {})
            
            if pre and post:
                # Calculate rhetorical element changes
                ethos_pre = pre.get("ethos", 0) * 100
                ethos_post = post.get("ethos", 0) * 100
                pathos_pre = pre.get("pathos", 0) * 100
                pathos_post = post.get("pathos", 0) * 100
                logos_pre = pre.get("logos", 0) * 100
                logos_post = post.get("logos", 0) * 100
                
                metrics_content = (
                    "\n\n[bold cyan]GA Influence Metrics:[/bold cyan]\n"
                    f"Ethos: {ethos_pre:.1f}% → {ethos_post:.1f}% ({ethos_post-ethos_pre:+.1f}%)\n"
                    f"Pathos: {pathos_pre:.1f}% → {pathos_post:.1f}% ({pathos_post-pathos_pre:+.1f}%)\n"
                    f"Logos: {logos_pre:.1f}% → {logos_post:.1f}% ({logos_post-logos_pre:+.1f}%)\n"
                )
                
                # Add tactics changes if available
                pre_tactics = ", ".join(pre.get("tactics", []))
                post_tactics = ", ".join(post.get("tactics", []))
                if pre_tactics or post_tactics:
                    metrics_content += f"\nTactics Evolution:\n{pre_tactics} → {post_tactics}\n"
        
        # Add AS metrics if available
        if debate_data and "as_model_stats" in debate_data:
            as_stats = debate_data.get("as_model_stats", {})
            confidence = as_stats.get("confidence_trend", [])
            
            if confidence:
                metrics_content += (
                    "\n[bold cyan]AS Influence Metrics:[/bold cyan]\n"
                    f"Prediction Confidence: {confidence[-1]:.2f}\n"
                )
        
        # Build the full content
        content = f"[bold cyan]GA Strategy:[/bold cyan]\n{ga_content}\n\n[bold cyan]AS Prediction:[/bold cyan]\n{as_content}{metrics_content}"
        
        self.console.print(Panel(content, title="[bold cyan]Strategy Information[/bold cyan]"))

    def main_menu(self):
        """Display main menu and handle user choices"""
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            self.console.print(DEBATE_BRAWL_BANNER)
            
            table = Table(show_header=False, box=None)
            table.add_column("", style="cyan")
            table.add_column("")
            
            table.add_row("[1]", "Start New Debate")
            table.add_row("[2]", "View Stats")
            table.add_row("[3]", "About")
            table.add_row("[4]", "Exit")
            
            self.console.print(table)
            
            choice = Prompt.ask("[cyan]Select an option[/cyan]", choices=["1", "2", "3", "4"])
            
            if choice == "1":
                self.start_new_debate_flow()
            elif choice == "2":
                self.view_stats()
            elif choice == "3":
                self.about()
            elif choice == "4":
                self.console.print("[cyan]Thanks for using DebateBrawl CLI![/cyan]")
                sys.exit(0)
    
    def start_new_debate_flow(self):
        """Flow for starting a new debate"""
        os.system('clear' if os.name == 'posix' else 'cls')
        self.console.print(Panel("[bold cyan]Start New Debate[/bold cyan]"))
        
        # Fetch topics
        with Progress() as progress:
            task = progress.add_task("[cyan]Fetching topics...", total=100)
            for i in range(101):
                time.sleep(0.01)
                progress.update(task, completed=i)
        
        topics = self.get_topics()
        
        if not topics:
            self.console.print("[red]No topics available[/red]")
            input("\nPress Enter to return to main menu...")
            return
        
        # Display topics
        table = Table(show_header=False)
        table.add_column("[cyan]#[/cyan]", style="cyan")
        table.add_column("Topic")
        
        for i, topic in enumerate(topics):
            table.add_row(f"[{i+1}]", topic)
        
        self.console.print(table)
        
        topic_choice = Prompt.ask(
            "[cyan]Select a topic number[/cyan]", 
            choices=[str(i+1) for i in range(len(topics))]
        )
        selected_topic = topics[int(topic_choice) - 1]
        
        # Select position
        self.console.print("\n[bold cyan]Choose your position:[/bold cyan]")
        position = Prompt.ask(
            "Position", 
            choices=["for", "against"]
        )
        
        # Start debate
        self.console.print(f"\nStarting debate on: [bold]{selected_topic}[/bold] (Position: {position})")
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Initializing debate...", total=100)
            for i in range(101):
                time.sleep(0.01)
                progress.update(task, completed=i)
        
        if self.start_debate(selected_topic, position):
            self.debate_session()
        else:
            self.console.print("[red]Failed to start debate[/red]")
            input("\nPress Enter to return to main menu...")
    
    def debate_session(self):
        """Main debate session loop"""
        if not self.current_debate_id:
            self.console.print("[red]No active debate[/red]")
            return
        
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Get current debate state
            debate_state = self.get_debate_state()
            if not debate_state:
                self.console.print("[red]Error retrieving debate state[/red]")
                input("\nPress Enter to return to main menu...")
                return
            
            # Check if debate is completed
            if debate_state.get("status") == "completed":
                self.debate_completed(debate_state)
                return
            
            # Display debate information
            self.display_debate_header(
                debate_state.get("topic", ""),
                debate_state.get("user_position", ""),
                debate_state.get("current_round", 1),
                debate_state.get("max_rounds", 5)
            )
            
            # Display scores
            scores = debate_state.get("scores", {})
            self.display_scores(
                scores.get("user", 0),
                scores.get("ai", 0)
            )
            
            # Display previous arguments if any
            if debate_state.get("current_round", 1) > 1:
                prev_round = debate_state.get("current_round", 1) - 1
                arguments = debate_state.get("arguments", {})
                
                # Find the previous round's arguments
                for round_key, args in arguments.items():
                    if round_key.endswith(str(prev_round)):
                        ai_arg = args.get("ai", "")
                        if ai_arg:
                            self.console.print("\n[bold cyan]Previous AI Response:[/bold cyan]")
                            self.display_ai_response(ai_arg)
                        break
            
            # Display suggestions
            suggestions = debate_state.get("llm_suggestions", [])
            if suggestions:
                self.console.print("\n[bold cyan]Argument Suggestions:[/bold cyan]")
                self.display_suggestions(suggestions)
            
            # Display strategy info
            ga_strategy = debate_state.get("ga_strategy", "")
            as_prediction = debate_state.get("as_prediction", "")
            if ga_strategy or as_prediction:
                self.console.print("\n[bold cyan]Strategy Information:[/bold cyan]")
                self.display_strategy_info(ga_strategy, as_prediction)
            
            # Get user argument
            self.console.print(Rule(style="cyan"))
            self.console.print("\n[bold cyan]Your turn. Enter your argument:[/bold cyan]")
            argument = ""
            while not argument.strip():
                argument = self.console.input("[cyan]> [/cyan]")
                if not argument.strip():
                    self.console.print("[yellow]Argument cannot be empty[/yellow]")
            
            # Submit argument
            self.console.print("\nSubmitting argument...")
            
            with Progress() as progress:
                task = progress.add_task("[cyan]Processing...", total=100)
                for i in range(101):
                    time.sleep(0.01)
                    progress.update(task, completed=i)
            
            result = self.submit_argument(argument)
            
            if not result:
                self.console.print("[red]Failed to submit argument[/red]")
                input("\nPress Enter to return to main menu...")
                return
            
            # Display AI response
            ai_argument = result.get("ai_argument", "")
            if ai_argument:
                self.console.print("\n[bold cyan]AI's Response:[/bold cyan]")
                self.display_ai_response(ai_argument)
            
            # Display round scores
            self.console.print(f"\n[bold cyan]Round Scores:[/bold cyan]")
            self.console.print(f"You: +{result.get('score', 0):.1f}  |  AI: +{result.get('ai_score', 0):.1f}")
            
            # Display evaluation feedback
            feedback = result.get("evaluation_feedback", "")
            if feedback:
                wrapped_feedback = textwrap.fill(feedback, width=80)
                self.console.print(Panel(
                    wrapped_feedback,
                    title="[bold cyan]Feedback on Your Argument[/bold cyan]"
                ))
            
            # Check if we've reached the final round
            current_round = result.get('current_round', 1)
            max_rounds = result.get('max_rounds', 5)
            
            if current_round > max_rounds:
                # Debate is complete
                self.debate_completed(debate_state)
                return
            
            input("\nPress Enter to continue to next round...")
    
    def debate_completed(self, debate_state: Dict[str, Any]):
        """Handle completed debate"""
        os.system('clear' if os.name == 'posix' else 'cls')
        self.console.print(Panel("[bold cyan]Debate Completed[/bold cyan]"))
        
        # Display final scores
        scores = debate_state.get("scores", {})
        user_score = scores.get("user", 0)
        ai_score = scores.get("ai", 0)
        
        self.console.print(f"[bold cyan]Final Scores:[/bold cyan]")
        self.display_scores(user_score, ai_score)
        
        # Determine winner
        if user_score > ai_score:
            self.console.print(Panel("[bold green]You win![/bold green]", border_style="green"))
        elif user_score < ai_score:
            self.console.print(Panel("[bold red]AI wins![/bold red]", border_style="red"))
        else:
            self.console.print(Panel("[bold yellow]It's a tie![/bold yellow]", border_style="yellow"))
        
        # Reset current debate
        self.current_debate_id = None
        
        input("\nPress Enter to return to main menu...")
    
    def view_stats(self):
        """View user statistics"""
        os.system('clear' if os.name == 'posix' else 'cls')
        self.console.print(Panel("[bold cyan]User Statistics[/bold cyan]"))
        
        try:
            response = requests.get(f"{self.base_url}/user_stats/{self.user_id}")
            if response.status_code == 200:
                stats = response.json()
                
                table = Table(show_header=True, header_style="bold cyan")
                table.add_column("Statistic", style="cyan")
                table.add_column("Value", justify="right")
                
                table.add_row("Total Debates", str(stats.get("totalDebates", 0)))
                table.add_row("Wins", str(stats.get("wins", 0)))
                table.add_row("Losses", str(stats.get("losses", 0)))
                table.add_row("Draws", str(stats.get("draws", 0)))
                table.add_row("Free Debates Left", str(stats.get("remainingFreeDebates", 0)))
                
                self.console.print(table)
            else:
                self.console.print(f"[red]Error fetching stats: {response.status_code}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error connecting to server: {str(e)}[/red]")
        
        input("\nPress Enter to return to main menu...")
    
    def about(self):
        """Display information about DebateBrawl"""
        os.system('clear' if os.name == 'posix' else 'cls')
        self.console.print(Panel("[bold cyan]About DebateBrawl[/bold cyan]"))
        
        about_text = """
        DebateBrawl is an AI-powered debate platform that integrates Large Language Models (LLMs) 
        with Genetic Algorithms (GA) and Adversarial Search (AS) to create an adaptive and 
        engaging debating experience.
        
        The system demonstrates remarkable performance in generating coherent, contextually 
        relevant arguments while adapting its strategy in real-time.
        
        - GA Component: Evolves debate strategies over time based on performance
        - AS Component: Predicts and counters opponent moves using game theory
        - LLM Integration: Uses Ollama to run local language models for all aspects of debate
        
        This CLI version provides access to the core functionality of DebateBrawl directly
        from your terminal.
        """
        
        self.console.print(textwrap.dedent(about_text))
        input("\nPress Enter to return to main menu...")
    
    def run(self):
        """Run the CLI application"""
        self.display_welcome()
        self.login()
        self.main_menu()


if __name__ == "__main__":
    cli = DebateBrawlCLI()
    cli.run()
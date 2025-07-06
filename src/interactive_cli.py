"""
Interactive CLI Helper Functions
Provides interactive prompts using questionary library for better user experience.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

try:
    import questionary  # type: ignore
    QUESTIONARY_AVAILABLE = True
except ImportError:
    QUESTIONARY_AVAILABLE = False
    print("Warning: questionary not available. Install with: pip install questionary")


class InteractiveCLI:
    """Interactive CLI helper class for Pixel Life commands."""
    
    def __init__(self):
        self.questionary_available = QUESTIONARY_AVAILABLE
    
    def _check_questionary(self) -> bool:
        """Check if questionary is available."""
        if not self.questionary_available:
            print("❌ Interactive mode requires questionary. Install with: pip install questionary")
            return False
        return True
    
    def select_size(self, default: int = 30) -> int:
        """Interactive size selection."""
        if not self._check_questionary():
            return default
        
        choices = [
            {"name": "Small (20x20)", "value": 20},
            {"name": "Medium (30x30)", "value": 30},
            {"name": "Large (50x50)", "value": 50},
            {"name": "Extra Large (100x100)", "value": 100},
            {"name": "Custom size", "value": "custom"}
        ]
        
        result = questionary.select(
            "Select environment size:",
            choices=choices,
            default=choices[1]  # Medium
        ).ask()
        
        if result == "custom":
            custom_size = questionary.text(
                "Enter custom size (10-200):",
                validate=lambda text: text.isdigit() and 10 <= int(text) <= 200
            ).ask()
            return int(custom_size)
        
        return result
    
    def select_steps(self, default: int = 200) -> int:
        """Interactive steps selection."""
        if not self._check_questionary():
            return default
        
        choices = [
            {"name": "Short (100 steps)", "value": 100},
            {"name": "Medium (200 steps)", "value": 200},
            {"name": "Long (500 steps)", "value": 500},
            {"name": "Very Long (1000 steps)", "value": 1000},
            {"name": "Custom steps", "value": "custom"}
        ]
        
        result = questionary.select(
            "Select number of simulation steps:",
            choices=choices,
            default=choices[1]  # Medium
        ).ask()
        
        if result == "custom":
            custom_steps = questionary.text(
                "Enter custom number of steps (10-10000):",
                validate=lambda text: text.isdigit() and 10 <= int(text) <= 10000
            ).ask()
            return int(custom_steps)
        
        return result
    
    def confirm_render(self, default: bool = False) -> bool:
        """Interactive render confirmation."""
        if not self._check_questionary():
            return default
        
        return questionary.confirm(
            "Enable visual rendering?",
            default=default
        ).ask()
    
    def select_device(self, available_devices: Optional[List[str]] = None) -> str:
        """Interactive device selection."""
        if not self._check_questionary():
            return "cpu"
        
        if available_devices is None:
            available_devices = ["cpu"]
            try:
                import torch
                if torch.cuda.is_available():
                    available_devices.append("cuda")
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    available_devices.append("mps")
            except ImportError:
                pass
        
        choices = [{"name": f"{device.upper()}", "value": device} for device in available_devices]
        
        return questionary.select(
            "Select computing device:",
            choices=choices,
            default=choices[0]
        ).ask()
    
    def select_training_params(self) -> Dict[str, Any]:
        """Interactive training parameters selection."""
        if not self._check_questionary():
            return {}
        
        params = {}
        
        # Timesteps
        timestep_choices = [
            {"name": "Quick (50K steps)", "value": 50000},
            {"name": "Standard (100K steps)", "value": 100000},
            {"name": "Thorough (500K steps)", "value": 500000},
            {"name": "Extensive (1M steps)", "value": 1000000},
            {"name": "Custom", "value": "custom"}
        ]
        
        timesteps = questionary.select(
            "Select training duration:",
            choices=timestep_choices
        ).ask()
        
        if timesteps == "custom":
            timesteps = int(questionary.text(
                "Enter custom timesteps (1000-10000000):",
                validate=lambda text: text.isdigit() and 1000 <= int(text) <= 10000000
            ).ask())
        
        params['timesteps'] = timesteps
        
        # Learning rate
        lr_choices = [
            {"name": "Conservative (1e-4)", "value": 1e-4},
            {"name": "Standard (3e-4)", "value": 3e-4},
            {"name": "Aggressive (5e-4)", "value": 5e-4},
            {"name": "Custom", "value": "custom"}
        ]
        
        lr = questionary.select(
            "Select learning rate:",
            choices=lr_choices,
            default=lr_choices[1]
        ).ask()
        
        if lr == "custom":
            lr = float(questionary.text(
                "Enter custom learning rate (1e-5 to 1e-3):",
                validate=lambda text: text.replace('.', '').replace('e', '').replace('-', '').isdigit() and 1e-5 <= float(text) <= 1e-3
            ).ask())
        
        params['learning_rate'] = lr
        
        # Number of environments
        n_envs = questionary.select(
            "Select number of parallel environments:",
            choices=[
                {"name": "Single (1)", "value": 1},
                {"name": "Few (4)", "value": 4},
                {"name": "Many (8)", "value": 8},
                {"name": "Custom", "value": "custom"}
            ],
            default={"name": "Few (4)", "value": 4}
        ).ask()
        
        if n_envs == "custom":
            n_envs = int(questionary.text(
                "Enter custom number of environments (1-16):",
                validate=lambda text: text.isdigit() and 1 <= int(text) <= 16
            ).ask())
        
        params['n_envs'] = n_envs
        
        return params
    
    def select_model(self, model_dir: str = "./models") -> Optional[str]:
        """Interactive model selection."""
        if not self._check_questionary():
            return None
        
        # Find available models
        available_models = []
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.zip'):
                    available_models.append(file)
        
        if not available_models:
            print("No models found in models directory")
            return None
        
        choices = [{"name": model, "value": os.path.join(model_dir, model)} for model in available_models]
        choices.append({"name": "Browse for model file...", "value": "browse"})
        
        result = questionary.select(
            "Select a model to load:",
            choices=choices
        ).ask()
        
        if result == "browse":
            result = questionary.path(
                "Select model file:",
                only_files=True,
                file_filter=lambda path: path.endswith('.zip')
            ).ask()
        
        return result
    
    def create_experiment_interactive(self) -> Dict[str, Any]:
        """Interactive experiment creation."""
        if not self._check_questionary():
            return {}
        
        experiment = {}
        
        # Name
        experiment['name'] = questionary.text(
            "Enter experiment name:",
            validate=lambda text: len(text.strip()) > 0
        ).ask()
        
        # Description
        experiment['description'] = questionary.text(
            "Enter experiment description (optional):"
        ).ask()
        
        # Tags
        tags_input = questionary.text(
            "Enter tags (comma-separated, optional):"
        ).ask()
        experiment['tags'] = [tag.strip() for tag in tags_input.split(',') if tag.strip()] if tags_input else []
        
        # Hyperparameters
        add_hyperparams = questionary.confirm(
            "Add hyperparameters?",
            default=False
        ).ask()
        
        if add_hyperparams:
            experiment['hyperparameters'] = {}
            while True:
                param_name = questionary.text(
                    "Enter parameter name (or 'done' to finish):"
                ).ask()
                
                if param_name.lower() == 'done':
                    break
                
                param_value = questionary.text(
                    f"Enter value for {param_name}:"
                ).ask()
                
                # Try to convert to appropriate type
                try:
                    if param_value.lower() in ['true', 'false']:
                        param_value = param_value.lower() == 'true'
                    elif '.' in param_value:
                        param_value = float(param_value)
                    else:
                        param_value = int(param_value)
                except ValueError:
                    pass  # Keep as string
                
                experiment['hyperparameters'][param_name] = param_value
        
        return experiment
    
    def select_log_level(self) -> str:
        """Interactive log level selection."""
        if not self._check_questionary():
            return "INFO"
        
        return questionary.select(
            "Select log level:",
            choices=[
                {"name": "DEBUG - Detailed information", "value": "DEBUG"},
                {"name": "INFO - General information", "value": "INFO"},
                {"name": "WARNING - Warning messages", "value": "WARNING"},
                {"name": "ERROR - Error messages only", "value": "ERROR"}
            ],
            default={"name": "INFO - General information", "value": "INFO"}
        ).ask()
    
    def select_workflow(self) -> str:
        """Interactive workflow selection."""
        if not self._check_questionary():
            return "train-multiple"
        
        return questionary.select(
            "Select workflow:",
            choices=[
                {"name": "Train Multiple Models", "value": "train-multiple"},
                {"name": "Evaluate All Models", "value": "evaluate-all"}
            ]
        ).ask()
    
    def confirm_action(self, action: str, details: str = "") -> bool:
        """Interactive action confirmation."""
        if not self._check_questionary():
            return True
        
        message = f"Confirm {action}?"
        if details:
            message += f"\n{details}"
        
        return questionary.confirm(
            message,
            default=False
        ).ask()


# Global instance
interactive_cli = InteractiveCLI()


def is_interactive_mode(args) -> bool:
    """Check if interactive mode should be used."""
    # Check if --no-interactive flag is set
    if hasattr(args, 'no_interactive') and args.no_interactive:
        return False
    
    # Check if questionary is available
    if not QUESTIONARY_AVAILABLE:
        return False
    
    # Check if running in non-interactive environment (CI, etc.)
    if not sys.stdin.isatty():
        return False
    
    return True 
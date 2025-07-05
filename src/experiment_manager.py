"""
Experiment Manager for Pixel Life CLI
Handles experiment creation, tracking, and metadata management.
"""

import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any


@dataclass
class ExperimentMetadata:
    """Experiment metadata structure."""
    experiment_id: str
    name: str
    description: str
    status: str
    created_at: str
    updated_at: str
    tags: List[str]
    hyperparameters: Dict[str, Any]
    results: Dict[str, Any]
    parent_experiment: Optional[str]


class ExperimentManager:
    """Manages experiment creation and metadata."""
    
    def __init__(self, experiments_dir: str = "./experiments"):
        self.experiments_dir = experiments_dir
        self.metadata_file = os.path.join(experiments_dir, "experiments.json")
        self._ensure_directory()
        self._load_metadata()
    
    def _ensure_directory(self):
        """Ensure experiments directory exists."""
        os.makedirs(self.experiments_dir, exist_ok=True)
    
    def _load_metadata(self):
        """Load experiment metadata from file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.experiments = {exp_id: ExperimentMetadata(**metadata) 
                                      for exp_id, metadata in data.items()}
            except Exception:
                self.experiments = {}
        else:
            self.experiments = {}
    
    def _save_metadata(self):
        """Save experiment metadata to file."""
        data = {exp_id: asdict(metadata) 
               for exp_id, metadata in self.experiments.items()}
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _generate_id(self, name: str) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{name.lower().replace(' ', '_')}_{timestamp}"
    
    def create_experiment(self, name: str, description: str = "",
                         hyperparameters: Optional[Dict[str, Any]] = None,
                         tags: Optional[List[str]] = None,
                         parent_experiment: Optional[str] = None) -> str:
        """Create a new experiment."""
        # Generate unique ID
        experiment_id = self._generate_id(name)
        
        # Create experiment directory
        exp_dir = os.path.join(self.experiments_dir, experiment_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Create metadata
        now = datetime.now().isoformat()
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            name=name,
            description=description,
            status="created",
            created_at=now,
            updated_at=now,
            tags=tags or [],
            hyperparameters=hyperparameters or {},
            results={},
            parent_experiment=parent_experiment
        )
        
        # Save metadata
        self.experiments[experiment_id] = metadata
        self._save_metadata()
        
        return experiment_id
    
    def list_experiments(self, status: Optional[str] = None, 
                        tags: Optional[List[str]] = None) -> List[ExperimentMetadata]:
        """List all experiments, optionally filtered by status and tags."""
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [exp for exp in experiments if exp.status == status]
        
        if tags:
            experiments = [exp for exp in experiments 
                          if any(tag in exp.tags for tag in tags)]
        
        return sorted(experiments, key=lambda x: x.created_at, reverse=True)
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Get experiment metadata by ID."""
        return self.experiments.get(experiment_id)
    
    def update_experiment(self, experiment_id: str, **kwargs):
        """Update experiment metadata."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        for key, value in kwargs.items():
            if hasattr(experiment, key):
                setattr(experiment, key, value)
        
        experiment.updated_at = datetime.now().isoformat()
        self._save_metadata()
    
    def delete_experiment(self, experiment_id: str):
        """Delete an experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Delete experiment directory
        exp_dir = os.path.join(self.experiments_dir, experiment_id)
        if os.path.exists(exp_dir):
            import shutil
            shutil.rmtree(exp_dir)
        
        # Remove from metadata
        del self.experiments[experiment_id]
        self._save_metadata()
    
    def get_experiment_dir(self, experiment_id: str) -> Optional[str]:
        """Get the directory path of an experiment."""
        if experiment_id not in self.experiments:
            return None
        
        exp_dir = os.path.join(self.experiments_dir, experiment_id)
        return exp_dir if os.path.exists(exp_dir) else None 
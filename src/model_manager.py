"""
Model Manager for Pixel Life CLI
Handles model registration, storage, and metadata management.
"""

import os
import json
import hashlib
import shutil
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import glob


@dataclass
class ModelMetadata:
    """Model metadata structure."""
    model_id: str
    name: str
    version: str
    description: str
    author: str
    created_at: str
    updated_at: str
    file_size_mb: float
    checksum: str
    tags: List[str]
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]


class ModelManager:
    """Manages model registration and metadata."""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = models_dir
        self.metadata_file = os.path.join(models_dir, "models.json")
        self._ensure_directory()
        self._load_metadata()
    
    def _ensure_directory(self):
        """Ensure models directory exists."""
        os.makedirs(self.models_dir, exist_ok=True)
    
    def _load_metadata(self):
        """Load model metadata from file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.models = {model_id: ModelMetadata(**metadata) 
                                 for model_id, metadata in data.items()}
            except Exception:
                self.models = {}
        else:
            self.models = {}
    
    def _save_metadata(self):
        """Save model metadata to file."""
        data = {model_id: asdict(metadata) 
               for model_id, metadata in self.models.items()}
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _generate_id(self, name: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{name.lower().replace(' ', '_')}_{timestamp}"
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def register_model(self, model_path: str, name: str, description: str = "",
                      author: str = "Unknown", tags: Optional[List[str]] = None,
                      performance_metrics: Optional[Dict[str, float]] = None,
                      hyperparameters: Optional[Dict[str, Any]] = None) -> str:
        """Register a new model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Generate unique ID
        model_id = self._generate_id(name)
        
        # Copy model file
        dest_path = os.path.join(self.models_dir, f"{model_id}.zip")
        shutil.copy2(model_path, dest_path)
        
        # Calculate file size and checksum
        file_size = os.path.getsize(dest_path) / (1024 * 1024)  # MB
        checksum = self._calculate_checksum(dest_path)
        
        # Create metadata
        now = datetime.now().isoformat()
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version="1.0.0",
            description=description,
            author=author,
            created_at=now,
            updated_at=now,
            file_size_mb=file_size,
            checksum=checksum,
            tags=tags or [],
            performance_metrics=performance_metrics or {},
            hyperparameters=hyperparameters or {}
        )
        
        # Save metadata
        self.models[model_id] = metadata
        self._save_metadata()
        
        return model_id
    
    def list_models(self, tags: Optional[List[str]] = None) -> List[ModelMetadata]:
        """List all registered models, optionally filtered by tags."""
        models = list(self.models.values())
        
        if tags:
            models = [model for model in models 
                     if any(tag in model.tags for tag in tags)]
        
        return sorted(models, key=lambda x: x.created_at, reverse=True)
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        return self.models.get(model_id)
    
    def delete_model(self, model_id: str):
        """Delete a registered model."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        # Delete model file
        model_file = os.path.join(self.models_dir, f"{model_id}.zip")
        if os.path.exists(model_file):
            os.remove(model_file)
        
        # Remove from metadata
        del self.models[model_id]
        self._save_metadata()
    
    def update_model(self, model_id: str, **kwargs):
        """Update model metadata."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        for key, value in kwargs.items():
            if hasattr(model, key):
                setattr(model, key, value)
        
        model.updated_at = datetime.now().isoformat()
        self._save_metadata()
    
    def get_model_path(self, model_id: str) -> Optional[str]:
        """Get the file path of a registered model."""
        if model_id not in self.models:
            return None
        
        model_file = os.path.join(self.models_dir, f"{model_id}.zip")
        return model_file if os.path.exists(model_file) else None 
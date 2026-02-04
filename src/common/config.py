"""
Configuration management for HVAC FDD Platform
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for HVAC FDD Platform"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration from dictionary"""
        self._config = config_dict
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key"""
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access"""
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        return self.get(key) is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self._config.copy()


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object
        
    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return Config(config_dict or {})
    
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def load_configs(config_dir: str = "configs") -> Dict[str, Config]:
    """
    Load all configuration files from a directory
    
    Args:
        config_dir: Directory containing YAML config files
        
    Returns:
        Dictionary mapping config names to Config objects
    """
    config_dir = Path(config_dir)
    configs = {}
    
    if not config_dir.exists():
        logger.warning(f"Configuration directory not found: {config_dir}")
        return configs
    
    for config_file in config_dir.glob("*.yaml"):
        config_name = config_file.stem
        try:
            configs[config_name] = load_config(str(config_file))
            logger.info(f"Loaded config: {config_name}")
        except Exception as e:
            logger.error(f"Failed to load config {config_name}: {e}")
    
    return configs


# Global configuration cache
_config_cache: Dict[str, Config] = {}


def get_config(config_name: str, config_dir: str = "configs") -> Config:
    """
    Get configuration with caching
    
    Args:
        config_name: Name of configuration (without .yaml extension)
        config_dir: Directory containing configuration files
        
    Returns:
        Config object
    """
    cache_key = f"{config_dir}/{config_name}"
    
    if cache_key not in _config_cache:
        config_path = Path(config_dir) / f"{config_name}.yaml"
        _config_cache[cache_key] = load_config(str(config_path))
    
    return _config_cache[cache_key]


def clear_config_cache():
    """Clear configuration cache"""
    global _config_cache
    _config_cache.clear()
    logger.info("Configuration cache cleared")


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override config into base config
    
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

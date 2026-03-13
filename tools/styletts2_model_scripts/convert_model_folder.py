import sys
import os
import argparse
from pathlib import Path
import torch
import yaml
import glob

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinfer.models.impl.styletts2.model.modules.load_utils import load_original_styletts2_model, save_model

from tinfer.models.impl.styletts2.model.modules.config import ModelConfig

def find_model_files(model_dir: Path):
    pth_files = list(model_dir.glob("*.pth"))
    yml_files = list(model_dir.glob("*.yml")) + list(model_dir.glob("*.yaml"))
    
    if not pth_files:
        raise ValueError(f"No .pth files found in {model_dir}")
    if not yml_files:
        raise ValueError(f"No .yml or .yaml config files found in {model_dir}")
    
    pth_file = pth_files[0]
    if len(pth_files) > 1:
        print(f"Warning: Multiple .pth files found, using {pth_file.name}")
    
    yml_file = yml_files[0]
    if len(yml_files) > 1:
        print(f"Warning: Multiple config files found, using {yml_file.name}")
    
    return pth_file, yml_file

def resolve_config_paths(config_dict: dict, extra_files_dir: Path):
    def resolve_path(path_str: str) -> str:
        if path_str and path_str.startswith("Utils/"):
            resolved = extra_files_dir / path_str
            if not resolved.exists():
                print(f"Warning: Path {path_str} resolved to {resolved} but file doesn't exist")
            return str(resolved.resolve())
        return path_str
    
    resolved_config = config_dict.copy()
    
    if 'ASR_config' in resolved_config:
        resolved_config['ASR_config'] = resolve_path(resolved_config['ASR_config'])
    if 'ASR_path' in resolved_config:
        resolved_config['ASR_path'] = resolve_path(resolved_config['ASR_path'])
    if 'F0_path' in resolved_config:
        resolved_config['F0_path'] = resolve_path(resolved_config['F0_path'])
    if 'PLBERT_dir' in resolved_config:
        resolved_config['PLBERT_dir'] = resolve_path(resolved_config['PLBERT_dir'])
    
    return resolved_config


def convert_model_folder(model_folder: str, output_dir: str = None, extra_files_dir: str = None):
    model_dir = Path(model_folder)
    if not model_dir.exists():
        raise ValueError(f"Model folder does not exist: {model_folder}")
    
    if output_dir is None:
        project_root = model_dir.parent.parent if model_dir.parent.name == "models" else model_dir.parent
        converted_models_dir = project_root / "converted_models"
        output_dir = converted_models_dir / model_dir.name
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pth_file, yml_file = find_model_files(model_dir)
    
    print(f"Loading model from {pth_file}")
    print(f"Using config from {yml_file}")
    
    with open(yml_file, 'r') as f:
        config_dict = yaml.safe_load(f) or {}
    
    runtime_config = config_dict
    
    if extra_files_dir:
        extra_files_dir = Path(extra_files_dir)
        if not extra_files_dir.exists():
            raise ValueError(f"Extra files directory does not exist: {extra_files_dir}")
        
        print(f"Resolving config paths using extra files from {extra_files_dir}")
        
        resolved_config = resolve_config_paths(config_dict, extra_files_dir)
        
        temp_config_path = output_dir / "temp_config_resolved.yml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(resolved_config, f, default_flow_style=False, sort_keys=False)
        
        try:
            model, model_config = load_original_styletts2_model(
                str(pth_file),
                str(temp_config_path)
            )
        finally:
            if temp_config_path.exists():
                temp_config_path.unlink()
    else:
        model, model_config = load_original_styletts2_model(
            str(pth_file),
            str(yml_file)
        )
    
    model_output_path = output_dir / "model.pth"
    
    print(f"Saving model to {model_output_path}")
    save_model(model, model_config, str(model_output_path), runtime_config=runtime_config)
    
    print("Conversion complete!")
    print(f"  Model saved to: {model_output_path}")
    print(f"  (Model config, weights, and runtime config are all included in model.pth)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert StyleTTS2 model folder to model.pth (includes style encoder)"
    )
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to model folder (e.g., models/magda)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: converted_models/<model_folder_name>)"
    )
    parser.add_argument(
        "-e", "--extra-files",
        type=str,
        default=None,
        help="Path to extra files directory (e.g., models/extra_files). Required if config uses Utils/ paths"
    )
        
    args = parser.parse_args()
    convert_model_folder(args.model_folder, args.output, args.extra_files)

if __name__ == "__main__":
    main()


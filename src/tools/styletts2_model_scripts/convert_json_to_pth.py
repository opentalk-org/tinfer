#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path
import torch


def convert_json_to_pth(json_path: str, output_dir: str = None):
    json_file = Path(json_path)
    if not json_file.exists():
        raise ValueError(f"JSON file does not exist: {json_path}")
    
    if output_dir is None:
        output_dir = json_file.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading embeddings from {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = []
    if isinstance(data, list):
        items = [(i, item, f"embedding_{i:04d}") for i, item in enumerate(data)]
    elif isinstance(data, dict):
        idx = 0
        for key, value in data.items():
            if isinstance(value, list):
                for j, item in enumerate(value):
                    items.append((idx, item, f"{key}_{j:04d}" if len(value) > 1 else key))
                    idx += 1
            else:
                items.append((idx, value, key))
                idx += 1
    else:
        raise ValueError("JSON file must contain a list or dictionary of embeddings")
    
    if len(items) == 0:
        raise ValueError("JSON file is empty")
    
    print(f"Found {len(items)} embeddings")
    print("Processing embeddings...")
    
    saved_files = []
    for i, item, default_name in items:
        if isinstance(item, dict):
            if 'tensor' in item:
                tensor_data = item['tensor']
                file_name = item.get('file', default_name)
                text = item.get('text')
            else:
                tensor_data = item
                file_name = default_name
                text = None
        else:
            tensor_data = item
            file_name = default_name
            text = None
        
        if isinstance(tensor_data, dict):
            if 'tensor' in tensor_data:
                tensor_data = tensor_data['tensor']
            else:
                raise ValueError(f"Item {i} ({default_name}) has nested dict but no 'tensor' field")
        
        if isinstance(tensor_data, list):
            if len(tensor_data) > 0 and isinstance(tensor_data[0], list):
                tensor = torch.tensor(tensor_data[0], dtype=torch.float32)
            else:
                tensor = torch.tensor(tensor_data, dtype=torch.float32)
        elif isinstance(tensor_data, (int, float)):
            raise ValueError(f"Item {i} ({default_name}) appears to be a scalar, expected tensor data")
        else:
            tensor = torch.tensor(tensor_data, dtype=torch.float32)
        
        if tensor.dim() == 2:
            tensor = tensor.squeeze(0)
        
        if isinstance(file_name, str):
            base_name = Path(file_name).stem
        else:
            base_name = str(file_name)
        
        output_path = output_dir / f"{base_name}.pth"
        
        metadata = {
            'source': str(json_file),
            'shape': list(tensor.shape),
            'index': i
        }
        if isinstance(item, dict):
            if 'file' in item:
                metadata['source_file'] = item['file']
            if text:
                metadata['text'] = text
        
        torch.save({
            'voice_vector': tensor,
            'metadata': metadata
        }, str(output_path))
        
        saved_files.append(output_path)
        print(f"  [{i+1}/{len(items)}] Saved {output_path.name} (shape: {tensor.shape})")
    
    print("\nConversion complete!")
    print(f"  Saved {len(saved_files)} voice embedding files to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON voice embeddings to PTH format"
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to JSON file with voice embeddings"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: same directory as JSON file)"
    )
    
    args = parser.parse_args()
    convert_json_to_pth(args.json_path, args.output)


if __name__ == "__main__":
    main()


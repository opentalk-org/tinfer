from pathlib import Path

base_dir = Path(__file__).parent.parent.parent.parent
model_id = "styletts2"
model_name = "magda"
voice_id = "magda_001"
model_path = base_dir / "converted_models" / model_name / "model.pth"
voices_folder = str(base_dir / "converted_models" / model_name / "voices")
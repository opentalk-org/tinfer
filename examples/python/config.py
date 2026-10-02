import os
from pathlib import Path

base_dir = Path(__file__).resolve().parents[2]
model_id = os.environ.get("TINFER_MODEL_ID", "magda")
model_name = model_id
voice_id = os.environ.get("TINFER_VOICE_ID", "magda_001")
server_url = os.environ.get("TINFER_HTTP_URL", "http://localhost:8000")
grpc_address = os.environ.get("TINFER_GRPC_ADDRESS", "localhost:50051")
sample_rate = 24000
output_dir = base_dir / "artifacts" / "outputs" / "examples"

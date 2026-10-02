from pathlib import Path

from grpc_tools import protoc


def main() -> None:
    output = Path(__file__).resolve().parent
    proto = output.parents[2] / "tinfer_rust" / "proto"
    result = protoc.main([
        "grpc_tools.protoc",
        f"-I{proto}",
        f"--python_out={output}",
        f"--grpc_python_out={output}",
        str(proto / "styletts.proto"),
    ])
    assert result == 0, "gRPC client generation failed"
    generated = output / "styletts_pb2_grpc.py"
    generated.write_text(generated.read_text().replace("import styletts_pb2 as", "from . import styletts_pb2 as"))
    print(f"Generated Rust-server client bindings in {output}")


if __name__ == "__main__":
    main()

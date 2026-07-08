import sys
sys.path.insert(0, "/tmp/ort")
import onnx
m = onnx.load("/workspace/converted_models/libri/tensorrt/decoder_dynamic.onnx")
g = m.graph
# map tensor -> producer node
prod = {}
for n in g.node:
    for o in n.output: prod[o] = n
def show(node, depth=0, seen=None):
    seen = seen or set()
    if node.name in seen: return
    seen.add(node.name)
    print("  "*depth + f"{node.op_type} '{node.name}'  in={list(node.input)} out={list(node.output)}")
    for inp in node.input:
        if inp in prod and depth < 4:
            show(prod[inp], depth+1, seen)

print("=== all Tanh nodes ===")
for n in g.node:
    if n.op_type == "Tanh":
        print(f"Tanh '{n.name}' input={list(n.input)} output={list(n.output)}")
        # trace back a few levels
        print(" upstream:")
        show(n, 1)
print("\n=== all Unsqueeze nodes (first 12) ===")
c=0
for n in g.node:
    if n.op_type == "Unsqueeze":
        print(f"Unsqueeze '{n.name}' in={list(n.input)} out={list(n.output)}")
        c+=1
        if c>=12: break
print("\n=== graph outputs ===")
for o in g.output:
    print(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])

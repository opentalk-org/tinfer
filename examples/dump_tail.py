import sys
sys.path.insert(0, "/tmp/ort")
import onnx
m = onnx.load("/workspace/converted_models/libri/tensorrt/decoder_dynamic.onnx")
g = m.graph
prod = {o: n for n in g.node for o in n.output}
cons = {}
for n in g.node:
    for i in n.input: cons.setdefault(i, []).append(n)

# what consumes the noise/f0 unsqueezes?
for t in ["/Unsqueeze_output_0", "/Unsqueeze_1_output_0"]:
    print(f"{t} consumed by: {[ (c.op_type,c.name) for c in cons.get(t,[]) ]}")

# walk backward from Tanh, print the chain of op types (unique) to see the tail region size
seen=set(); order=[]
stack=[prod['audio']]  # Tanh
while stack:
    n=stack.pop()
    if n.name in seen: continue
    seen.add(n.name); order.append((n.op_type,n.name,list(n.input)))
    for i in n.input:
        if i in prod: stack.append(prod[i])
print(f"\ntail region size (nodes reachable from output): {len(order)} / total {len(g.node)}")
from collections import Counter
print("op types in full graph:", dict(Counter(n.op_type for n in g.node)))

# list Conv nodes in order with their output tensor (candidate cut points)
print("\nConv/ConvTranspose nodes (index, name, out):")
idx=0
for n in g.node:
    if n.op_type in ("Conv","ConvTranspose"):
        print(f"  [{idx}] {n.op_type} {n.name} -> {n.output[0]}"); idx+=1

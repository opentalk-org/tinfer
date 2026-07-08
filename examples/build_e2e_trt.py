import sys, time
sys.path.insert(0, "/tmp/ort")
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
SRC = "/tmp/e2e.onnx"
MAXS = 10   # MAX_STEPS; step_noise has MAXS-1

builder = trt.Builder(logger)
network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)
if not parser.parse_from_file(SRC):   # resolves external weight files relative to SRC dir
    for i in range(parser.num_errors): print("PARSE ERR:", parser.get_error(i))
    sys.exit(2)
print("parsed. num_inputs:", network.num_inputs, "num_outputs:", network.num_outputs, "layers:", network.num_layers)
for i in range(network.num_inputs):
    inp = network.get_input(i)
    print("  in", inp.name, inp.shape, inp.dtype)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
prof = builder.create_optimization_profile()
def setp(name, mn, op, mx):
    prof.set_shape(name, mn, op, mx)
setp("tokens",       (1,16), (1,171), (4,512))
setp("input_lengths",(1,),   (1,),    (4,))
setp("ref_s",        (1,256),(1,256), (4,256))
setp("diff_noise",   (1,1,256),(1,1,256),(4,1,256))
setp("step_noise",   (1,MAXS-1,256),(1,MAXS-1,256),(4,MAXS-1,256))
config.add_optimization_profile(prof)

print("building (baked, fp32, dynamic B=1..4 L=16..512) ...", flush=True)
t0 = time.time()
plan = builder.build_serialized_network(network, config)
dt = time.time() - t0
if plan is None:
    print(f"BUILD FAILED after {dt:.1f}s")
    sys.exit(1)
print(f"BUILD OK in {dt:.1f}s, engine {plan.nbytes/1e6:.1f} MB")
open("/tmp/e2e.plan", "wb").write(bytes(plan))
print("saved /tmp/e2e.plan")

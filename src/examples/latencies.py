import asyncio
from time import monotonic
import matplotlib.pyplot as plt

from tinfer.core.async_engine import AsyncStreamingTTS
from tinfer.config.engine_config import StreamingTTSConfig

from utils import base_dir, model_id, voice_id, load_model

async def measure_user_latency(async_tts, user_id):
    text = "To jest przykład tekstu do syntezy mowy, który będzie używany do pomiaru opóźnień w systemie TTS."
    params = {}
    
    start_time = monotonic()
    first_chunk_time = None
    
    async for chunk in async_tts.generate(model_id, voice_id, text, params):
        if first_chunk_time is None:
            first_chunk_time = monotonic()
            latency = first_chunk_time - start_time
            break    
    
    if first_chunk_time is None:
        return None
    
    return latency

async def run_latency_test(num_users_list):
    default_max_batch_size = 10
    config = StreamingTTSConfig(
        default_max_batch_size=default_max_batch_size,
        default_chunk_schedule=[80, 160, 250, 290]
    )
    tts = load_model(config=config, warmup_kwargs={"num_warmup_tasks": default_max_batch_size})
    async_tts = AsyncStreamingTTS(tts)
    
    results_by_num_users = {}
    
    for num_users in num_users_list:
        print(f"\n=== Testing with {num_users} concurrent users ===")
        
        tasks = [measure_user_latency(async_tts, user_id) for user_id in range(num_users)]
        latencies = await asyncio.gather(*tasks)
        
        latencies = [l for l in latencies if l is not None]
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            results_by_num_users[num_users] = {
                'avg_latency': avg_latency,
                'min_latency': min_latency,
                'max_latency': max_latency,
                'all_latencies': latencies
            }
            
            print(f"Average latency: {avg_latency*1000:.2f}ms")
            print(f"Min latency: {min_latency*1000:.2f}ms")
            print(f"Max latency: {max_latency*1000:.2f}ms")
        else:
            print("No latencies measured")
    
    async_tts.stop()
    tts.stop()
    
    return results_by_num_users

def plot_results(results_by_num_users):
    num_users_list = sorted(results_by_num_users.keys())
    avg_latencies = [results_by_num_users[n]['avg_latency'] * 1000 for n in num_users_list]
    min_latencies = [results_by_num_users[n]['min_latency'] * 1000 for n in num_users_list]
    max_latencies = [results_by_num_users[n]['max_latency'] * 1000 for n in num_users_list]
    
    plt.figure(figsize=(10, 6))
    plt.plot(num_users_list, avg_latencies, marker='o', linewidth=2, markersize=8, label='Mean', color='blue')
    plt.plot(num_users_list, min_latencies, marker='s', linewidth=2, markersize=6, label='Min', color='green')
    plt.plot(num_users_list, max_latencies, marker='^', linewidth=2, markersize=6, label='Max', color='red')
    plt.xlabel('Number of Concurrent Users (N)', fontsize=12)
    plt.ylabel('Latency to First Audio (ms)', fontsize=12)
    plt.title('TTS First Chunk Latency vs Number of Concurrent Users', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = base_dir / "output_wavs" / "latency_plot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.show()
    
    print("\n=== Detailed Results ===")
    for num_users in num_users_list:
        result = results_by_num_users[num_users]
        print(f"\n{num_users} users:")
        print(f"  Average latency: {result['avg_latency']*1000:.2f}ms")
        print(f"  Min latency: {result['min_latency']*1000:.2f}ms")
        print(f"  Max latency: {result['max_latency']*1000:.2f}ms")

async def main():
    num_users_list = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
    
    print("Starting first chunk latency test...")
    results = await run_latency_test(num_users_list)
    
    plot_results(results)
    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(main())


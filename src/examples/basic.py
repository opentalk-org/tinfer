import numpy as np
import time
import soundfile as sf
import asyncio

from tinfer.core.async_engine import AsyncStreamingTTS

from utils import base_dir, model_id, voice_id, load_model


def save_audio_chunks(chunks, output_path, sample_rate):
    audio_arrays = [chunk.audio for chunk in chunks]
    concatenated_audio = np.concatenate(audio_arrays)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), concatenated_audio, sample_rate)

def example_generate_full(tts):
    print("\n=== Example 1: generate_full (synchronous) ===")
    output_dir = base_dir / "output_wavs"
    text = "Jest to szczególnie przydatne w programowaniu."
    params = {}
    
    audio_chunk = tts.generate_full(model_id, voice_id, text, params)
    output_path = output_dir / "example_generate_full.wav"
    save_audio_chunks([audio_chunk], output_path, audio_chunk.sample_rate)
    print(f"Saved audio to {output_path}")
    print(f"Sample rate: {audio_chunk.sample_rate}")
    print(f"Audio length: {len(audio_chunk.audio)} samples")

def example_generate_full_batch(tts):
    print("\n=== Example 2: generate_full_batch ===")
    output_dir = base_dir / "output_wavs"
    texts = [
        "To jest pierwszy dłuższy tekst w grupie, który demonstruje możliwości przetwarzania wielu tekstów jednocześnie. Przetwarzanie grupowe pozwala na efektywne wykorzystanie zasobów systemowych i znacznie przyspiesza generowanie dźwięku dla wielu fragmentów. Jest to szczególnie przydatne w aplikacjach, które muszą przetwarzać duże ilości treści w krótkim czasie.",
        "To jest drugi tekst w grupie, który pokazuje, jak system radzi sobie z różnymi długościami i stylami. Każdy tekst jest przetwarzany niezależnie, ale jednocześnie, co pozwala na optymalne wykorzystanie mocy obliczeniowej. Ta funkcjonalność jest kluczowa dla aplikacji produkcyjnych, które wymagają wysokiej wydajności i skalowalności.",
    ]
    params = [{} for _ in texts]
    
    audio_chunks = tts.generate_full_batch(model_id, voice_id, texts, params)

    sample_rate = audio_chunks[0].sample_rate
    output_path = output_dir / "example_batch.wav"
    save_audio_chunks(audio_chunks, output_path, sample_rate)
    print(f"Saved {len(audio_chunks)} audio chunks to {output_path}")

def example_streaming_sync(tts):
    print("\n=== Example 3: Streaming (synchronous) ===")
    output_dir = base_dir / "output_wavs"
    stream = tts.create_stream(model_id, voice_id, {})
    
    stream.add_text("To jest przykład strumieniowego generowania tekstu, które pozwala na otrzymywanie fragmentów dźwięku w miarę ich generowania. ")
    stream.add_text("Ta metoda jest szczególnie przydatna w aplikacjach interaktywnych, gdzie użytkownik może słuchać nagrania jeszcze przed zakończeniem całego procesu. ")
    stream.add_text("Strumieniowanie znacznie poprawia doświadczenie użytkownika, redukując opóźnienia i umożliwiając natychmiastowe rozpoczęcie odtwarzania.")
    stream.force_generate()
    
    audio_chunks = stream.get_audio()
    if audio_chunks:
        sample_rate = audio_chunks[0].sample_rate
        output_path = output_dir / "example_streaming_sync.wav"
        save_audio_chunks(audio_chunks, output_path, sample_rate)
        print(f"Saved {len(audio_chunks)} chunks to {output_path}")
    
    stream.cancel()

async def example_streaming_async(tts):
    print("\n=== Example 4: Streaming (async) ===")
    async_tts = AsyncStreamingTTS(tts)
    
    output_dir = base_dir / "output_wavs"
    text = "To jest przykład asynchronicznego strumieniowego generowania tekstu, które wykorzystuje mechanizmy asynchroniczne do efektywnego przetwarzania. Asynchroniczne podejście pozwala na lepsze wykorzystanie zasobów systemowych i umożliwia obsługę wielu żądań jednocześnie bez blokowania głównego wątku wykonania. Ta funkcjonalność jest niezwykle ważna w aplikacjach serwerowych, które muszą obsługiwać wielu klientów jednocześnie."
    params = {}
    
    audio_chunks = []
    async for chunk in async_tts.generate(model_id, voice_id, text, params):
        audio_chunks.append(chunk)
        print(f"Received chunk {chunk.chunk_index}, length: {len(chunk.audio)} samples")
    
    if audio_chunks:
        sample_rate = audio_chunks[0].sample_rate
        output_path = output_dir / "example_streaming_async.wav"
        save_audio_chunks(audio_chunks, output_path, sample_rate)
        print(f"Saved {len(audio_chunks)} chunks to {output_path}")
    
    async_tts.stop()

def example_force_generate(tts):
    print("\n=== Example 5: Force generate ===")
    output_dir = base_dir / "output_wavs"
    stream = tts.create_stream(model_id, voice_id, {})
    
    stream.add_text("To jest pierwszy dłuższy fragment tekstu, który demonstruje wymuszenie natychmiastowego generowania. Można wymusić rozpoczęcie generowania dźwięku, nawet jeśli bufor tekstu nie jest jeszcze pełny. Jest to szczególnie przydatne w sytuacjach, gdy chcemy uzyskać szybką odpowiedź lub gdy tekst jest dodawany fragmentami.")
    stream.force_generate()
    
    time.sleep(0.5)
    
    stream.add_text(" A to jest drugi, równie obszerny fragment tekstu, który pokazuje, jak system radzi sobie z kolejnymi porcjami danych. Wymuszenie może być wywoływane wielokrotnie, co pozwala na kontrolowanie momentu generowania w zależności od potrzeb aplikacji.")
    stream.force_generate()
    
    audio_chunks = stream.get_audio()
    if audio_chunks:
        sample_rate = audio_chunks[0].sample_rate
        output_path = output_dir / "example_force_generate.wav"
        save_audio_chunks(audio_chunks, output_path, sample_rate)
        print(f"Saved {len(audio_chunks)} chunks to {output_path}")
        print("Force generate triggered immediate generation")
    
    stream.cancel()

def example_cancel(tts):
    print("\n=== Example 6: Cancel ===")
    stream = tts.create_stream(model_id, voice_id, {})
    
    stream.add_text("Ten dłuższy tekst zostanie przerwany przed zakończeniem generowania, co demonstruje możliwość zatrzymania procesu w dowolnym momencie. Możliwość przerwania jest kluczowa w aplikacjach interaktywnych, gdzie użytkownik może zmienić zdanie lub wprowadzić nowe dane przed zakończeniem poprzedniego zadania. System powinien elegancko obsłużyć takie sytuacje i zwolnić zasoby.")
    
    time.sleep(0.1)
    
    print("Cancelling stream...")
    stream.cancel()
    
    audio_chunks = stream.get_audio()
    print(f"Received {len(audio_chunks)} chunks after cancel")

    stream.add_text("A to jest kolejny, obszerny fragment tekstu, który zostanie dodany po przerwaniu poprzedniego zadania. Ten przykład pokazuje, że strumień może być używany ponownie po przerwaniu, co jest ważną funkcjonalnością dla aplikacji, które muszą obsługiwać dynamiczne zmiany w treści. System powinien poprawnie obsłużyć takie scenariusze i kontynuować pracę bez błędów.")
    stream.force_generate()
    
    audio_chunks = stream.get_audio()
    print(f"Received {len(audio_chunks)} chunks after adding text")


def example_alignments(tts):
    print("\n=== Example 7: Alignments ===")
    output_dir = base_dir / "output_wavs"
    text = "To jest dłuższy przykład z synchronizacją, która pokazuje szczegółowe dopasowanie tekstu do dźwięku. Synchronizacja jest niezwykle przydatna w aplikacjach wymagających precyzyjnego dopasowania wyświetlanego tekstu do odtwarzanego dźwięku, takich jak czytniki książek, aplikacje edukacyjne czy narzędzia do nauki języków. Informacje o synchronizacji pozwalają na dokładne podświetlanie aktualnie odtwarzanych słów i fraz, co znacznie poprawia doświadczenie użytkownika i ułatwia śledzenie treści."
    params = {}
    
    audio_chunk = tts.generate_full(model_id, voice_id, text, params)
    
    if audio_chunk.alignments and audio_chunk.alignments.items:
        print(f"Alignment type: {audio_chunk.alignments.type_}")
        print(f"Number of alignment items: {len(audio_chunk.alignments.items)}")
        print("\nFirst 10 alignment items:")
        for i, item in enumerate(audio_chunk.alignments.items[:10]):
            print(f"  {i+1}. '{item.item}' - chars [{item.char_start}:{item.char_end}], time [{item.start_ms}ms:{item.end_ms}ms]")
        
        output_path = output_dir / "example_alignments.wav"
        save_audio_chunks([audio_chunk], output_path, audio_chunk.sample_rate)
        print(f"\nSaved audio with alignments to {output_path}")
    else:
        print("No alignments available for this chunk")

def example_timeout_inference(tts):
    print("\n=== Example 8: Timeout inference ===")
    output_dir = base_dir / "output_wavs"
    stream = tts.create_stream(model_id, voice_id, {})

    stream._request.timeout_trigger_ms = 500.0
    
    print(f"Timeout ustawiony na {stream._request.timeout_trigger_ms}ms")
    print("Dodawanie tekstu fragmentami, generowanie zostanie uruchomione automatycznie po timeout...")
    
    stream.add_text("To jest pierwszy fragment tekstu dodawany do kolejki. ")
    time.sleep(0.2)

    audio_chunks = stream.get_audio()

    print(f"Received {len(audio_chunks)} chunks after adding text")

    stream.add_text("To jest drugi fragment tekstu dodawany do kolejki. ")
    time.sleep(0.35)
    
    audio_chunks = stream.get_audio()
    if audio_chunks:
        sample_rate = audio_chunks[0].sample_rate
        output_path = output_dir / "example_timeout.wav"
        save_audio_chunks(audio_chunks, output_path, sample_rate)
        print(f"Saved {len(audio_chunks)} chunks to {output_path}")
        print("Generowanie zostało uruchomione automatycznie przez timeout")
    else:
        print("Brak chunków audio - może timeout nie został jeszcze osiągnięty")
    
    stream.cancel()

def example_scheduled_inference(tts):
    print("\n=== Example 9: Scheduled inference (bez force_generate) ===")
    output_dir = base_dir / "output_wavs"
    stream = tts.create_stream(model_id, voice_id, {})
    
    print("Dodawanie wystarczająco długiego tekstu, aby przekroczyć próg długości chunka...")
    print("Generowanie zostanie uruchomione automatycznie, gdy tekst przekroczy chunk_length_schedule")
    
    long_text = "To jest bardzo długi fragment tekstu, który został specjalnie przygotowany, aby przekroczyć domyślny próg długości porcji w systemie. System automatycznie wykrywa, gdy tekst w buforze przekracza określony limit znaków i uruchamia generowanie dźwięku bez konieczności wymuszenia. Ta funkcjonalność jest kluczowa dla efektywnego przetwarzania długich tekstów, ponieważ pozwala systemowi na automatyczne zarządzanie procesem generowania w oparciu o długość tekstu. Dzięki temu aplikacje mogą dodawać tekst do kolejki, a system sam zdecyduje, kiedy rozpocząć generowanie, optymalizując wykorzystanie zasobów i zapewniając płynne doświadczenie użytkownika."
    
    stream.add_text(long_text)
    
    audio_chunks = stream.get_audio()
    if audio_chunks:
        sample_rate = audio_chunks[0].sample_rate
        output_path = output_dir / "example_scheduled.wav"
        save_audio_chunks(audio_chunks, output_path, sample_rate)
        print(f"Saved {len(audio_chunks)} chunks to {output_path}")
        print("Generowanie zostało uruchomione automatycznie przez przekroczenie progu długości tekstu")
        print(f"Długość dodanego tekstu: {len(long_text)} znaków")
    else:
        print("Brak chunków audio - może tekst nie przekroczył jeszcze progu")
    
    stream.cancel()

def main():
    tts = load_model()
    
    example_generate_full(tts)
    example_generate_full_batch(tts)
    example_streaming_sync(tts)
    example_force_generate(tts)
    example_cancel(tts)
    example_alignments(tts)
    example_timeout_inference(tts)
    example_scheduled_inference(tts)
    
    print("\n=== Running async examples ===")
    asyncio.run(example_streaming_async(tts))

    print("\nDone! All examples completed.")
    tts.stop()

if __name__ == "__main__":
    main()

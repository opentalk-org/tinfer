import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

from tinfer.core.request import AlignmentType

from utils import base_dir, model_id, voice_id, load_model

text = (
    "Moim zdaniem to nie ma tak, że dobrze albo że nie dobrze. Gdybym miał powiedzieć, co cenię w życiu najbardziej, powiedziałbym, że ludzi. Ekhm… Ludzi, którzy podali mi pomocną dłoń, kiedy sobie nie radziłem, kiedy byłem sam."
)

def plot_spectrogram_with_alignments(audio, sample_rate, alignments, text, title, output_path):
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    duration_seconds = len(audio) / sample_rate
    
    spectrogram = librosa.stft(audio, n_fft=2048, hop_length=512)
    magnitude = np.abs(spectrogram)
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    axes[0].imshow(magnitude_db, aspect='auto', origin='lower', extent=[0, duration_seconds, 0, sample_rate/2], cmap='viridis')
    axes[0].set_ylabel('Frequency (Hz)', fontsize=11)
    axes[0].set_title('Speech Spectrogram', fontsize=12, fontweight='bold')
    axes[0].set_xlim(0, duration_seconds)
    
    if alignments and alignments.items:
        requested_type = alignments.type_
        
        if requested_type == AlignmentType.PHONEME:
            axes[1].set_title('Phoneme Alignments', fontsize=12, fontweight='bold')
            y_pos = 0
            for item in alignments.items:
                start_s = item.start_ms / 1000.0
                end_s = item.end_ms / 1000.0
                axes[1].barh(y_pos, end_s - start_s, left=start_s, height=0.8, alpha=0.6, color='blue')
                axes[1].text(start_s + (end_s - start_s) / 2, y_pos, item.item, 
                            ha='center', va='center', fontsize=8, fontweight='bold')
                y_pos += 1
            axes[1].set_ylabel('Phoneme Index', fontsize=11)
            axes[1].set_xlim(0, duration_seconds)
            axes[1].set_ylim(-0.5, len(alignments.items) - 0.5)
        else:
            axes[1].text(0.5, 0.5, 'Phoneme alignments not available', 
                         ha='center', va='center', transform=axes[1].transAxes, fontsize=11)
            axes[1].set_title('Phoneme Alignments (Not Available)', fontsize=12, fontweight='bold')
        
        if requested_type == AlignmentType.WORD:
            axes[2].set_title('Word Alignments', fontsize=12, fontweight='bold')
            y_pos = 0
            for item in alignments.items:
                start_s = item.start_ms / 1000.0
                end_s = item.end_ms / 1000.0
                axes[2].barh(y_pos, end_s - start_s, left=start_s, height=0.8, alpha=0.6, color='green')
                axes[2].text(start_s + (end_s - start_s) / 2, y_pos, item.item, 
                            ha='center', va='center', fontsize=9, fontweight='bold')
                y_pos += 1
            axes[2].set_ylabel('Word Index', fontsize=11)
            axes[2].set_xlim(0, duration_seconds)
            axes[2].set_ylim(-0.5, len(alignments.items) - 0.5)
        else:
            axes[2].text(0.5, 0.5, 'Word alignments not available', 
                         ha='center', va='center', transform=axes[2].transAxes, fontsize=11)
            axes[2].set_title('Word Alignments (Not Available)', fontsize=12, fontweight='bold')
        
        if requested_type == AlignmentType.CHAR:
            axes[3].set_title('Character Alignments', fontsize=12, fontweight='bold')
            y_pos = 0
            for item in alignments.items:
                start_s = item.start_ms / 1000.0
                end_s = item.end_ms / 1000.0
                color = 'red' if item.item == ' ' else 'orange'
                axes[3].barh(y_pos, end_s - start_s, left=start_s, height=0.8, alpha=0.6, color=color)
                if item.item != ' ':
                    axes[3].text(start_s + (end_s - start_s) / 2, y_pos, item.item, 
                                ha='center', va='center', fontsize=7, fontweight='bold')
                y_pos += 1
            axes[3].set_ylabel('Character Index', fontsize=11)
            axes[3].set_xlabel('Time (seconds)', fontsize=11)
            axes[3].set_xlim(0, duration_seconds)
            axes[3].set_ylim(-0.5, len(alignments.items) - 0.5)
        else:
            axes[3].text(0.5, 0.5, 'Character alignments not available', 
                         ha='center', va='center', transform=axes[3].transAxes, fontsize=11)
            axes[3].set_title('Character Alignments (Not Available)', fontsize=12, fontweight='bold')
            axes[3].set_xlabel('Time (seconds)', fontsize=11)
    else:
        for ax in axes[1:]:
            ax.text(0.5, 0.5, 'No alignments available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=11)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()

def example_1_phoneme_alignments(tts):
    print("\n=== Example 1: Phoneme Alignments ===")
    output_dir = base_dir / "output_wavs"
    params = {"alignment_type": AlignmentType.PHONEME}
    
    audio_chunk = tts.generate_full(model_id, voice_id, text, params)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / "alignment_audio_example1_phoneme.wav"
    sf.write(str(audio_path), audio_chunk.audio, audio_chunk.sample_rate)
    print(f"Audio saved to {audio_path}")
    
    if audio_chunk.alignments and audio_chunk.alignments.items:
        output_path = output_dir / "alignment_plot_example1_phoneme.png"
        plot_spectrogram_with_alignments(
            audio_chunk.audio,
            audio_chunk.sample_rate,
            audio_chunk.alignments,
            text,
            "Example 1: Phoneme Alignments",
            output_path
        )
    else:
        print("No alignments available for this chunk")

def example_2_word_alignments(tts):
    print("\n=== Example 2: Word Alignments ===")
    output_dir = base_dir / "output_wavs"
    params = {"alignment_type": AlignmentType.WORD}
    
    audio_chunk = tts.generate_full(model_id, voice_id, text, params)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / "alignment_audio_example2_word.wav"
    sf.write(str(audio_path), audio_chunk.audio, audio_chunk.sample_rate)
    print(f"Audio saved to {audio_path}")
    
    if audio_chunk.alignments and audio_chunk.alignments.items:
        output_path = output_dir / "alignment_plot_example2_word.png"
        plot_spectrogram_with_alignments(
            audio_chunk.audio,
            audio_chunk.sample_rate,
            audio_chunk.alignments,
            text,
            "Example 2: Word Alignments",
            output_path
        )
    else:
        print("No alignments available for this chunk")

def example_3_char_alignments(tts):
    print("\n=== Example 3: Character Alignments ===")
    output_dir = base_dir / "output_wavs"
    params = {"alignment_type": AlignmentType.CHAR}
    
    audio_chunk = tts.generate_full(model_id, voice_id, text, params)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / "alignment_audio_example3_char.wav"
    sf.write(str(audio_path), audio_chunk.audio, audio_chunk.sample_rate)
    print(f"Audio saved to {audio_path}")
    
    if audio_chunk.alignments and audio_chunk.alignments.items:
        output_path = output_dir / "alignment_plot_example3_char.png"
        plot_spectrogram_with_alignments(
            audio_chunk.audio,
            audio_chunk.sample_rate,
            audio_chunk.alignments,
            text,
            "Example 3: Character Alignments",
            output_path
        )
    else:
        print("No alignments available for this chunk")

def main():
    tts = load_model()
    
    example_1_phoneme_alignments(tts)
    example_2_word_alignments(tts)
    example_3_char_alignments(tts)
    
    print("\nDone! All alignment plots and audios generated.")
    tts.stop()

if __name__ == "__main__":
    main()

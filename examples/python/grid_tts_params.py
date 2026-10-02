from itertools import product
from urllib.request import urlopen

from config import model_id, output_dir, voice_id
from utils import VoiceSettings, save_audio, speech_request


def main() -> None:
    for speed, alpha, beta in product((0.9, 1.1), (0.3, 0.7), (0.3, 0.7)):
        settings = VoiceSettings(speed=speed, alpha=alpha, beta=beta)
        request = speech_request("To jest porównanie parametrów syntezy.", model_id, voice_id, "", settings)
        with urlopen(request, timeout=180) as response:
            audio = response.read()
        path = output_dir / "parameters" / f"speed_{speed}_alpha_{alpha}_beta_{beta}.wav"
        save_audio(audio, path)


if __name__ == "__main__":
    main()

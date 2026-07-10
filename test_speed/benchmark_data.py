from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random

import numpy as np


@dataclass(frozen=True)
class BenchmarkConfig:
    archive_path: Path
    model_path: Path
    results_dir: Path
    seed: int
    voice_count: int
    highlighted_voice_count: int


@dataclass(frozen=True)
class SynthesisProfile:
    name: str
    results_dir: Path
    use_diffusion: bool


@dataclass(frozen=True)
class TextInput:
    text_id: str
    text: str


@dataclass(frozen=True)
class RequestMetric:
    voice_id: str
    text_id: str
    text: str
    text_length: int
    phoneme_count: int
    predicted_seconds: float
    phonemes_per_second: float
    audio_path: str


@dataclass(frozen=True)
class PhonemeMetric:
    phoneme: str
    duration_seconds: float
    voice_id: str
    text_id: str


@dataclass(frozen=True)
class SummaryRow:
    phoneme: str
    count: int
    average_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    p10_seconds: float
    p90_seconds: float

    @classmethod
    def from_values(cls, phoneme: str, values: list[float]) -> "SummaryRow":
        durations = np.asarray(values, dtype=np.float64)
        return cls(
            phoneme=phoneme,
            count=len(values),
            average_seconds=float(np.mean(durations)),
            minimum_seconds=float(np.min(durations)),
            maximum_seconds=float(np.max(durations)),
            p10_seconds=float(np.percentile(durations, 10)),
            p90_seconds=float(np.percentile(durations, 90)),
        )


POLISH_INPUTS = [
    TextInput("length_002", "No"),
    TextInput("length_011", "Pada deszcz"),
    TextInput("length_022", "Dzisiaj świeci słońce."),
    TextInput("length_034", "Mały kot spokojnie zasnął na fotelu."),
    TextInput("length_046", "Po południu spotkamy się przy starej fontannie."),
    TextInput("length_060", "W sobotę odwiedzimy targ, kupimy świeże jabłka i pachnący chleb."),
    TextInput("length_076", "Pociąg do Krakowa odjechał punktualnie, choć na peronie czekało wielu ludzi."),
    TextInput("length_091", "Marta otworzyła okno, wpuściła do pokoju chłodne powietrze i wróciła do czytania książki."),
    TextInput("length_108", "Nad jeziorem panowała cisza, tylko trzciny szumiały lekko, a pojedyncza łódź przesuwała się ku drugiemu brzegowi."),
    TextInput("length_127", "Kiedy skończył się letni deszcz, dzieci wybiegły na podwórko, przeskakiwały przez kałuże i szukały kolorowej tęczy nad dachami."),
    TextInput("length_146", "W niewielkiej piekarni od rana pachniało cynamonem, masłem i świeżym ciastem, dlatego przechodnie chętnie zaglądali po bułki na śniadanie."),
    TextInput("length_165", "Podczas niedzielnego spaceru znaleźliśmy w lesie polanę pełną fioletowych kwiatów, usiedliśmy pod sosną i słuchaliśmy dzięcioła pracującego wysoko w koronie."),
    TextInput("length_187", "Wieczorem sąsiedzi przynieśli krzesła do ogrodu, zapalili małe lampiony i długo rozmawiali o podróżach, planach na jesień oraz zmianach, które zaszły ostatnio w miasteczku."),
    TextInput("length_218", "Bibliotekarka ułożyła nowe książki na wystawie, przygotowała dla uczniów zagadki i zaznaczyła na mapie miejsca opisane w powieściach, aby kolejne spotkanie klubu czytelniczego było ciekawe, żywe i pełne odkryć."),
    TextInput("length_255", "O świcie rybacy wypłynęli z małego portu, sprawdzili sieci i skierowali łodzie ku spokojnej zatoce, podczas gdy na nabrzeżu budziły się kawiarnie, dostawcy rozwozili pieczywo, a pierwsi turyści fotografowali mewy krążące nad latarnią."),
    TextInput("length_300", "Rankiem mieszkańcy kamienicy spotkali się na dziedzińcu, aby posadzić zioła, naprawić drewnianą ławkę i zawiesić kolorowe lampki. Każdy przyniósł potrzebne narzędzia, a dzieci napełniały konewki wodą. Po kilku godzinach wspólnej pracy ogród wyglądał pięknie i zachęcał do odpoczynku, także po zmroku."),
]


def select_names(names: list[str], count: int, seed: int) -> list[str]:
    if len(names) < count:
        raise ValueError(f"Need {count} inputs, found {len(names)}")
    return sorted(Random(seed).sample(sorted(names), count))


def summarize_phonemes(metrics: list[PhonemeMetric]) -> list[SummaryRow]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for metric in metrics:
        grouped[metric.phoneme].append(metric.duration_seconds)
    return [
        SummaryRow.from_values(phoneme, grouped[phoneme])
        for phoneme in sorted(grouped)
    ]

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tinfer.models.impl.styletts2.model.inference_config import StyleTTS2Params

from test_speed.benchmark_data import TextInput


POLISH_PASSAGE = (
    "Rankiem mieszkańcy kamienicy spotkali się na dziedzińcu, aby posadzić "
    "zioła, naprawić drewnianą ławkę i zawiesić kolorowe lampki. Każdy "
    "przyniósł potrzebne narzędzia, a dzieci napełniały konewki wodą. Po "
    "kilku godzinach wspólnej pracy ogród wyglądał pięknie i zachęcał do "
    "odpoczynku. W pobliskiej piekarni od rana pachniało cynamonem, masłem "
    "i świeżym chlebem, dlatego przechodnie chętnie zaglądali po śniadanie. "
    "Później bibliotekarka ułożyła książki na wystawie, przygotowała dla "
    "uczniów zagadki i zaznaczyła na mapie miejsca opisane w powieściach. "
    "Nad jeziorem panowała cisza, tylko trzciny szumiały lekko, a mała łódź "
    "przesuwała się powoli ku drugiemu brzegowi. Kiedy skończył się letni "
    "deszcz, dzieci wybiegły na podwórko, przeskakiwały przez kałuże i "
    "szukały kolorowej tęczy nad dachami. Podczas niedzielnego spaceru "
    "znaleźliśmy w lesie polanę pełną fioletowych kwiatów, usiedliśmy pod "
    "sosną i słuchaliśmy dzięcioła pracującego wysoko w koronie. Wieczorem "
    "sąsiedzi przynieśli krzesła do ogrodu, zapalili lampiony i długo "
    "rozmawiali o podróżach, planach na jesień oraz zmianach w miasteczku. "
    "O świcie rybacy wypłynęli z portu, sprawdzili sieci i skierowali łodzie "
    "ku spokojnej zatoce, podczas gdy na nabrzeżu budziły się kawiarnie, "
    "dostawcy rozwozili pieczywo, a pierwsi turyści fotografowali mewy. "
    "Po południu na rynku rozpoczął się koncert, sprzedawcy ustawili stoiska, "
    "muzycy stroili instrumenty, a rodziny zajmowały miejsca przy scenie."
)
ENGLISH_PASSAGE = (
    "Early in the morning, the residents gathered in the courtyard to plant "
    "herbs, repair the wooden bench, and hang small colored lights. Everyone "
    "brought useful tools while the children filled watering cans beside the "
    "gate. After several hours of shared work, the garden looked welcoming "
    "and offered a quiet place to rest. Across the street, warm bread and "
    "cinnamon rolls filled the bakery with a rich smell, so passing neighbors "
    "stopped for breakfast. Later, the librarian arranged books in the front "
    "window, prepared puzzles for students, and marked every location from the "
    "week's stories on a large map. Near the lake, reeds moved gently in the "
    "wind as a narrow boat crossed toward the opposite shore. When the summer "
    "rain ended, children ran outside, jumped over puddles, and searched for "
    "a bright rainbow above the roofs. During an afternoon walk, a family "
    "found a meadow full of purple flowers, sat beneath an old pine tree, and "
    "listened to a woodpecker high in the branches. At sunset, neighbors "
    "carried chairs into the garden, lit lanterns, and talked about journeys, "
    "autumn plans, and recent changes around town. Before dawn, fishing boats "
    "left the harbor and moved toward the calm bay while cafes opened along "
    "the waterfront, delivery vans brought fresh food, and the first visitors "
    "photographed seabirds circling the lighthouse. In the afternoon, a "
    "concert began in the central square. Vendors arranged their stalls, "
    "musicians tuned their instruments, and families found places near the "
    "stage. The program continued through the evening with songs, stories, "
    "and short performances from local schools. Volunteers served tea, helped "
    "older guests find seats, and collected empty cups after each interval. "
    "When the final music ended, everyone worked together to fold the tables, "
    "carry equipment indoors, and leave the square clean for the next morning."
)


class TokenCountingModel(Protocol):
    def _text_token_count(
        self,
        text: str,
        params: StyleTTS2Params,
    ) -> int: ...


@dataclass(frozen=True)
class PrefixCandidate:
    text: str
    token_count: int


def build_phoneme_grid(
    model: TokenCountingModel,
    passage: str,
    point_count: int,
    max_tokens: int,
) -> list[TextInput]:
    words = passage.split()
    candidates = []
    overflow_found = False
    for word_count in range(1, len(words) + 1):
        text = " ".join(words[:word_count])
        token_count = model._text_token_count(text, StyleTTS2Params()) - 1
        if token_count > max_tokens:
            overflow_found = True
            break
        if not candidates or token_count > candidates[-1].token_count:
            candidates.append(PrefixCandidate(text, token_count))

    assert overflow_found, "Polish passage must exceed the model token window"
    assert len(candidates) >= point_count, "Not enough unique prefix candidates"
    targets = np.linspace(
        candidates[0].token_count,
        candidates[-1].token_count,
        point_count,
    )
    selected = []
    next_index = 0
    for position, target in enumerate(targets):
        remaining_points = point_count - position
        last_allowed_index = len(candidates) - remaining_points
        selected_index = next_index
        while (
            selected_index < last_allowed_index
            and candidates[selected_index + 1].token_count <= target
        ):
            selected_index += 1
        selected.append(candidates[selected_index])
        next_index = selected_index + 1

    assert selected[-1] == candidates[-1]
    return [
        TextInput(
            text_id=f"phonemes_{item.token_count:03d}",
            text=item.text,
            input_phoneme_tokens=item.token_count,
        )
        for item in selected
    ]

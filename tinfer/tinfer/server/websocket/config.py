class WebSocketConfig:
    model_id: str
    voice_id: str
    language_code: str | None = None
    output_format: str = "mp3_44100_32"
    sync_alignment: bool = False
    auto_mode: bool = False
    apply_text_normalization: str = "auto"
    seed: int | None = None
    inactivity_timeout: int = 20
    enable_logging: bool = True
    enable_ssml_parsing: bool = False
    # pronunciation_dictionary_locators: list[PronunciationDictionaryLocator] = field(
    #     default_factory=list
    # )
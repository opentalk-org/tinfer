function defaultOption(value, label) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  return option;
}

function updateSelection(catalog, modelSelect, voiceSelect, languageSelect, makeOption = defaultOption) {
  const model = catalog.find((item) => item.id === modelSelect.value);
  if (!model) {
    voiceSelect.replaceChildren();
    languageSelect.replaceChildren();
    return;
  }
  const voices = model.voices.map((id) => makeOption(id, id));
  const languages = model.languages.map((language) => makeOption(language.id, language.name));
  voiceSelect.replaceChildren(...voices);
  languageSelect.replaceChildren(...languages);
  languageSelect.value = model.defaultLanguage;
}

function populate(catalog, modelSelect, voiceSelect, languageSelect, makeOption = defaultOption) {
  const models = catalog.map((model) => makeOption(model.id, model.id));
  modelSelect.replaceChildren(...models);
  updateSelection(catalog, modelSelect, voiceSelect, languageSelect, makeOption);
}

function controlState(protocol, mode) {
  const socketMode = mode === "ws_single" || mode === "ws_multi";
  const timing = protocol === "grpc" || socketMode || mode === "post_timing" || mode === "stream_timing";
  return {
    timing,
    granularityDisabled: protocol === "grpc" || !timing,
    voiceSettingsDisabled: protocol === "grpc",
    scheduleDisabled: !socketMode,
  };
}

const catalogUi = { controlState, populate, updateSelection };
if (typeof module !== "undefined") module.exports = catalogUi;
if (typeof window !== "undefined") window.TinferCatalogUi = catalogUi;

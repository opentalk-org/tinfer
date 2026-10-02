function voiceMap(voices) {
  const result = new Map();
  for (const voice of voices) {
    const ids = result.get(voice.model_id) || [];
    ids.push(voice.voice_id);
    result.set(voice.model_id, ids);
  }
  return result;
}

function sortedVoices(mapping, modelId) {
  return (mapping.get(modelId) || []).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

function fromGrpc(modelResponse, voiceResponse) {
  const mapping = voiceMap(voiceResponse.voices);
  return modelResponse.models.map((model) => ({
    id: model.model_id,
    voices: sortedVoices(mapping, model.model_id),
    languages: model.supported_languages.map((id) => ({ id, name: id })),
    defaultLanguage: model.default_language,
  }));
}

function fromHttp(models, voiceResponse) {
  const mapping = voiceMap(voiceResponse.voices);
  return models.map((model) => ({
    id: model.model_id,
    voices: sortedVoices(mapping, model.model_id),
    languages: model.languages.map((language) => ({ id: language.language_id, name: language.name })),
    defaultLanguage: model.default_language,
  }));
}

module.exports = { fromGrpc, fromHttp };

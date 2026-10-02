const assert = require("node:assert/strict");
const test = require("node:test");

const { fromGrpc, fromHttp } = require("../catalog");

const voices = [{ model_id: "libri", voice_id: "7" }, { model_id: "libri", voice_id: "2" }];

test("normalizes structured gRPC models with voices and baked language default", () => {
  const catalog = fromGrpc({ models: [{ model_id: "libri", supported_languages: ["en-gb", "en-us"], default_language: "en-us" }] }, { voices });

  assert.deepEqual(catalog, [{
    id: "libri",
    voices: ["2", "7"],
    languages: [{ id: "en-gb", name: "en-gb" }, { id: "en-us", name: "en-us" }],
    defaultLanguage: "en-us",
  }]);
});

test("normalizes top-level HTTP model array and associates voices", () => {
  const catalog = fromHttp([{
    model_id: "libri",
    languages: [{ language_id: "en-us", name: "English" }],
    default_language: "en-us",
  }], { voices });

  assert.deepEqual(catalog[0], {
    id: "libri",
    voices: ["2", "7"],
    languages: [{ id: "en-us", name: "English" }],
    defaultLanguage: "en-us",
  });
});

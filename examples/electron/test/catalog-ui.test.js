const assert = require("node:assert/strict");
const test = require("node:test");

const { controlState, updateSelection } = require("../catalog-ui");

class Select {
  constructor(value = "") { this.value = value; this.children = []; }
  replaceChildren(...children) { this.children = children; this.value = children[0]?.value || ""; }
}

test("model changes replace language choices and select baked default", () => {
  const model = new Select("polish");
  const voice = new Select();
  const language = new Select();
  const catalog = [{
    id: "polish", voices: ["1"],
    languages: [{ id: "pl", name: "Polish" }, { id: "en-us", name: "English" }],
    defaultLanguage: "pl",
  }];
  const makeOption = (value, label) => ({ value, textContent: label });

  updateSelection(catalog, model, voice, language, makeOption);

  assert.deepEqual(voice.children.map((item) => item.value), ["1"]);
  assert.deepEqual(language.children.map((item) => item.value), ["pl", "en-us"]);
  assert.equal(language.value, "pl");
});

test("single socket mode enables timing and its chunk schedule after protocol switch", () => {
  assert.deepEqual(controlState("api", "ws_single"), {
    timing: true,
    granularityDisabled: false,
    voiceSettingsDisabled: false,
    scheduleDisabled: false,
  });
});

test("POST modes disable socket schedule and plain audio disables highlighting", () => {
  assert.deepEqual(controlState("api", "post_audio"), {
    timing: false,
    granularityDisabled: true,
    voiceSettingsDisabled: false,
    scheduleDisabled: true,
  });
  assert.equal(controlState("api", "stream_timing").granularityDisabled, false);
  assert.equal(controlState("api", "stream_timing").scheduleDisabled, true);
});

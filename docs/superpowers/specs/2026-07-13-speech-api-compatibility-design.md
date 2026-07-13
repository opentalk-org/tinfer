# Speech API Compatibility Design

## Objective

Make Tinfer a pragmatic drop-in replacement for the six ElevenLabs text-to-speech surfaces and the discovery endpoints used by clients. Match request and response shapes, defaults, validation, formats, streaming behavior, context state, and generation semantics. A client written for the reference API must work without request changes.

Hosted account behavior is outside the target. Every documented syntactically valid field is accepted. Fields that Tinfer or the loaded model cannot represent are silently ignored and identified as no-effect in the compatibility matrix. Supported values are mapped to the closest meaningful model controls rather than rejected because the underlying parameter names differ.

Target source code, tests, and comments remain provider-neutral. Tinfer extensions such as `alpha`, `beta`, baked language metadata, and `default_language` remain available and documented.

## Approaches Considered

### Schema-first compatibility adapter

Define provider-neutral request models, validation, response formatting, context transitions, and capability policies at the API boundary. Translate validated requests into Tinfer engine parameters.

This is the selected approach because it gives all six routes one contract, records unsupported behavior without breaking clients, and limits future schema drift.

### Patch the current handlers

Add missing cases directly to each HTTP and WebSocket handler. This is initially faster, but duplicates defaults and validation across single-context, multi-context, unary, and streaming paths. The implementations would continue to diverge.

### Full hosted-service clone

Reproduce authentication, accounts, billing, retention, history storage, and all service-side behavior. This is unnecessary for local inference and would substantially expand the project.

## Compatibility Policy

Each documented field has one explicit capability classification:

- **Implemented:** affects the response as documented.
- **Mapped:** affects the closest semantically equivalent Tinfer model control.
- **Accepted/no-effect:** accepted and silently ignored because the loaded model or local service cannot represent it.
- **Tinfer extension:** accepted and documented without changing the reference field meanings.

Unsupported fields never cause an otherwise valid request to fail. Invalid types, invalid enum members, missing required fields, unknown resources, and invalid message transitions use only the reference API's status codes, response bodies, and WebSocket close behavior. The compatibility matrix remains exhaustive and is updated whenever a field changes classification.

## Ranked Compatibility Gaps

| Rank | Gap | User-visible risk | Proposed change |
| ---: | --- | --- | --- |
| 1 | HTTP inference and encoding failures are inconsistent | A request can return `500`, partial audio, or an abruptly truncated stream; unary merging can discard chunk errors | Propagate every failed chunk, return structured errors before headers, and terminate prepared streams predictably |
| 2 | An omitted `model_id` can resolve to a model that is not loaded | Ordinary SDK requests using the documented default fail with `404` | Configure a compatibility-default model or alias mapped to a loaded model |
| 3 | Valid optional request shapes are rejected | `voice_settings: null`, dictionaries, and history fields can fail before synthesis | Accept every documented shape; map supported values and silently ignore documented no-effect fields |
| 4 | Multi-context reinitialization differs | Reinitializing a live context with new settings fails, disrupting interruption workflows | Drain and replace the named context atomically |
| 5 | Most voice settings have no effect | Stability, similarity, style, and speaker boost appear to work while audio is unchanged | Map representable controls into each model and silently ignore only controls without a defensible equivalent |
| 6 | `auto_mode` is ignored | Clients expecting automatic low-latency generation can wait for a schedule threshold or flush | Generate automatically from buffered sentence and readiness state |
| 7 | `try_trigger_generation` behaves like `flush` | Undersized buffers generate unconditionally, changing quality and latency | Trigger only after the minimum safe buffer; retain unconditional `flush` |
| 8 | Normalization and SSML controls are ignored | Numbers, dates, abbreviations, and markup can be pronounced differently than requested | Connect normalization modes to preprocessing; parse supported SSML and silently ignore unsupported constructs |
| 9 | Pronunciation dictionaries are ignored | Names and domain terminology are spoken incorrectly despite supplied locators | Resolve locators into per-request pronunciation substitutions before phonemization |
| 10 | Unary WAV formats are missing | Valid unary WAV requests fail | Add `wav_8000`, `wav_16000`, `wav_22050`, `wav_24000`, `wav_32000`, `wav_44100`, and `wav_48000` |
| 11 | History and stitching inputs are unsupported | Adjacent clips can have discontinuous prosody | First support `previous_text` and `next_text`; defer request-ID stitching until history storage exists |
| 12 | `seed` is ignored | Callers requesting repeatability receive different audio | Pass a request-scoped seed through scheduling and model inference |
| 13 | Validation and error shapes differ | Generated SDKs and strict clients may fail to parse errors or observe different status codes | Enforce documented validation and reproduce the reference HTTP bodies, status codes, and WebSocket close behavior without Tinfer-specific alternatives |
| 14 | Normalized alignment duplicates original alignment | Highlighting normalized speech can point to the wrong characters | Preserve separate original and normalized span maps through preprocessing |
| 15 | Voice discovery uses a custom response | Standard clients cannot consume `/v1/voices` | Return the documented voice-list envelope and fields with local capability metadata |
| 16 | Compressed streaming lacks persistent-encoder guarantees | Concatenated MP3 or Opus chunks can behave differently across decoders | Use one encoder session per response and test decoding the complete concatenated stream |
| 17 | Model metadata is sparse | Capability-driven clients may hide models or assume options are unavailable | Populate meaningful optional capability fields and retain `default_language` as an extension |
| 18 | Minor wire differences remain | Extra fields and permissive numeric bounds can break strict validators | Remove undocumented response fields and enforce exact types and ranges |

## Delivery Order

### P0: reliability and valid-client compatibility

Implement ranks 1–4, 13, and 16 first. These changes prevent corrupted or truncated responses, allow ordinary valid requests to enter synthesis, and align context state and failure behavior.

### P1: core generation behavior

Implement ranks 5–8, 10, 12, 14, and 18. These changes align audio controls, latency triggers, preprocessing, formats, determinism, alignment, and strict wire behavior.

### P2: advanced behavior and discovery

Implement ranks 9, 11, 15, and 17. Dictionary resolution, continuity context, and complete discovery metadata are valuable but do not block the most common known-voice synthesis flow.

## Model Value Mapping

Mappings are capability-driven so a different model can provide different equivalents without changing the public API. For the current StyleTTS2 implementation:

| Public field | StyleTTS2 mapping | No-effect condition |
| --- | --- | --- |
| `speed` | `speed` | Never when the model exposes speaking rate |
| `stability` | `style_interpolation_factor`; higher stability preserves more style across chunks | Model has no cross-chunk style state |
| `similarity_boost` | Reduce `alpha` and `beta` as similarity rises so synthesis stays closer to the reference voice | Model has no reference/predicted-style blend |
| `style` | Increase `embedding_scale` where the runtime supports guidance scaling | Runtime fixes embedding scale, including unsupported TensorRT configurations |
| `use_speaker_boost` | No direct StyleTTS2 equivalent | Always for current models |
| `seed` | Request-scoped diffusion generator and deterministic scheduling inputs | A runtime operation has no seeded implementation |

Mappings clamp into each model's valid ranges after the public value passes reference validation. Explicit Tinfer `alpha` and `beta` extensions take precedence over derived similarity values because they are direct model controls.

## Architecture

The API boundary is split into focused units, each below the repository file-size limit:

- Contract models represent HTTP requests, WebSocket messages, formats, errors, and capability classifications.
- Parsers validate exact wire types, nullability, defaults, bounds, and message variants.
- A capability mapper converts supported controls into `StreamParams` and drops unavailable controls without failing the request.
- HTTP response writers own unary encoding, persistent streaming encoders, timing frames, and post-header failure termination.
- WebSocket context state machines own initialization, text buffering, triggering, reinitialization, keepalive, finalization, and policy errors.
- Discovery formatters expose model and voice capabilities from baked metadata without inventing unavailable features.

Package initializers export only public server entry points. Contract types are imported from their defining modules rather than re-exported through alias modules.

## Request Flow

1. Parse the route, query, headers, and body or WebSocket frame into typed contract models.
2. Apply reference defaults and validate exact types, bounds, and message state.
3. Resolve the model alias, voice, language, and baked capabilities.
4. Map supported controls and discard documented no-effect controls.
5. Map supported fields into Tinfer engine parameters and preprocessing context.
6. Generate raw audio and distinct original and normalized alignments.
7. Encode through a response-scoped encoder and format the exact response schema.
8. Drain or abort resources before releasing health admission.

## Error Handling

HTTP validation errors use the documented `detail` array with field locations, messages, and error types. Unknown model, voice, overload, and inference failures use the same status and body shape as the reference API. Errors after streaming headers end the transport in the same way as the reference service and never append a Tinfer-specific payload to audio or timing data.

WebSocket handshake, message validation, invalid state transitions, context failures, and socket failures reproduce the reference frame and close behavior. No additional Tinfer error schema or close convention is exposed. Cleanup and health admission still complete after the reference-compatible transport outcome.

## Testing

- Contract-table tests cover every documented argument, default, null value, bound, enum member, mapped value, no-effect field, and Tinfer extension.
- Golden route tests cover the four HTTP and two WebSocket surfaces with identical request fixtures.
- State-machine tests cover first messages, triggers, flushes, keepalives, reinitialization, context closure, socket closure, inactivity, disconnects, and failures.
- Encoder tests decode complete unary and concatenated streaming output for every supported codec family and sample rate.
- Alignment tests verify separate original and normalized character sequences and timing units.
- Failure tests inject inference and encoding errors before and after response preparation and verify both reference-compatible transport behavior and health cleanup.
- Discovery tests verify standard model and voice envelopes plus baked language defaults.
- Electron tests exercise all six modes against the same contract fixtures used by the server tests.

## Acceptance Criteria

- Every documented syntactically valid request is accepted, including unsupported options.
- Unsupported options are silently ignored and listed as accepted/no-effect in the compatibility matrix.
- Supported public controls are mapped to the closest meaningful loaded-model values, with deterministic precedence for Tinfer extensions.
- All six synthesis routes share defaults, validation, capability mapping, error shapes, and format behavior where their reference contracts agree.
- Multi-context reinitialization, triggering, finalization, and failure isolation are deterministic.
- Unary and streaming audio decode completely in every advertised supported format.
- Original and normalized alignments are independently correct.
- The model and voice endpoints are consumable by standard clients while retaining documented Tinfer metadata extensions.
- The compatibility matrix lists every request and response field and accurately reflects its implemented classification.
- Target source, tests, and comments contain no provider name.
- No changed file exceeds 300 lines and no changed folder exceeds 16 files.

## Explicit Non-Goals

- Hosted authentication or authorization enforcement; related fields remain accepted/no-effect
- Billing, subscription tiers, quotas, or character charging
- Data-retention guarantees or zero-retention storage policy
- Hosted request analytics and logging semantics
- Request-ID stitching until local history storage is deliberately added; related fields remain accepted/no-effect

use std::fs;

use super::manifest::Manifest;
use super::preprocessing::{StyleTts2Params, diffusion_schedule};
use super::text::{normalize_text, tokenize};

#[test]
fn params_match_the_python_model_defaults() {
    let params: StyleTts2Params = serde_json::from_value(serde_json::json!({})).unwrap();
    assert!(params.use_diffusion);
    assert!(!params.phonemized);
    assert_eq!(params.diffusion_steps, 5);
    assert_eq!((params.alpha, params.beta, params.speed), (0.3, 0.7, 1.0));
}

#[test]
fn normalization_is_language_agnostic_and_tokens_follow_the_exported_vocabulary() {
    let normalized = normalize_text("  Ala—ma (kota)  ");
    assert_eq!(normalized, "Ala—ma kota.");
    let symbols = ["$", "A", "l", "a", "—", "m", " ", "k", "o", "t", "."];
    assert_eq!(tokenize(&normalized, &symbols), vec![0, 1, 2, 3, 4, 5, 3, 6, 7, 8, 9, 3, 10]);
    assert_eq!(tokenize("a", &["$", "a", "a"]), vec![0, 2]);
}

#[test]
fn manifest_indexes_voices_without_reading_them() {
    let root = std::env::temp_dir().join(format!("tinfer-styletts2-manifest-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("model.toml"),
        r#"architecture_id = "styletts2-istftnet-v1"
sample_rate = 24000
default_language = "pl"
supported_languages = ["pl", "en"]
symbols = ["$", "a", "a"]
"#,
    )
    .unwrap();
    fs::create_dir(root.join("voices")).unwrap();
    fs::write(root.join("voices/alice.tinf"), b"not-read-during-manifest-load").unwrap();

    let manifest = Manifest::load(&root).unwrap();

    assert_eq!(manifest.architecture_id, "styletts2-istftnet-v1");
    assert_eq!(manifest.voices["alice"], root.join("voices/alice.tinf"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn diffusion_schedule_repeats_the_minimum_after_requested_steps() {
    let three = diffusion_schedule(3);
    let start = 3.0_f32.powf(1.0 / 9.0);
    let end = 0.0001_f32.powf(1.0 / 9.0);
    let midpoint = ((start + end) / 2.0).powi(9);
    assert_eq!(three.len(), 6);
    assert!((three[0] - 3.0).abs() < 1e-5);
    assert!((three[1] - midpoint).abs() < 1e-5);
    assert_eq!(&three[2..], &[0.0001; 4]);

    let five = diffusion_schedule(5);
    assert!(five.windows(2).take(4).all(|values| values[0] > values[1]));
    assert_eq!(five[4], 0.0001);
    assert_eq!(five[5], 0.0001);
}

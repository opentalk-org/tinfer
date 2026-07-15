use super::assert_cases;

#[test]
fn my_matches_upstream() {
    assert_cases("my", false, None, CASES);
}

const CASES: &[(&str, &[&str])] = &[("ခင္ဗ်ားနာမည္ဘယ္လိုေခၚလဲ။၇ွင္ေနေကာင္းလား။", &["ခင္ဗ်ားနာမည္ဘယ္လိုေခၚလဲ။", "၇ွင္ေနေကာင္းလား။"])];

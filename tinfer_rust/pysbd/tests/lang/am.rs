use super::assert_cases;

#[test]
fn am_matches_upstream() {
    assert_cases("am", false, None, CASES);
}

const CASES: &[(&str, &[&str])] = &[("እንደምን አለህ፧መልካም ቀን ይሁንልህ።እባክሽ ያልሽዉን ድገሚልኝ።", &["እንደምን አለህ፧", "መልካም ቀን ይሁንልህ።", "እባክሽ ያልሽዉን ድገሚልኝ።"])];

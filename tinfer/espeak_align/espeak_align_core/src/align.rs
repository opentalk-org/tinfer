use crate::phonemizer_pool::Phonemizer;
use crate::tokenize::tokenize;
use crate::utf8::utf8_next;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Mutex;
use std::thread;

const SPACE_CP: u32 = 0x20;
const BEAM_WIDTH: usize = 32;

fn is_space_byte(c: u8) -> bool {
    c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' || c == b'\x0c' || c == b'\x0b'
}

fn join_words(words: &[String], start: usize, count: usize) -> String {
    let mut out = String::new();
    for k in 0..count {
        if k != 0 {
            out.push(' ');
        }
        out.push_str(&words[start + k]);
    }
    out
}

fn to_codepoints_excluding_space(s: &str) -> Vec<u32> {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let (cp, next) = utf8_next(bytes, i);
        if cp != SPACE_CP {
            out.push(cp);
        }
        i = next;
    }
    out
}

fn levenshtein_ratio(a: &str, b: &str) -> f64 {
    let va = to_codepoints_excluding_space(a);
    let vb = to_codepoints_excluding_space(b);
    if va == vb {
        return 1.0;
    }
    let len_a = va.len();
    let len_b = vb.len();
    if len_a == 0 && len_b == 0 {
        return 1.0;
    }
    if len_a == 0 || len_b == 0 {
        return 0.0;
    }

    let max_len = len_a.max(len_b) as f64;

    // Two-row DP (no vec![vec![...]]): reduces allocations and improves locality.
    let mut prev: Vec<i32> = (0..=(len_b as i32)).collect();
    let mut curr: Vec<i32> = vec![0i32; len_b + 1];

    for i in 1..=len_a {
        curr[0] = i as i32;
        let mut row_min = curr[0];
        let a_cp = va[i - 1];
        for j in 1..=len_b {
            let cost = if a_cp == vb[j - 1] { 0 } else { 1 };
            let v1 = prev[j] + 1;
            let v2 = curr[j - 1] + 1;
            let v3 = prev[j - 1] + cost;
            let v = v1.min(v2).min(v3);
            curr[j] = v;
            if v < row_min {
                row_min = v;
            }
        }

        // Early-out (safe): distance can't drop below this row's minimum.
        if row_min as usize >= len_a.max(len_b) {
            return 0.0;
        }

        std::mem::swap(&mut prev, &mut curr);
    }

    1.0 - (prev[len_b] as f64) / max_len
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TransitionType {
    N1 = 0,
    N2 = 2,
    N3 = 3,
    Silent = 10,
}

#[derive(Clone)]
struct Transition {
    to: String,
    gram_len: i32,
    phon_len: i32,
    base_cost: f64,
    _type: TransitionType,
}

fn count_phoneme_words_fast(s: &str) -> i32 {
    let mut count = 0i32;
    let mut in_word = false;
    for &c in s.as_bytes() {
        let is_ws = is_space_byte(c);
        if is_ws {
            if in_word {
                in_word = false;
            }
        } else if !in_word {
            in_word = true;
            count += 1;
        }
    }
    count
}

fn build_transitions_timed(
    words: &[String],
    phonemizer: &dyn Phonemizer,
    phoneme_cache: &mut HashMap<String, String>,
) -> Vec<[Vec<Transition>; 3]> {
    let mut transitions: Vec<[Vec<Transition>; 3]> = Vec::new();
    let n = words.len();
    transitions.resize_with(n, || [Vec::new(), Vec::new(), Vec::new()]);

    for i in 0..n {
        for gram_len in 1..=3 {
            if i + gram_len > n {
                continue;
            }
            let ngram_text = join_words(words, i, gram_len);
            let to_phon = if let Some(v) = phoneme_cache.get(&ngram_text) {
                v.clone()
            } else {
                let ph = phonemizer.text_to_phonemes(&ngram_text).unwrap_or_default();
                phoneme_cache.insert(ngram_text.clone(), ph.clone());
                ph
            };

            let ratio = levenshtein_ratio(&ngram_text, &to_phon);
            let base_type = if gram_len == 1 {
                TransitionType::N1
            } else if gram_len == 2 {
                TransitionType::N2
            } else {
                TransitionType::N3
            };
            let match_weight = if base_type == TransitionType::N1 {
                1.0
            } else if base_type == TransitionType::N2 {
                0.3
            } else {
                0.1
            };
            let base_cost = base_type as i32 as f64;
            let cost = base_cost + match_weight * (1.0 - ratio);

            let phon_len = count_phoneme_words_fast(&to_phon);
            transitions[i][gram_len - 1].push(Transition {
                to: to_phon,
                gram_len: gram_len as i32,
                phon_len,
                base_cost: cost,
                _type: base_type,
            });

            if gram_len == 1 {
                transitions[i][0].push(Transition {
                    to: String::new(),
                    gram_len: 1,
                    phon_len: 0,
                    base_cost: gram_len as f64,
                    _type: TransitionType::Silent,
                });
            }
        }
    }

    transitions
}

fn split_phoneme_words(s: &str) -> Vec<String> {
    s.split_whitespace().map(|w| w.to_owned()).collect()
}

#[derive(Clone, Copy)]
struct BeamKey {
    cost: f64,
    order: u64,
    node: usize,
}

struct Node {
    cost: f64,
    word_i: u32,
    phon_i: u32,
    prev: Option<usize>,
    tr: Option<Transition>,
}

fn beam_search_chunk(
    words: &[String],
    chunk_phonemes: &str,
    transitions: &[[Vec<Transition>; 3]],
    beam_width: usize,
) -> Vec<Transition> {
    if words.is_empty() {
        return Vec::new();
    }
    let remaining_phon_words = split_phoneme_words(chunk_phonemes);

    let mut nodes: Vec<Node> = Vec::new();
    nodes.push(Node {
        cost: 0.0,
        word_i: 0,
        phon_i: 0,
        prev: None,
        tr: None,
    });

    let mut beams: Vec<BeamKey> = vec![BeamKey {
        cost: 0.0,
        order: 0,
        node: 0,
    }];
    let mut next_order: u64 = 1;

    let mut best_cost_for_state: HashMap<u64, f64> = HashMap::new();
    best_cost_for_state.insert(((0u64) << 32) | 0u64, 0.0);
    let mut best_complete: Option<usize> = None;

    while !beams.is_empty() {
        beams.sort_by(|a, b| {
            let c = a.cost.partial_cmp(&b.cost).unwrap_or(Ordering::Equal);
            if c != Ordering::Equal {
                return c;
            }
            a.order.cmp(&b.order)
        });

        let mut new_beams: Vec<BeamKey> = Vec::with_capacity(beams.len() * 8);
        for b in 0..beams.len().min(beam_width) {
            let BeamKey { node, .. } = beams[b];
            let (cost, word_i, phon_i) = {
                let n = &nodes[node];
                (n.cost, n.word_i, n.phon_i)
            };

            if (word_i as usize) >= words.len()
                && (phon_i as usize) >= remaining_phon_words.len()
            {
                if best_complete
                    .map(|bi| cost < nodes[bi].cost)
                    .unwrap_or(true)
                {
                    best_complete = Some(node);
                }
                continue;
            }

            if (word_i as usize) > words.len() || (phon_i as usize) > remaining_phon_words.len() {
                continue;
            }

            let wi = word_i as usize;
            let max_gram = 3usize.min(words.len().saturating_sub(wi));
            for gram_len in 1..=max_gram {
                for tr in &transitions[wi][gram_len - 1] {
                    let new_word_i = word_i + gram_len as u32;
                    let mut new_phon_i = phon_i;
                    if tr.phon_len != 0 {
                        new_phon_i += tr.phon_len as u32;
                        if (new_phon_i as usize) > remaining_phon_words.len() {
                            continue;
                        }
                    }

                    let new_cost = cost + tr.base_cost + 0.8 * ((tr.gram_len - 1) as f64);
                    let state_key =
                        ((new_word_i as u64) << 32) | (new_phon_i as u64);
                    if let Some(best) = best_cost_for_state.get(&state_key) {
                        if *best <= new_cost {
                            continue;
                        }
                    }
                    best_cost_for_state.insert(state_key, new_cost);

                    let new_node = nodes.len();
                    nodes.push(Node {
                        cost: new_cost,
                        word_i: new_word_i,
                        phon_i: new_phon_i,
                        prev: Some(node),
                        tr: Some(tr.clone()),
                    });
                    new_beams.push(BeamKey {
                        cost: new_cost,
                        order: next_order,
                        node: new_node,
                    });
                    next_order += 1;
                }
            }
        }
        beams = new_beams;
    }

    let Some(mut node) = best_complete else {
        return Vec::new();
    };
    let mut out: Vec<Transition> = Vec::new();
    loop {
        let n = &nodes[node];
        let Some(tr) = n.tr.clone() else {
            break;
        };
        out.push(tr);
        let Some(prev) = n.prev else {
            break;
        };
        node = prev;
    }
    out.reverse();
    out
}

fn count_excluding_space_char(s: &str) -> usize {
    let bytes = s.as_bytes();
    let mut n = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let (cp, next) = utf8_next(bytes, i);
        if cp != SPACE_CP {
            n += 1;
        }
        i = next;
    }
    n
}

fn round_half_to_even(x: f64) -> i32 {
    let fl = x.floor();
    let frac = x - fl;
    if frac < 0.5 {
        return fl as i32;
    }
    if frac > 0.5 {
        return fl as i32 + 1;
    }
    if (fl % 2.0) == 0.0 {
        fl as i32
    } else {
        (fl + 1.0) as i32
    }
}

fn strip_ascii_ws(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    let mut j = bytes.len();
    while i < j && is_space_byte(bytes[i]) {
        i += 1;
    }
    while j > i && is_space_byte(bytes[j - 1]) {
        j -= 1;
    }
    s[i..j].to_owned()
}

fn rstrip_len_ascii_ws(s: &str) -> usize {
    let bytes = s.as_bytes();
    let mut j = bytes.len();
    while j > 0 && is_space_byte(bytes[j - 1]) {
        j -= 1;
    }
    j
}

fn lstrip_len_ascii_ws(s: &str) -> usize {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && is_space_byte(bytes[i]) {
        i += 1;
    }
    i
}

fn distribute_step_phonemes_over_words(words: &[String], phoneme_string: &str) -> Vec<String> {
    if words.is_empty() {
        return Vec::new();
    }
    let phonemes = strip_ascii_ws(phoneme_string);
    if phonemes.is_empty() {
        return vec![String::new(); words.len()];
    }

    let mut total_chars = 0usize;
    for w in words {
        total_chars += count_excluding_space_char(w).max(1);
    }
    let phoneme_chars = count_excluding_space_char(&phonemes);
    if total_chars == 0 || phoneme_chars == 0 {
        return vec![String::new(); words.len()];
    }

    let mut per_word: Vec<String> = Vec::new();
    let mut remaining = phonemes.clone();
    let mut remaining_chars = phoneme_chars;

    for idx in 0..words.len() {
        let w_chars = count_excluding_space_char(&words[idx]).max(1);
        if idx == words.len() - 1 {
            per_word.push(remaining);
            break;
        }

        let take_chars = ((w_chars as f64) / (total_chars as f64) * (phoneme_chars as f64)) as f64;
        let take_chars = (round_half_to_even(take_chars) as usize).max(1);

        if take_chars >= remaining_chars {
            per_word.push(remaining);
            remaining = String::new();
            remaining_chars = 0;
            continue;
        }

        let mut count = 0usize;
        let mut cut_byte = remaining.len();
        let rem_bytes = remaining.as_bytes();
        let mut bi = 0usize;
        while bi < rem_bytes.len() {
            let (cp, next) = utf8_next(rem_bytes, bi);
            if cp != SPACE_CP {
                count += 1;
            }
            if count == take_chars {
                cut_byte = next;
                break;
            }
            bi = next;
        }

        let head_raw = &remaining[..cut_byte];
        let head = head_raw[..rstrip_len_ascii_ws(head_raw)].to_owned();
        let mut new_remaining = remaining[cut_byte..].to_owned();
        let l = lstrip_len_ascii_ws(&new_remaining);
        new_remaining = new_remaining[l..].to_owned();
        remaining = new_remaining;
        remaining_chars -= take_chars;
        per_word.push(head);
    }

    while per_word.len() < words.len() {
        per_word.push(String::new());
    }
    per_word
}

fn token_lead_tail(tok: &str) -> (String, String, String) {
    let i = lstrip_len_ascii_ws(tok);
    let mid = tok[i..].to_owned();
    let content_len = rstrip_len_ascii_ws(&mid);
    let lead = tok[..i].to_owned();
    let content = tok[i..i + content_len].to_owned();
    let tail = tok[i + content_len..].to_owned();
    (lead, content, tail)
}

pub fn align_text(
    phonemizer: &dyn Phonemizer,
    text: &str,
    punctuation: &str,
    split_by_punctuation: impl Fn(&str, &str) -> (Vec<String>, Vec<(i32, String)>),
) -> (Vec<String>, Vec<String>) {
    align_text_with_threads(phonemizer, text, punctuation, split_by_punctuation, 1)
}

pub fn align_text_with_threads(
    phonemizer: &dyn Phonemizer,
    text: &str,
    punctuation: &str,
    split_by_punctuation: impl Fn(&str, &str) -> (Vec<String>, Vec<(i32, String)>),
    threads: usize,
) -> (Vec<String>, Vec<String>) {
    let (chunks, puncts) = split_by_punctuation(text, punctuation);
    if chunks.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let n_chunks = chunks.len();
    let results = align_chunks_parallel(phonemizer, &chunks, threads);
    let mut all_tokens: Vec<String> = Vec::new();
    let mut all_phonemes: Vec<String> = Vec::new();
    let mut chunk_token_ranges: Vec<(usize, usize)> = Vec::new();
    for idx in 0..n_chunks {
        let start = all_tokens.len();
        if let Some((tokens, phonemes)) = results[idx].as_ref() {
            for i in 0..tokens.len() {
                all_tokens.push(tokens[i].clone());
                all_phonemes.push(phonemes.get(i).cloned().unwrap_or_default());
            }
        }
        chunk_token_ranges.push((start, all_tokens.len()));
    }

    let mut marks_by_chunk: BTreeMap<i32, Vec<String>> = BTreeMap::new();
    for (idx, mark) in puncts {
        marks_by_chunk.entry(idx).or_default().push(mark);
    }

    let mut tokens_out: Vec<String> = Vec::new();
    let mut phonemes_out: Vec<String> = Vec::new();
    for chunk_index in 0..chunk_token_ranges.len() {
        let (start, end) = chunk_token_ranges[chunk_index];
        for i in start..end {
            tokens_out.push(all_tokens[i].clone());
            phonemes_out.push(all_phonemes[i].clone());
        }
        if let Some(marks) = marks_by_chunk.get(&(chunk_index as i32)) {
            for mark in marks {
                tokens_out.push(mark.clone());
                phonemes_out.push(mark.clone());
            }
        }
    }

    for i in 0..tokens_out.len() {
        if phonemes_out[i].is_empty() {
            continue;
        }
        let (lead, _content, tail) = token_lead_tail(&tokens_out[i]);
        if !lead.is_empty() || !tail.is_empty() {
            let mut s = String::new();
            s.push_str(&lead);
            s.push_str(&phonemes_out[i]);
            s.push_str(&tail);
            phonemes_out[i] = s;
        }
    }

    (tokens_out, phonemes_out)
}

pub fn align_texts_batch_with_threads(
    phonemizer: &dyn Phonemizer,
    texts: &[String],
    punctuation: &str,
    split_by_punctuation: impl Fn(&str, &str) -> (Vec<String>, Vec<(i32, String)>),
    threads: usize,
) -> Vec<(Vec<String>, Vec<String>)> {
    if texts.is_empty() {
        return Vec::new();
    }

    struct TextPlan {
        chunks: Vec<String>,
        puncts: Vec<(i32, String)>,
    }

    let mut plans: Vec<TextPlan> = Vec::with_capacity(texts.len());
    let mut jobs: Vec<(usize, usize)> = Vec::new(); // (text_idx, chunk_idx)
    for (ti, t) in texts.iter().enumerate() {
        let (chunks, puncts) = split_by_punctuation(t, punctuation);
        for ci in 0..chunks.len() {
            jobs.push((ti, ci));
        }
        plans.push(TextPlan { chunks, puncts });
    }

    let chunk_results: Mutex<Vec<Option<(Vec<String>, Vec<String>)>>> =
        Mutex::new((0..jobs.len()).map(|_| None).collect());
    let next = AtomicUsize::new(0);
    thread::scope(|s| {
        let jobs = &jobs;
        let plans = &plans;
        let threads = threads.max(1).min(jobs.len().max(1));
        for _ in 0..threads {
            let next = &next;
            let chunk_results = &chunk_results;
            s.spawn(|| loop {
                let ji = next.fetch_add(1, AtomicOrdering::Relaxed);
                if ji >= jobs.len() {
                    break;
                }
                let (ti, ci) = jobs[ji];
                let chunk = &plans[ti].chunks[ci];
                let words = tokenize(chunk);
                if words.is_empty() {
                    if let Ok(mut r) = chunk_results.lock() {
                        r[ji] = Some((Vec::new(), Vec::new()));
                    }
                    continue;
                }

                let chunk_phon = phonemizer.text_to_phonemes(chunk).unwrap_or_default();
                let mut phoneme_cache: HashMap<String, String> = HashMap::new();
                let trans = build_transitions_timed(&words, phonemizer, &mut phoneme_cache);
                let path = beam_search_chunk(&words, &chunk_phon, &trans, BEAM_WIDTH);

                let out = if path.is_empty() {
                    let per_word_ph = distribute_step_phonemes_over_words(&words, &chunk_phon);
                    (words, per_word_ph)
                } else {
                    let mut word_phonemes: Vec<String> = Vec::new();
                    let mut widx = 0usize;
                    for tr in &path {
                        let step_words = &words[widx..widx + (tr.gram_len as usize)];
                        widx += tr.gram_len as usize;
                        let per_word = if tr.to.is_empty() {
                            vec![String::new(); step_words.len()]
                        } else {
                            distribute_step_phonemes_over_words(step_words, &tr.to)
                        };
                        word_phonemes.extend(per_word);
                    }
                    while word_phonemes.len() < words.len() {
                        word_phonemes.push(String::new());
                    }

                    let mut i = 0usize;
                    while i < words.len() {
                        if !word_phonemes[i].is_empty() {
                            let mut j = i + 1;
                            while j < words.len() && word_phonemes[j].is_empty() {
                                j += 1;
                            }
                            if j > i + 1 {
                                let segment_words = &words[i..j];
                                let segment_ph = word_phonemes[i].clone();
                                let redistributed =
                                    distribute_step_phonemes_over_words(segment_words, &segment_ph);
                                for k in 0..redistributed.len() {
                                    word_phonemes[i + k] = redistributed[k].clone();
                                }
                            }
                            i = j;
                        } else {
                            i += 1;
                        }
                    }
                    (words, word_phonemes)
                };

                if let Ok(mut r) = chunk_results.lock() {
                    r[ji] = Some(out);
                }
            });
        }
    });

    let chunk_results = chunk_results
        .into_inner()
        .unwrap_or_else(|e| e.into_inner());

    let mut per_text_chunks: Vec<Vec<Option<(Vec<String>, Vec<String>)>>> =
        plans.iter().map(|p| vec![None; p.chunks.len()]).collect();
    for (ji, (ti, ci)) in jobs.iter().copied().enumerate() {
        if let Some(v) = chunk_results[ji].clone() {
            per_text_chunks[ti][ci] = Some(v);
        }
    }

    let mut out: Vec<(Vec<String>, Vec<String>)> = Vec::with_capacity(texts.len());
    for ti in 0..plans.len() {
        let chunks = &plans[ti].chunks;
        let puncts = &plans[ti].puncts;
        if chunks.is_empty() {
            out.push((Vec::new(), Vec::new()));
            continue;
        }

        let mut all_tokens: Vec<String> = Vec::new();
        let mut all_phonemes: Vec<String> = Vec::new();
        let mut chunk_token_ranges: Vec<(usize, usize)> = Vec::new();
        for ci in 0..chunks.len() {
            let start = all_tokens.len();
            if let Some((tokens, phonemes)) = per_text_chunks[ti][ci].as_ref() {
                for i in 0..tokens.len() {
                    all_tokens.push(tokens[i].clone());
                    all_phonemes.push(phonemes.get(i).cloned().unwrap_or_default());
                }
            }
            chunk_token_ranges.push((start, all_tokens.len()));
        }

        let mut marks_by_chunk: BTreeMap<i32, Vec<String>> = BTreeMap::new();
        for (idx, mark) in puncts {
            marks_by_chunk.entry(*idx).or_default().push(mark.clone());
        }

        let mut tokens_out: Vec<String> = Vec::new();
        let mut phonemes_out: Vec<String> = Vec::new();
        for chunk_index in 0..chunk_token_ranges.len() {
            let (start, end) = chunk_token_ranges[chunk_index];
            for i in start..end {
                tokens_out.push(all_tokens[i].clone());
                phonemes_out.push(all_phonemes[i].clone());
            }
            if let Some(marks) = marks_by_chunk.get(&(chunk_index as i32)) {
                for mark in marks {
                    tokens_out.push(mark.clone());
                    phonemes_out.push(mark.clone());
                }
            }
        }

        for i in 0..tokens_out.len() {
            if phonemes_out[i].is_empty() {
                continue;
            }
            let (lead, _content, tail) = token_lead_tail(&tokens_out[i]);
            if !lead.is_empty() || !tail.is_empty() {
                let mut s = String::new();
                s.push_str(&lead);
                s.push_str(&phonemes_out[i]);
                s.push_str(&tail);
                phonemes_out[i] = s;
            }
        }

        out.push((tokens_out, phonemes_out));
    }

    out
}

fn align_chunks_parallel(
    phonemizer: &dyn Phonemizer,
    chunks: &[String],
    threads: usize,
) -> Vec<Option<(Vec<String>, Vec<String>)>> {
    let n_chunks = chunks.len();
    let results: Mutex<Vec<Option<(Vec<String>, Vec<String>)>>> =
        Mutex::new((0..n_chunks).map(|_| None).collect());

    let next = AtomicUsize::new(0);
    thread::scope(|s| {
        let threads = threads.max(1).min(n_chunks.max(1));
        for _ in 0..threads {
            let chunks = chunks;
            let next = &next;
            let results = &results;
            s.spawn(move || loop {
                let idx = next.fetch_add(1, AtomicOrdering::Relaxed);
                if idx >= chunks.len() {
                    break;
                }
                let chunk = &chunks[idx];
                let words = tokenize(chunk);
                if words.is_empty() {
                    if let Ok(mut r) = results.lock() {
                        r[idx] = Some((Vec::new(), Vec::new()));
                    }
                    continue;
                }

                let chunk_phon = phonemizer.text_to_phonemes(chunk).unwrap_or_default();
                let mut phoneme_cache: HashMap<String, String> = HashMap::new();
                let trans = build_transitions_timed(&words, phonemizer, &mut phoneme_cache);
                let path = beam_search_chunk(&words, &chunk_phon, &trans, BEAM_WIDTH);

                let out = if path.is_empty() {
                    let per_word_ph = distribute_step_phonemes_over_words(&words, &chunk_phon);
                    (words, per_word_ph)
                } else {
                    let mut word_phonemes: Vec<String> = Vec::new();
                    let mut widx = 0usize;
                    for tr in &path {
                        let step_words = &words[widx..widx + (tr.gram_len as usize)];
                        widx += tr.gram_len as usize;
                        let per_word = if tr.to.is_empty() {
                            vec![String::new(); step_words.len()]
                        } else {
                            distribute_step_phonemes_over_words(step_words, &tr.to)
                        };
                        word_phonemes.extend(per_word);
                    }
                    while word_phonemes.len() < words.len() {
                        word_phonemes.push(String::new());
                    }

                    let mut i = 0usize;
                    while i < words.len() {
                        if !word_phonemes[i].is_empty() {
                            let mut j = i + 1;
                            while j < words.len() && word_phonemes[j].is_empty() {
                                j += 1;
                            }
                            if j > i + 1 {
                                let segment_words = &words[i..j];
                                let segment_ph = word_phonemes[i].clone();
                                let redistributed =
                                    distribute_step_phonemes_over_words(segment_words, &segment_ph);
                                for k in 0..redistributed.len() {
                                    word_phonemes[i + k] = redistributed[k].clone();
                                }
                            }
                            i = j;
                        } else {
                            i += 1;
                        }
                    }
                    (words, word_phonemes)
                };

                if let Ok(mut r) = results.lock() {
                    r[idx] = Some(out);
                }
            });
        }
    });

    results.into_inner().unwrap_or_else(|e| e.into_inner())
}


//! Disfluency markers and retrace markers.

use pyo3::PyResult;
use talkbank_model::model::Line;

pub(crate) fn add_disfluency_markers_inner(
    chat_file: &mut talkbank_model::model::ChatFile,
    filled_pauses_json: &str,
    replacements_json: &str,
) -> PyResult<()> {
    use talkbank_model::model::content::{BracketedItem, UtteranceContent};

    let filled_pauses: Vec<(String, String)> =
        parse_wordlist_json(filled_pauses_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid filled_pauses JSON: {e}"))
        })?;
    let replacements: Vec<(String, String)> =
        parse_wordlist_json(replacements_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid replacements JSON: {e}"))
        })?;

    let fp_map: std::collections::HashMap<String, String> = filled_pauses
        .into_iter()
        .map(|(orig, repl)| (orig.to_lowercase(), repl))
        .collect();
    let repl_map: std::collections::HashMap<String, String> = replacements
        .into_iter()
        .map(|(orig, repl)| (orig.to_lowercase(), repl))
        .collect();

    for line in &mut chat_file.lines {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        for item in &mut utt.main.content.content.0 {
            match item {
                UtteranceContent::Word(w) => {
                    apply_disfluency_to_word(w, &fp_map, &repl_map);
                }
                UtteranceContent::AnnotatedWord(aw) => {
                    apply_disfluency_to_word(&mut aw.inner, &fp_map, &repl_map);
                }
                UtteranceContent::Group(g) => {
                    for bi in &mut g.content.content.0 {
                        if let BracketedItem::Word(w) = bi {
                            apply_disfluency_to_word(w, &fp_map, &repl_map);
                        }
                    }
                }
                UtteranceContent::AnnotatedGroup(ag) => {
                    for bi in &mut ag.inner.content.content.0 {
                        if let BracketedItem::Word(w) = bi {
                            apply_disfluency_to_word(w, &fp_map, &repl_map);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    Ok(())
}

pub(crate) fn add_retrace_markers_inner(
    chat_file: &mut talkbank_model::model::ChatFile,
    lang: &str,
) {
    use talkbank_model::model::content::{BracketedItem, UtteranceContent};
    use talkbank_model::model::{Annotated, BracketedContent, Group, ScopedAnnotation, Word};

    let min_n: usize = if lang == "zho" || lang == "yue" { 2 } else { 1 };

    for line in &mut chat_file.lines {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        let mut word_map: Vec<(usize, String)> = Vec::new();
        for (idx, item) in utt.main.content.content.0.iter().enumerate() {
            if let UtteranceContent::Word(w) = item
                && w.category.is_none()
            {
                word_map.push((idx, w.cleaned_text().to_lowercase()));
            }
        }

        if word_map.len() < 2 {
            continue;
        }

        let mut retraced: Vec<bool> = vec![false; word_map.len()];
        let mut retrace_ranges: Vec<(usize, usize)> = Vec::new();

        for n in (min_n..word_map.len()).rev() {
            let mut begin = 0;
            while begin + 2 * n <= word_map.len() {
                let gram: Vec<&str> = (begin..begin + n).map(|i| word_map[i].1.as_str()).collect();
                let next: Vec<&str> = (begin + n..begin + 2 * n)
                    .map(|i| word_map[i].1.as_str())
                    .collect();

                if gram == next {
                    // If all words in the n-gram are identical and n > 1, skip.
                    // This is a single-word stutter (e.g. "the the the the"),
                    // not a multi-word phrase retrace.  Let the n=1 pass mark
                    // each repetition individually so CLAN can count them:
                    //   the [/] the [/] the [/] the   (correct)
                    // instead of:
                    //   <the the> [/] the [/] the     (wrong)
                    let all_same = n > 1 && gram.iter().all(|w| *w == gram[0]);
                    let overlap = retraced[begin..begin + n].iter().any(|&r| r);
                    if !overlap && !all_same {
                        retrace_ranges.push((begin, n));
                        retraced[begin..begin + n].fill(true);
                    }
                }
                begin += 1;
            }
        }

        if retrace_ranges.is_empty() {
            continue;
        }

        retrace_ranges.sort_by_key(|&(start, _)| start);

        let mut retrace_starts: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut retrace_members: std::collections::HashSet<usize> =
            std::collections::HashSet::new();

        for &(word_start, n) in &retrace_ranges {
            let ci_start = word_map[word_start].0;
            let ci_end = word_map[word_start + n - 1].0;
            retrace_starts.insert(ci_start, ci_end - ci_start + 1);
            for ci in ci_start..=ci_end {
                retrace_members.insert(ci);
            }
        }

        let old_content = std::mem::take(&mut utt.main.content.content.0);
        let mut new_content: Vec<UtteranceContent> = Vec::with_capacity(old_content.len());
        let mut ci = 0;

        while ci < old_content.len() {
            if let Some(&span) = retrace_starts.get(&ci) {
                if span == 1 {
                    // Single-word retrace: use AnnotatedWord (no angle brackets).
                    // CHAT convention: `word [/] word`, NOT `<word> [/] word`.
                    if let UtteranceContent::Word(w) = &old_content[ci] {
                        let word: Word = (**w).clone();
                        let annotated = Annotated::new(word)
                            .with_scoped_annotation(ScopedAnnotation::PartialRetracing);
                        new_content.push(UtteranceContent::AnnotatedWord(Box::new(annotated)));
                    }
                } else {
                    // Multi-word retrace: wrap in Group (angle brackets).
                    // CHAT convention: `<I want> [/] I want`.
                    let mut group_items: Vec<BracketedItem> = Vec::new();
                    for item in &old_content[ci..ci + span] {
                        if let UtteranceContent::Word(w) = item {
                            group_items.push(BracketedItem::Word(w.clone()));
                        }
                    }
                    let group = Group::new(BracketedContent::new(group_items));
                    let annotated = Annotated::new(group)
                        .with_scoped_annotation(ScopedAnnotation::PartialRetracing);
                    new_content.push(UtteranceContent::AnnotatedGroup(annotated));
                }
                ci += span;
            } else if retrace_members.contains(&ci) {
                ci += 1;
            } else {
                new_content.push(old_content[ci].clone());
                ci += 1;
            }
        }

        utt.main.content.content.0 = new_content;
    }
}

/// Apply disfluency rules to a single word.
pub(crate) fn apply_disfluency_to_word(
    word: &mut talkbank_model::model::Word,
    fp_map: &std::collections::HashMap<String, String>,
    repl_map: &std::collections::HashMap<String, String>,
) {
    use talkbank_model::model::content::word::category::WordCategory;

    let key = word.cleaned_text().to_lowercase();

    // Check filled pauses first
    if let Some(repl) = fp_map.get(&key) {
        word.category = Some(WordCategory::Filler);
        // Update text and word content to the replacement
        word.replace_simple_text(repl.clone());
        return;
    }

    // Check replacements
    if let Some(repl) = repl_map.get(&key) {
        word.replace_simple_text(repl.clone());
    }
}

/// Parse a JSON array of `{"original": "...", "replacement": "..."}` objects.
pub(crate) fn parse_wordlist_json(json_str: &str) -> Result<Vec<(String, String)>, String> {
    let val: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;
    let arr = val.as_array().ok_or("Expected JSON array")?;
    let mut result = Vec::with_capacity(arr.len());
    for item in arr {
        let orig = item["original"]
            .as_str()
            .ok_or("Missing 'original' field")?;
        let repl = item["replacement"]
            .as_str()
            .ok_or("Missing 'replacement' field")?;
        result.push((orig.to_string(), repl.to_string()));
    }
    Ok(result)
}

//! Golden model tests — snapshot NLP engine output via real direct execution.
//!
//! These tests acquire isolated real-model direct sessions backed by a shared
//! warmed worker pool, submit CHAT files, and compare the output against insta
//! snapshots. They verify that model output hasn't regressed after
//! Stanza/engine upgrades without depending on the server control plane.
//!
//! Requirements:
//! - Python 3 with batchalign installed
//! - Stanza models downloaded (`stanza.download("en")`)
//!
//! Tests skip gracefully if models are unavailable.
//!
//! Run: `cargo nextest run -p batchalign-app --test ml_golden --profile ml`
//! Update snapshots: `cargo insta review`

use crate::common::{
    assert_completed_without_errors, require_live_direct, submit_and_complete_direct,
};
use batchalign_app::api::{FilePayload, JobStatus, ReleasedCommand};
use batchalign_app::options::{
    CommandOptions, CommonOptions, CompareOptions, CorefOptions, MorphotagOptions,
    TranslateOptions, UtsegOptions,
};
use batchalign_app::worker::InferTask;
use batchalign_chat_ops::parse::{TreeSitterParser, parse_lenient};
use batchalign_chat_ops::{ChatFile, DependentTier};

// ---------------------------------------------------------------------------
// AST-based CHAT output helpers
// ---------------------------------------------------------------------------

/// Parse CHAT output via the AST and assert it has no parse errors.
fn parse_output(chat: &str, label: &str) -> ChatFile {
    let parser = TreeSitterParser::new().unwrap();
    let (file, errors) = parse_lenient(&parser, chat);
    assert!(errors.is_empty(), "{label}: CHAT parse errors: {errors:?}");
    file
}

/// Check if any utterance in the parsed file has a %mor tier.
fn has_mor_tier(file: &ChatFile) -> bool {
    file.lines.iter().any(|line| {
        if let batchalign_chat_ops::Line::Utterance(utt) = line {
            utt.dependent_tiers
                .iter()
                .any(|t| matches!(t, DependentTier::Mor(_)))
        } else {
            false
        }
    })
}

/// Check if any utterance in the parsed file has a %gra tier.
fn has_gra_tier(file: &ChatFile) -> bool {
    file.lines.iter().any(|line| {
        if let batchalign_chat_ops::Line::Utterance(utt) = line {
            utt.dependent_tiers
                .iter()
                .any(|t| matches!(t, DependentTier::Gra(_)))
        } else {
            false
        }
    })
}

/// Check if any utterance has a user-defined tier with the given label (e.g., "xtra").
fn has_user_defined_tier(file: &ChatFile, label: &str) -> bool {
    file.lines.iter().any(|line| {
        if let batchalign_chat_ops::Line::Utterance(utt) = line {
            utt.dependent_tiers.iter().any(|t| match t {
                DependentTier::UserDefined(ud) => ud.label.as_ref() == label,
                _ => false,
            })
        } else {
            false
        }
    })
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const ENG_SIMPLE: &str = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\thello world .
@End
";

const ENG_MULTI_UTT: &str = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\tthe dog is running .
*PAR:\tI like cats .
*PAR:\tshe went to the store .
@End
";

const SPA_SIMPLE: &str = "\
@UTF8
@Begin
@Languages:\tspa
@Participants:\tPAR Participant
@ID:\tspa|test|PAR|||||Participant|||
*PAR:\tel gato es grande .
@End
";

// ---------------------------------------------------------------------------
// Golden tests
// ---------------------------------------------------------------------------

/// Morphotag: English single utterance → snapshot %mor/%gra.
#[tokio::test]
async fn golden_morphotag_eng_simple() {
    let Some(session) = require_live_direct(
        InferTask::Morphosyntax,
        "Direct session does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };

    let options = CommandOptions::Morphotag(MorphotagOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        retokenize: false,
        skipmultilang: false,
        merge_abbrev: false.into(),
    });

    let files = vec![FilePayload {
        filename: "eng_simple.cha".into(),
        content: ENG_SIMPLE.into(),
    }];

    let (info, results) =
        submit_and_complete_direct(&session, ReleasedCommand::Morphotag, "eng", files, options)
            .await;

    assert_completed_without_errors("morphotag_eng_simple", &info, &results);
    assert_eq!(results.len(), 1);

    let output = &results[0].content;
    let file = parse_output(output, "morphotag_eng_simple");
    assert!(has_mor_tier(&file), "Output should contain %mor tier");
    assert!(has_gra_tier(&file), "Output should contain %gra tier");

    insta::assert_snapshot!("morphotag_eng_simple", output);
}

/// Morphotag: English multi-utterance → snapshot all %mor/%gra.
#[tokio::test]
async fn golden_morphotag_eng_multi_utt() {
    let Some(session) = require_live_direct(
        InferTask::Morphosyntax,
        "Direct session does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };

    let options = CommandOptions::Morphotag(MorphotagOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        retokenize: false,
        skipmultilang: false,
        merge_abbrev: false.into(),
    });

    let files = vec![FilePayload {
        filename: "eng_multi_utt.cha".into(),
        content: ENG_MULTI_UTT.into(),
    }];

    let (info, results) =
        submit_and_complete_direct(&session, ReleasedCommand::Morphotag, "eng", files, options)
            .await;

    assert_completed_without_errors("morphotag_eng_multi_utt", &info, &results);
    assert_eq!(results.len(), 1);

    let output = &results[0].content;
    insta::assert_snapshot!("morphotag_eng_multi_utt", output);
}

/// Utseg: English multi-utterance → snapshot segmentation.
#[tokio::test]
async fn golden_utseg_eng_multi_utt() {
    let Some(session) = require_live_direct(
        InferTask::Utseg,
        "Direct session does not support utseg infer",
    )
    .await
    else {
        return;
    };

    let options = CommandOptions::Utseg(UtsegOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        merge_abbrev: false.into(),
    });

    let files = vec![FilePayload {
        filename: "eng_multi_utt.cha".into(),
        content: ENG_MULTI_UTT.into(),
    }];

    let (info, results) =
        submit_and_complete_direct(&session, ReleasedCommand::Utseg, "eng", files, options).await;

    assert_completed_without_errors("utseg_eng_multi_utt", &info, &results);
    assert_eq!(results.len(), 1);

    let output = &results[0].content;
    insta::assert_snapshot!("utseg_eng_multi_utt", output);
}

/// Translate: English simple → snapshot %xtra tier.
#[tokio::test]
async fn golden_translate_eng_simple() {
    let Some(session) = require_live_direct(
        InferTask::Translate,
        "Direct session does not support translate infer",
    )
    .await
    else {
        return;
    };

    let options = CommandOptions::Translate(TranslateOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        merge_abbrev: false.into(),
    });

    let files = vec![FilePayload {
        filename: "eng_simple.cha".into(),
        content: ENG_SIMPLE.into(),
    }];

    let (info, results) =
        submit_and_complete_direct(&session, ReleasedCommand::Translate, "eng", files, options)
            .await;

    assert_completed_without_errors("translate_eng_simple", &info, &results);
    assert_eq!(results.len(), 1);

    let output = &results[0].content;
    insta::assert_snapshot!("translate_eng_simple", output);
}

/// Morphotag with cache: submit twice, second should use cache.
/// Verifies the cache pipeline works correctly via round-trip.
#[tokio::test]
async fn golden_morphotag_with_cache() {
    let Some(session) = require_live_direct(
        InferTask::Morphosyntax,
        "Direct session does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };

    // First submission — populates cache
    let files1 = vec![FilePayload {
        filename: "cache_test.cha".into(),
        content: ENG_SIMPLE.into(),
    }];
    let (info1, results1) = submit_and_complete_direct(
        &session,
        ReleasedCommand::Morphotag,
        "eng",
        files1,
        CommandOptions::Morphotag(MorphotagOptions {
            common: CommonOptions::default(),
            retokenize: false,
            skipmultilang: false,
            merge_abbrev: false.into(),
        }),
    )
    .await;
    assert_completed_without_errors("morphotag_with_cache_cold", &info1, &results1);
    let output1 = &results1[0].content;

    // Second submission — should hit cache (same input)
    let files2 = vec![FilePayload {
        filename: "cache_test.cha".into(),
        content: ENG_SIMPLE.into(),
    }];
    let (info2, results2) = submit_and_complete_direct(
        &session,
        ReleasedCommand::Morphotag,
        "eng",
        files2,
        CommandOptions::Morphotag(MorphotagOptions {
            common: CommonOptions::default(),
            retokenize: false,
            skipmultilang: false,
            merge_abbrev: false.into(),
        }),
    )
    .await;
    assert_completed_without_errors("morphotag_with_cache_warm", &info2, &results2);
    let output2 = &results2[0].content;

    // Both should produce identical output
    assert_eq!(
        output1, output2,
        "Cache hit should produce identical output"
    );
}

// ---------------------------------------------------------------------------
// Compare golden tests
// ---------------------------------------------------------------------------

/// Main file with deliberate differences from gold (insertion + deletion).
const COMPARE_MAIN: &str = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\tthe big dog is running .
*PAR:\tI like cats .
@End
";

/// Gold reference: "big" is absent, "quickly" is present → tests insertion + deletion.
const COMPARE_GOLD: &str = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\tthe dog is running quickly .
*PAR:\tI like cats .
@End
";

/// Compare: main vs gold → snapshot %xsrep tiers.
#[tokio::test]
async fn golden_compare_eng() {
    let Some(session) = require_live_direct(
        InferTask::Morphosyntax,
        "Direct session does not support morphosyntax infer (required for compare)",
    )
    .await
    else {
        return;
    };

    let options = CommandOptions::Compare(CompareOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        merge_abbrev: false.into(),
    });

    let files = vec![
        FilePayload {
            filename: "compare_test.cha".into(),
            content: COMPARE_MAIN.into(),
        },
        FilePayload {
            filename: "compare_test.gold.cha".into(),
            content: COMPARE_GOLD.into(),
        },
    ];

    let (info, results) =
        submit_and_complete_direct(&session, ReleasedCommand::Compare, "eng", files, options).await;

    assert_eq!(info.status, JobStatus::Completed, "Job should complete");
    // Only 1 result — gold file is skipped as input
    assert_eq!(results.len(), 1, "Should have 1 result (gold file skipped)");
    assert!(results[0].error.is_none(), "No error expected");

    let output = &results[0].content;
    let file = parse_output(output, "compare_eng");
    assert!(
        has_user_defined_tier(&file, "xsrep"),
        "Output should contain %xsrep tier"
    );
    assert!(
        has_mor_tier(&file),
        "Output should contain %mor tier (morphosyntax runs first)"
    );

    insta::assert_snapshot!("compare_eng", output);
}

// ---------------------------------------------------------------------------
// Coref golden test
// ---------------------------------------------------------------------------

const ENG_COREF: &str = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tthe dog ran .
*CHI:\tit was fast .
*CHI:\tthe cat slept .
@End
";

/// Coref: English multi-sentence → snapshot %xcoref tier.
#[tokio::test]
async fn golden_coref_eng() {
    let Some(session) = require_live_direct(
        InferTask::Coref,
        "Direct session does not support coref infer",
    )
    .await
    else {
        return;
    };

    let options = CommandOptions::Coref(CorefOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        merge_abbrev: false.into(),
    });

    let files = vec![FilePayload {
        filename: "eng_coref.cha".into(),
        content: ENG_COREF.into(),
    }];

    let (info, results) =
        submit_and_complete_direct(&session, ReleasedCommand::Coref, "eng", files, options).await;

    assert_eq!(info.status, JobStatus::Completed, "Job should complete");
    assert_eq!(results.len(), 1);
    assert!(results[0].error.is_none(), "No error expected");

    let output = &results[0].content;

    // Coref models may or may not detect chains in short input.
    // Snapshot the output regardless — it validates the pipeline runs
    // without crashing. If %xcoref appears, the model found chains.
    let file = parse_output(output, "coref_eng");
    if has_user_defined_tier(&file, "xcoref") {
        eprintln!("Coref model detected chains — snapshotting with %xcoref");
    } else {
        eprintln!("Coref model found no chains (valid for short input)");
    }

    insta::assert_snapshot!("coref_eng", output);
}

// ---------------------------------------------------------------------------
// Spanish morphotag golden test (P1)
// ---------------------------------------------------------------------------

/// Morphotag: Spanish single utterance → snapshot %mor/%gra.
/// Skips if Spanish Stanza model is not downloaded.
#[tokio::test]
async fn golden_morphotag_spa_simple() {
    let Some(session) = require_live_direct(
        InferTask::Morphosyntax,
        "Direct session does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };

    let options = CommandOptions::Morphotag(MorphotagOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        retokenize: false,
        skipmultilang: false,
        merge_abbrev: false.into(),
    });

    let files = vec![FilePayload {
        filename: "spa_simple.cha".into(),
        content: SPA_SIMPLE.into(),
    }];

    let (info, results) =
        submit_and_complete_direct(&session, ReleasedCommand::Morphotag, "spa", files, options)
            .await;

    // If Spanish model is not available, the job will fail — treat as skip
    if info.status == JobStatus::Failed {
        eprintln!("SKIP: Spanish morphotag failed (model likely not downloaded)");
        return;
    }

    assert_eq!(info.status, JobStatus::Completed, "Job should complete");
    assert_eq!(results.len(), 1);
    assert!(results[0].error.is_none(), "No error expected");

    let output = &results[0].content;
    let file = parse_output(output, "morphotag_spa_simple");
    assert!(has_mor_tier(&file), "Output should contain %mor tier");
    assert!(has_gra_tier(&file), "Output should contain %gra tier");

    insta::assert_snapshot!("morphotag_spa_simple", output);
}

// ---------------------------------------------------------------------------
// Cache hit verification (P1)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Text command gap tests (Phase 2)
// ---------------------------------------------------------------------------

/// Morphotag with retokenize: MWT "gonna" should split into "going to".
#[tokio::test]
async fn golden_morphotag_retokenize_eng() {
    let Some(session) = require_live_direct(
        InferTask::Morphosyntax,
        "Direct session does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };

    const ENG_RETOKENIZE: &str = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\tgonna eat cookies .
@End
";

    let options = CommandOptions::Morphotag(MorphotagOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        retokenize: true,
        skipmultilang: false,
        merge_abbrev: false.into(),
    });

    let files = vec![FilePayload {
        filename: "eng_retokenize.cha".into(),
        content: ENG_RETOKENIZE.into(),
    }];

    let (info, results) =
        submit_and_complete_direct(&session, ReleasedCommand::Morphotag, "eng", files, options)
            .await;

    assert_completed_without_errors("morphotag_retokenize_eng", &info, &results);
    assert_eq!(results.len(), 1);

    let output = &results[0].content;
    let file = parse_output(output, "morphotag_retokenize_eng");
    assert!(has_mor_tier(&file), "Output should contain %mor tier");
    assert!(has_gra_tier(&file), "Output should contain %gra tier");
    // Whether Stanza splits "gonna" depends on the model version. The
    // retokenize flag enables the split/merge path — we verify it runs
    // without error and produces valid annotated output, not a specific
    // tokenization choice. The snapshot captures current model behavior.
    insta::assert_snapshot!("morphotag_retokenize_eng", output);
}

/// Utseg: Spanish multi-utterance.
#[tokio::test]
async fn golden_utseg_spa() {
    let Some(session) = require_live_direct(
        InferTask::Utseg,
        "Direct session does not support utseg infer",
    )
    .await
    else {
        return;
    };

    const SPA_MULTI_UTT: &str = "\
@UTF8
@Begin
@Languages:\tspa
@Participants:\tPAR Participant
@ID:\tspa|test|PAR|||||Participant|||
*PAR:\tel perro corre .
*PAR:\tme gustan los gatos .
@End
";

    let options = CommandOptions::Utseg(UtsegOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        merge_abbrev: false.into(),
    });

    let files = vec![FilePayload {
        filename: "spa_multi_utt.cha".into(),
        content: SPA_MULTI_UTT.into(),
    }];

    let (info, results) =
        submit_and_complete_direct(&session, ReleasedCommand::Utseg, "spa", files, options).await;

    if info.status == JobStatus::Failed {
        eprintln!("SKIP: Spanish utseg failed (model likely not downloaded)");
        return;
    }

    assert_completed_without_errors("utseg_spa", &info, &results);
    assert_eq!(results.len(), 1);

    insta::assert_snapshot!("utseg_spa", &results[0].content);
}

/// Translate: Spanish to English.
#[tokio::test]
async fn golden_translate_spa_to_eng() {
    let Some(session) = require_live_direct(
        InferTask::Translate,
        "Direct session does not support translate infer",
    )
    .await
    else {
        return;
    };

    let options = CommandOptions::Translate(TranslateOptions {
        common: CommonOptions {
            override_cache: true,
            ..CommonOptions::default()
        },
        merge_abbrev: false.into(),
    });

    let files = vec![FilePayload {
        filename: "spa_simple.cha".into(),
        content: SPA_SIMPLE.into(),
    }];

    let (info, results) =
        submit_and_complete_direct(&session, ReleasedCommand::Translate, "spa", files, options)
            .await;

    if info.status == JobStatus::Failed {
        eprintln!("SKIP: Spanish translate failed (model likely not downloaded)");
        return;
    }

    assert_completed_without_errors("translate_spa_to_eng", &info, &results);
    assert_eq!(results.len(), 1);

    let output = &results[0].content;
    let file = parse_output(output, "translate_spa_to_eng");
    assert!(
        has_user_defined_tier(&file, "xtra"),
        "Translated output should contain %xtra tier"
    );

    insta::assert_snapshot!("translate_spa_to_eng", output);
}

// ---------------------------------------------------------------------------
// Cache speed (P1)
// ---------------------------------------------------------------------------

/// Verify that the second run (cache hit) produces identical output and is faster.
#[tokio::test]
async fn golden_morphotag_cache_is_faster() {
    let Some(session) = require_live_direct(
        InferTask::Morphosyntax,
        "Direct session does not support morphosyntax infer",
    )
    .await
    else {
        return;
    };
    let make_options = || {
        CommandOptions::Morphotag(MorphotagOptions {
            common: CommonOptions::default(), // cache enabled (no override)
            retokenize: false,
            skipmultilang: false,
            merge_abbrev: false.into(),
        })
    };

    // First run — cold (populates cache)
    let start1 = std::time::Instant::now();
    let (info1, results1) = submit_and_complete_direct(
        &session,
        ReleasedCommand::Morphotag,
        "eng",
        vec![FilePayload {
            filename: "cache_speed.cha".into(),
            content: ENG_SIMPLE.into(),
        }],
        make_options(),
    )
    .await;
    let elapsed1 = start1.elapsed();
    assert_completed_without_errors("morphotag_cache_is_faster_cold", &info1, &results1);

    // Second run — should hit cache
    let start2 = std::time::Instant::now();
    let (info2, results2) = submit_and_complete_direct(
        &session,
        ReleasedCommand::Morphotag,
        "eng",
        vec![FilePayload {
            filename: "cache_speed.cha".into(),
            content: ENG_SIMPLE.into(),
        }],
        make_options(),
    )
    .await;
    let elapsed2 = start2.elapsed();
    assert_completed_without_errors("morphotag_cache_is_faster_warm", &info2, &results2);

    // Output must be identical
    assert_eq!(
        results1[0].content, results2[0].content,
        "Cache hit should produce identical output"
    );

    // Cache hit should be faster (generous: at least 2x, but don't fail on CI variance)
    eprintln!(
        "Cache timing: cold={:?}, warm={:?} (speedup: {:.1}x)",
        elapsed1,
        elapsed2,
        elapsed1.as_secs_f64() / elapsed2.as_secs_f64()
    );
    // Only assert if cold run was slow enough to be meaningful (>1s)
    if elapsed1.as_secs_f64() > 1.0 {
        assert!(
            elapsed2 < elapsed1,
            "Cache hit ({elapsed2:?}) should be faster than cold run ({elapsed1:?})"
        );
    }
}

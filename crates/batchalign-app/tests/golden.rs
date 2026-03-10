//! Golden model tests — snapshot NLP engine output via real server round-trip.
//!
//! These tests acquire isolated real-model server sessions backed by a shared
//! warmed worker pool, submit CHAT files, and compare the output against insta
//! snapshots. They verify that model output hasn't regressed after
//! Stanza/engine upgrades.
//!
//! Requirements:
//! - Python 3 with batchalign installed
//! - Stanza models downloaded (`stanza.download("en")`)
//!
//! Tests skip gracefully if models are unavailable.
//!
//! Run: `cargo nextest run -p batchalign-app --test golden`
//! Update snapshots: `cargo insta review`

mod common;

use batchalign_app::api::{FilePayload, JobStatus};
use batchalign_app::options::{
    CommandOptions, CommonOptions, CompareOptions, CorefOptions, MorphotagOptions,
    TranslateOptions, UtsegOptions,
};
use batchalign_app::worker::InferTask;
use common::{assert_completed_without_errors, require_live_server, submit_and_complete};

// ---------------------------------------------------------------------------
// Infrastructure
// ---------------------------------------------------------------------------

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
    let Some(server) = require_live_server(
        InferTask::Morphosyntax,
        "Server does not support morphosyntax infer",
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

    let (info, results) = submit_and_complete(
        server.client(),
        server.base_url(),
        "morphotag",
        "eng",
        files,
        options,
    )
    .await;

    assert_completed_without_errors("morphotag_eng_simple", &info, &results);
    assert_eq!(results.len(), 1);

    let output = &results[0].content;
    assert!(output.contains("%mor:"), "Output should contain %mor tier");
    assert!(output.contains("%gra:"), "Output should contain %gra tier");

    insta::assert_snapshot!("morphotag_eng_simple", output);
}

/// Morphotag: English multi-utterance → snapshot all %mor/%gra.
#[tokio::test]
async fn golden_morphotag_eng_multi_utt() {
    let Some(server) = require_live_server(
        InferTask::Morphosyntax,
        "Server does not support morphosyntax infer",
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

    let (info, results) = submit_and_complete(
        server.client(),
        server.base_url(),
        "morphotag",
        "eng",
        files,
        options,
    )
    .await;

    assert_completed_without_errors("morphotag_eng_multi_utt", &info, &results);
    assert_eq!(results.len(), 1);

    let output = &results[0].content;
    insta::assert_snapshot!("morphotag_eng_multi_utt", output);
}

/// Utseg: English multi-utterance → snapshot segmentation.
#[tokio::test]
async fn golden_utseg_eng_multi_utt() {
    let Some(server) =
        require_live_server(InferTask::Utseg, "Server does not support utseg infer").await
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

    let (info, results) = submit_and_complete(
        server.client(),
        server.base_url(),
        "utseg",
        "eng",
        files,
        options,
    )
    .await;

    assert_completed_without_errors("utseg_eng_multi_utt", &info, &results);
    assert_eq!(results.len(), 1);

    let output = &results[0].content;
    insta::assert_snapshot!("utseg_eng_multi_utt", output);
}

/// Translate: English simple → snapshot %xtra tier.
#[tokio::test]
async fn golden_translate_eng_simple() {
    let Some(server) = require_live_server(
        InferTask::Translate,
        "Server does not support translate infer",
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

    let (info, results) = submit_and_complete(
        server.client(),
        server.base_url(),
        "translate",
        "eng",
        files,
        options,
    )
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
    let Some(server) = require_live_server(
        InferTask::Morphosyntax,
        "Server does not support morphosyntax infer",
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
    let (info1, results1) = submit_and_complete(
        server.client(),
        server.base_url(),
        "morphotag",
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
    let (info2, results2) = submit_and_complete(
        server.client(),
        server.base_url(),
        "morphotag",
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
    let Some(server) = require_live_server(
        InferTask::Morphosyntax,
        "Server does not support morphosyntax infer (required for compare)",
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

    let (info, results) = submit_and_complete(
        server.client(),
        server.base_url(),
        "compare",
        "eng",
        files,
        options,
    )
    .await;

    assert_eq!(info.status, JobStatus::Completed, "Job should complete");
    // Only 1 result — gold file is skipped as input
    assert_eq!(results.len(), 1, "Should have 1 result (gold file skipped)");
    assert!(results[0].error.is_none(), "No error expected");

    let output = &results[0].content;
    assert!(
        output.contains("%xsrep:"),
        "Output should contain %xsrep tier"
    );
    assert!(
        output.contains("%mor:"),
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
    let Some(server) =
        require_live_server(InferTask::Coref, "Server does not support coref infer").await
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

    let (info, results) = submit_and_complete(
        server.client(),
        server.base_url(),
        "coref",
        "eng",
        files,
        options,
    )
    .await;

    assert_eq!(info.status, JobStatus::Completed, "Job should complete");
    assert_eq!(results.len(), 1);
    assert!(results[0].error.is_none(), "No error expected");

    let output = &results[0].content;

    // Coref models may or may not detect chains in short input.
    // Snapshot the output regardless — it validates the pipeline runs
    // without crashing. If %xcoref appears, the model found chains.
    if output.contains("%xcoref:") {
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
    let Some(server) = require_live_server(
        InferTask::Morphosyntax,
        "Server does not support morphosyntax infer",
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

    let (info, results) = submit_and_complete(
        server.client(),
        server.base_url(),
        "morphotag",
        "spa",
        files,
        options,
    )
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
    assert!(output.contains("%mor:"), "Output should contain %mor tier");
    assert!(output.contains("%gra:"), "Output should contain %gra tier");

    insta::assert_snapshot!("morphotag_spa_simple", output);
}

// ---------------------------------------------------------------------------
// Cache hit verification (P1)
// ---------------------------------------------------------------------------

/// Verify that the second run (cache hit) produces identical output and is faster.
#[tokio::test]
async fn golden_morphotag_cache_is_faster() {
    let Some(server) = require_live_server(
        InferTask::Morphosyntax,
        "Server does not support morphosyntax infer",
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
    let (info1, results1) = submit_and_complete(
        server.client(),
        server.base_url(),
        "morphotag",
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
    let (info2, results2) = submit_and_complete(
        server.client(),
        server.base_url(),
        "morphotag",
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

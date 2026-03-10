//! Validation of generated %gra structures.

use super::errors::MappingError;
use std::collections::HashSet;
use talkbank_model::model::GrammaticalRelation;

/// Validates generated %gra structure for correctness.
///
/// Since we are GENERATING %gra (not parsing pre-existing files), we can enforce
/// strict validation rules. This prevents us from ever creating broken %gra tiers.
///
/// Enforces:
/// 1. **Sequential indices**: 1, 2, 3, ..., N (already guaranteed by construction)
/// 2. **Single root**: Exactly one word has head=0 or head=self
/// 3. **No cycles**: No word can be its own ancestor
/// 4. **Valid heads**: All heads reference existing words (or 0/self)
///
/// Validate that generated %gra relations form a valid tree.
///
/// Returns `Err` on validation failure. Circular dependencies can come from
/// Stanza's dependency parser (especially on aphasic/disordered speech), so
/// this must NOT panic — the caller logs the error and skips the utterance.
pub(super) fn validate_generated_gra(gras: &[GrammaticalRelation]) -> Result<(), MappingError> {
    if gras.is_empty() {
        return Ok(());
    }

    // Rule 1: Find all roots (head=0 or head=self)
    let mut roots = Vec::new();
    for rel in gras {
        if rel.head == 0 || rel.head == rel.index {
            roots.push(rel.index);
        }
    }

    // Enforce single root (excluding terminator PUNCT which points to root)
    // The last item is always the terminator, so check roots in gras[..gras.len()-1]
    let non_terminator_roots: Vec<_> = roots
        .iter()
        .filter(|&&idx| idx != gras.len())
        .copied()
        .collect();

    if non_terminator_roots.is_empty() {
        return Err(MappingError::InvalidRoot {
            details: format!("no ROOT relation. GRA: {:?}", gras),
        });
    }

    if non_terminator_roots.len() > 1 {
        return Err(MappingError::InvalidRoot {
            details: format!(
                "multiple ROOT relations: {:?}. GRA: {:?}",
                non_terminator_roots, gras
            ),
        });
    }

    // Rule 2: No cycles — single-pass memoized DFS (O(N), each node visited at most once)
    if let Some(word) = has_any_cycle_generated(gras) {
        return Err(MappingError::CircularDependency {
            details: format!("involving word {}. GRA: {:?}", word, gras),
        });
    }

    // Rule 3: Valid heads - all heads reference existing words or are 0
    let max_index = gras.len();
    for rel in gras {
        if rel.head != 0 && rel.head > max_index {
            return Err(MappingError::InvalidHeadReference {
                details: format!(
                    "word {} points to non-existent word {}. GRA: {:?}",
                    rel.index, rel.head, gras
                ),
            });
        }
    }

    Ok(())
}

/// Single-pass memoized cycle detector for generated %gra relations.
///
/// Returns `Some(word_index)` if a cycle is found, `None` if the graph is acyclic.
/// O(N) with memoization — each node is visited at most once across all starting points.
/// Modeled after the parser's `has_any_cycle` in `tier.rs`.
fn has_any_cycle_generated(gras: &[GrammaticalRelation]) -> Option<usize> {
    let mut safe: HashSet<usize> = HashSet::new();
    for rel in gras {
        if safe.contains(&rel.index) {
            continue;
        }
        let mut path: HashSet<usize> = HashSet::new();
        let mut current = rel.index;
        loop {
            if safe.contains(&current) {
                safe.extend(&path);
                break;
            }
            if path.contains(&current) {
                return Some(current); // Cycle!
            }
            path.insert(current);
            if let Some(r) = gras.iter().find(|r| r.index == current) {
                if r.head == 0 || r.head == current {
                    safe.extend(&path);
                    break;
                }
                current = r.head;
            } else {
                safe.extend(&path);
                break;
            }
        }
    }
    None
}

//! File discovery — mirrors `file_io.py` (`_discover_files`, `_discover_inputs`).
//!
//! Walks directories, filters by extension, sorts by size (largest first),
//! detects and skips dummy CHAT files.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use walkdir::WalkDir;

/// Check whether a CHAT file is a "dummy" placeholder that should be copied,
/// not processed.
///
/// Reads the first 512 bytes and checks for the `@Options:\tdummy` header or
/// the standard TalkBank dummy-file text.
pub fn is_dummy_chat(path: &Path) -> bool {
    let Ok(mut f) = fs::File::open(path) else {
        return false;
    };
    let mut buf = [0u8; 512];
    let n = match f.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return false,
    };
    let text = String::from_utf8_lossy(&buf[..n]);
    text.contains("@Options:\tdummy")
        || text.contains("This is a dummy file to permit playback from the TalkBank browser")
}

/// Discover files from a single directory for server dispatch.
///
/// Walks `in_dir` recursively, filters by `extensions`, sorts by file size
/// (largest first). Dummy CHAT files are skipped (should be copied separately).
///
/// Returns `(files, outputs)` where `outputs[i]` is the output path for `files[i]`.
pub fn discover_client_files(
    in_dir: &Path,
    out_dir: &Path,
    extensions: &[&str],
) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut files = Vec::new();
    let mut outputs = Vec::new();

    for entry in WalkDir::new(in_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Skip dummy CHAT files
        if ext == "cha" && is_dummy_chat(path) {
            // Copy to output (unless in-place)
            if in_dir != out_dir
                && let Ok(rel) = path.strip_prefix(in_dir)
            {
                let dest = out_dir.join(rel);
                if let Some(parent) = dest.parent() {
                    let _ = fs::create_dir_all(parent);
                }
                let _ = fs::copy(path, dest);
            }
            continue;
        }

        if extensions.contains(&ext.as_str()) || extensions.contains(&"*") {
            let rel = path.strip_prefix(in_dir).unwrap_or(path);
            let out_path = out_dir.join(rel);
            files.push(path.to_path_buf());
            outputs.push(out_path);
        }
    }

    // Sort by file size (largest first) to avoid stragglers
    sort_by_size_desc(&mut files, &mut outputs);

    (files, outputs)
}

/// Discover files from mixed inputs (directories + individual files) for server dispatch.
///
/// For directories: walks recursively via [`discover_client_files`].
/// For individual files: adds directly (no extension filtering — user chose them).
pub fn discover_server_inputs(
    inputs: &[String],
    out_dir: Option<&str>,
    extensions: &[&str],
) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut all_files = Vec::new();
    let mut all_outputs = Vec::new();

    for inp in inputs {
        let inp_path = Path::new(inp);
        if inp_path.is_dir() {
            let d_out = out_dir
                .map(PathBuf::from)
                .unwrap_or_else(|| inp_path.to_path_buf());
            let (fs, os) = discover_client_files(inp_path, &d_out, extensions);
            all_files.extend(fs);
            all_outputs.extend(os);
        } else if inp_path.is_file() {
            let out_path = if let Some(od) = out_dir {
                let name = inp_path.file_name().unwrap_or_default();
                PathBuf::from(od).join(name)
            } else {
                inp_path.to_path_buf() // in-place
            };
            all_files.push(inp_path.to_path_buf());
            all_outputs.push(out_path);
        }
    }

    // Sort by file size (largest first)
    sort_by_size_desc(&mut all_files, &mut all_outputs);

    (all_files, all_outputs)
}

/// Sort two parallel vectors by file size (largest first).
fn sort_by_size_desc(files: &mut Vec<PathBuf>, outputs: &mut Vec<PathBuf>) {
    if files.is_empty() {
        return;
    }
    let mut pairs: Vec<_> = files.drain(..).zip(outputs.drain(..)).collect();
    pairs.sort_by(|a, b| {
        let sa = fs::metadata(&a.0).map(|m| m.len()).unwrap_or(0);
        let sb = fs::metadata(&b.0).map(|m| m.len()).unwrap_or(0);
        sb.cmp(&sa)
    });
    for (f, o) in pairs {
        files.push(f);
        outputs.push(o);
    }
}

/// Infer a base directory from the inputs list for media mapping detection.
///
/// For directory inputs: returns the first directory.
/// For individual files: returns the common ancestor directory.
pub fn infer_base_dir(inputs: &[String]) -> PathBuf {
    let dirs: Vec<&str> = inputs
        .iter()
        .map(|s| s.as_str())
        .filter(|p| Path::new(p).is_dir())
        .collect();

    if let Some(&d) = dirs.first() {
        return PathBuf::from(d);
    }

    // All inputs are files — common ancestor
    if !inputs.is_empty() {
        let abs: Vec<PathBuf> = inputs
            .iter()
            .filter_map(|p| fs::canonicalize(p).ok())
            .collect();
        if abs.len() > 1 {
            // Find common prefix
            if let Some(first) = abs.first() {
                let mut common = first.clone();
                for path in &abs[1..] {
                    while !path.starts_with(&common) {
                        if !common.pop() {
                            break;
                        }
                    }
                }
                if common.is_file()
                    && let Some(parent) = common.parent()
                {
                    return parent.to_path_buf();
                }
                return common;
            }
        } else if let Some(first) = abs.first()
            && let Some(parent) = first.parent()
        {
            return parent.to_path_buf();
        }
    }

    PathBuf::from(".")
}

/// Build unique relative names for server payload and a result mapping.
///
/// Returns `(server_names, result_map)` where `result_map[server_name] = output_path`.
pub fn build_server_names(
    files: &[PathBuf],
    outputs: &[PathBuf],
    inputs: &[String],
) -> (Vec<String>, std::collections::HashMap<String, PathBuf>) {
    use std::collections::HashMap;

    let dir_inputs: Vec<PathBuf> = inputs
        .iter()
        .filter(|p| Path::new(p).is_dir())
        .filter_map(|p| fs::canonicalize(p).ok())
        .collect();

    // Find individual files (not under any dir input)
    let individual_abs: Vec<PathBuf> = files
        .iter()
        .filter_map(|f| fs::canonicalize(f).ok())
        .filter(|abs| {
            !dir_inputs.iter().any(|d| {
                let d_str = format!("{}/", d.display());
                abs.to_string_lossy().starts_with(&d_str)
            })
        })
        .collect();

    // Common ancestor for individual files
    let common: PathBuf = if individual_abs.len() > 1 {
        let mut c = individual_abs[0].clone();
        for path in &individual_abs[1..] {
            while !path.starts_with(&c) {
                if !c.pop() {
                    break;
                }
            }
        }
        if c.is_file() {
            c.parent().map(|p| p.to_path_buf()).unwrap_or(c)
        } else {
            c
        }
    } else if let Some(first) = individual_abs.first() {
        first.parent().map(|p| p.to_path_buf()).unwrap_or_default()
    } else {
        PathBuf::from("/")
    };

    let mut server_names = Vec::with_capacity(files.len());
    let mut result_map = HashMap::with_capacity(files.len());

    for (fpath, opath) in files.iter().zip(outputs.iter()) {
        let abs = fs::canonicalize(fpath).unwrap_or_else(|_| fpath.clone());

        // Check if this file is under a directory input
        let mut rel: Option<String> = None;
        for d in &dir_inputs {
            let d_str = format!("{}/", d.display());
            if let Some(suffix) = abs.to_string_lossy().strip_prefix(&d_str) {
                rel = Some(suffix.to_string());
                break;
            }
        }

        let rel = rel.unwrap_or_else(|| {
            abs.strip_prefix(&common)
                .map(|r| r.to_string_lossy().to_string())
                .unwrap_or_else(|_| {
                    abs.file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string()
                })
        });

        server_names.push(rel.clone());
        result_map.insert(rel, opath.clone());
    }

    (server_names, result_map)
}

/// Commands that create new files from media input.
///
/// These should never copy non-matching files to output.
pub const GENERATION_COMMANDS: &[&str] = &["transcribe", "transcribe_s", "benchmark", "opensmile"];

/// Copy files whose extension doesn't match `extensions` from `in_dir` to `out_dir`.
///
/// Preserves relative directory structure. Skipped for in-place mode and
/// generation commands.
pub fn copy_nonmatching(in_dir: &Path, out_dir: &Path, extensions: &[&str], command: &str) {
    if GENERATION_COMMANDS.contains(&command) {
        return;
    }
    if let (Ok(a), Ok(b)) = (fs::canonicalize(in_dir), fs::canonicalize(out_dir))
        && a == b
    {
        return;
    }

    for entry in WalkDir::new(in_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if !extensions.contains(&ext.as_str())
            && !extensions.contains(&"*")
            && let Ok(rel) = path.strip_prefix(in_dir)
        {
            let dest = out_dir.join(rel);
            if let Some(parent) = dest.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let _ = fs::copy(path, &dest);
        }
    }
}

#[cfg(test)]
mod tests;

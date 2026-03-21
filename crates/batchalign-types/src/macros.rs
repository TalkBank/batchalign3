//! Helper macros for declaring newtype wrappers.
//!
//! This is a local copy kept for `scheduling.rs` and any future batchalign-app
//! types. The canonical copy lives in `batchalign-types`.
//!
//! All generated types use `#[serde(transparent)]` so the wire format is
//! unchanged — JSON values remain bare strings or numbers.

/// Declare a `String`-wrapping newtype with serde-transparent serialization.
///
/// Derives: `Debug`, `Clone`, `PartialEq`, `Eq`, `Hash`, `Serialize`,
/// `Deserialize`, `ToSchema`, `JsonSchema`, plus `Display`, `From<String>`,
/// `From<&str>`, `Into<String>`, `Deref<Target=str>`, `AsRef<str>`,
/// `PartialEq<&str>`.
macro_rules! string_id {
    ($(#[$meta:meta])* $vis:vis $name:ident) => {
        $(#[$meta])*
        #[derive(
            Debug, Clone, PartialEq, Eq, Hash,
            serde::Serialize, serde::Deserialize,
            utoipa::ToSchema,
            schemars::JsonSchema,
        )]
        #[serde(transparent)]
        $vis struct $name(pub String);

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self { Self(s) }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self { Self(s.to_owned()) }
        }

        impl From<$name> for String {
            fn from(v: $name) -> String { v.0 }
        }

        impl std::ops::Deref for $name {
            type Target = str;
            fn deref(&self) -> &str { &self.0 }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str { &self.0 }
        }

        impl PartialEq<&str> for $name {
            fn eq(&self, other: &&str) -> bool { self.0 == *other }
        }

        impl std::borrow::Borrow<str> for $name {
            fn borrow(&self) -> &str { &self.0 }
        }

        impl Default for $name {
            fn default() -> Self { Self(String::new()) }
        }
    };
}

/// Declare a numeric newtype with serde-transparent serialization.
///
/// Derives: `Debug`, `Clone`, `Copy`, `PartialEq`, `Serialize`,
/// `Deserialize`, `ToSchema`, plus `Display`, `From<inner>`,
/// `Into<inner>`, `Deref<Target=inner>`, `PartialEq<inner>`.
///
/// Append `[Eq]` for integer types that also need `Eq` and `Hash`:
/// ```ignore
/// numeric_id!(pub DurationMs(u64) [Eq]);
/// ```
macro_rules! numeric_id {
    ($(#[$meta:meta])* $vis:vis $name:ident($inner:ty) [Eq]) => {
        numeric_id!(@base $(#[$meta])* $vis $name($inner));

        impl Eq for $name {}

        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.0.hash(state);
            }
        }
    };
    ($(#[$meta:meta])* $vis:vis $name:ident($inner:ty)) => {
        numeric_id!(@base $(#[$meta])* $vis $name($inner));
    };
    (@base $(#[$meta:meta])* $vis:vis $name:ident($inner:ty)) => {
        $(#[$meta])*
        #[derive(
            Debug, Clone, Copy, PartialEq, PartialOrd,
            serde::Serialize, serde::Deserialize,
            utoipa::ToSchema,
            schemars::JsonSchema,
        )]
        #[serde(transparent)]
        $vis struct $name(pub $inner);

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl From<$inner> for $name {
            fn from(v: $inner) -> Self { Self(v) }
        }

        impl From<$name> for $inner {
            fn from(v: $name) -> $inner { v.0 }
        }

        impl std::ops::Deref for $name {
            type Target = $inner;
            fn deref(&self) -> &$inner { &self.0 }
        }

        impl PartialEq<$inner> for $name {
            fn eq(&self, other: &$inner) -> bool { self.0 == *other }
        }

        impl Default for $name {
            fn default() -> Self { Self(Default::default()) }
        }
    };
}

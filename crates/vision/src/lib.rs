//! Picture analysis for FilmCraft.
//!
//! - [`tflite`]: a reader for TensorFlow Lite model files (FlatBuffers, from the public schema).
//! - [`nn`]: a pure-Rust float interpreter for the TFLite operator subset mobile CNNs use.
//! - [`face`]: face boxes and eye centres (MediaPipe BlazeFace + Face Mesh), for eye-line matching
//!   across jump cuts (`video.faces`, `video.alignEyes` in the engine).
//!
//! The model weights (Apache-2.0, Google MediaPipe) are downloaded on demand by the engine into the
//! per-user data directory, pinned by SHA-256; they are never bundled or committed.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod face;
pub mod nn;
pub mod tflite;

pub use face::{BBox, DetectOptions, DetectorKind, Face, FaceModels, Image, Method, Point};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// The model file is malformed or uses something the interpreter doesn't run.
    #[error("model: {0}")]
    Model(String),
    /// A bad picture or parameter.
    #[error("{0}")]
    Input(String),
}

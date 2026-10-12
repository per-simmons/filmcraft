# filmcraft-vision

Picture analysis for FilmCraft, in pure Rust with no dependencies beyond `serde` / `thiserror`
(it builds for wasm too).

- `tflite`: reads TensorFlow Lite model files: the FlatBuffers binary layout and the tables of the
  public TFLite schema (`tensorflow/lite/schema/schema.fbs`, Apache-2.0, schema version 3). Every
  offset is bounds-checked; a malformed file is an error.
- `nn`: a float interpreter for the operator subset of mobile CNNs (CONV_2D, DEPTHWISE_CONV_2D,
  ADD / SUB / MUL with broadcasting, PRELU, RELU, RELU6, LOGISTIC, TANH, MAX / AVERAGE_POOL_2D,
  PAD, CONCATENATION, RESHAPE, RESIZE_BILINEAR, DEQUANTIZE of f16 / int8 constants). NHWC, batch 1.
- `face`: face boxes and eye centres. MediaPipe BlazeFace (short range, 128 px) finds the faces
  (letterboxed whole picture, then overlapping tiles when nothing is found: webcam insets), Face
  Mesh (468 points) refines each one on a crop levelled on the eyes, twice. An eye centre is the
  midpoint of its two corners (mesh points 33/133 and 362/263), which blinks and gaze don't move.

The weights (Apache-2.0, Google MediaPipe; 1.5 MB) are not in the repository: the engine downloads
them on request into `<data dir>/models/mediapipe-face/`, pinned by object generation and SHA-256
(`crates/engine/src/face.rs`, `ATTRIBUTION.md`).

## Why this approach

- Licence: the MediaPipe code and models are Apache-2.0. Classical alternatives were weaker or not
  clean: dlib's 68-point model is trained on iBUG 300-W (non-commercial); Haar cascades find faces
  but not precise eye centres.
- Accuracy on webcam talking heads: Face Mesh eye corners on a 4K full-frame talking head jitter
  about ±0.7 px frame to frame (eye height) and follow a shifted picture within ±1 px; on webcam
  insets (faces ~130 px, eyes ~40 px apart at 4K) they follow within ±1.6 px.
- A 40-line candle graph would still need this TFLite reader; the interpreter is small enough to
  own outright, keeps the never-crash rules, and runs a 4K talking-head frame in ~50 ms (release).

## Tests

`cargo test -p filmcraft-vision` covers the reader (mutation fuzz), every operator, anchors, NMS,
crops and geometry. With `FILMCRAFT_FACE_MODELS=<dir with the two .tflite files>` and
`FILMCRAFT_FACE_IMAGE=<a PNG of a face>` it also checks the real models: a face is found, and
shifting or mirroring the picture moves the eye centres with it.

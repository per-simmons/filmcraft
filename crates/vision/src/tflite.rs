//! A TensorFlow Lite model reader: the FlatBuffers binary layout and the parts of the TFLite schema
//! (`tensorflow/lite/schema/schema.fbs`, Apache-2.0, a public specification) that a float CNN uses.
//!
//! Everything read from the file is hostile: every offset is bounds-checked, vectors are capped,
//! and a malformed file is an [`Error::Model`], never a panic.

use crate::Error;

/// Upper bound for any vector length read from a model (tensors, operators, shape entries…).
const MAX_VEC: usize = 1 << 24;

fn bad(msg: impl Into<String>) -> Error {
    Error::Model(msg.into())
}

/// A FlatBuffers view over a byte slice.
#[derive(Clone, Copy)]
struct Buf<'a>(&'a [u8]);

impl<'a> Buf<'a> {
    fn u8_at(&self, pos: usize) -> Result<u8, Error> {
        self.0.get(pos).copied().ok_or_else(|| bad("offset out of range"))
    }
    fn u16_at(&self, pos: usize) -> Result<u16, Error> {
        let b = self.0.get(pos..pos.checked_add(2).ok_or_else(|| bad("offset overflow"))?).ok_or_else(|| bad("offset out of range"))?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32_at(&self, pos: usize) -> Result<u32, Error> {
        let b = self.0.get(pos..pos.checked_add(4).ok_or_else(|| bad("offset overflow"))?).ok_or_else(|| bad("offset out of range"))?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn i32_at(&self, pos: usize) -> Result<i32, Error> {
        Ok(self.u32_at(pos)? as i32)
    }
    fn u64_at(&self, pos: usize) -> Result<u64, Error> {
        let lo = u64::from(self.u32_at(pos)?);
        let hi = u64::from(self.u32_at(pos.checked_add(4).ok_or_else(|| bad("offset overflow"))?)?);
        Ok(lo | (hi << 32))
    }
    /// Follow a `uoffset_t` stored at `pos`.
    fn deref(&self, pos: usize) -> Result<usize, Error> {
        let off = self.u32_at(pos)? as usize;
        let to = pos.checked_add(off).ok_or_else(|| bad("offset overflow"))?;
        if to >= self.0.len() {
            return Err(bad("offset out of range"));
        }
        Ok(to)
    }
    fn root(&self) -> Result<Table<'a>, Error> {
        let pos = self.deref(0)?;
        Table::at(*self, pos)
    }
}

/// A FlatBuffers table.
#[derive(Clone, Copy)]
struct Table<'a> {
    buf: Buf<'a>,
    pos: usize,
    vt: usize,
    vt_len: usize,
}

impl<'a> Table<'a> {
    fn at(buf: Buf<'a>, pos: usize) -> Result<Self, Error> {
        let soff = i64::from(buf.i32_at(pos)?);
        let vt = i64::try_from(pos).map_err(|_| bad("offset overflow"))?.checked_sub(soff).ok_or_else(|| bad("vtable offset"))?;
        let vt = usize::try_from(vt).map_err(|_| bad("vtable out of range"))?;
        let vt_len = usize::from(buf.u16_at(vt)?);
        if vt_len < 4 || vt.checked_add(vt_len).is_none_or(|e| e > buf.0.len()) {
            return Err(bad("malformed vtable"));
        }
        Ok(Table { buf, pos, vt, vt_len })
    }
    /// Absolute position of field `i`, if present.
    fn field(&self, i: usize) -> Result<Option<usize>, Error> {
        let slot = 4 + 2 * i;
        if slot + 2 > self.vt_len {
            return Ok(None);
        }
        let off = usize::from(self.buf.u16_at(self.vt + slot)?);
        if off == 0 {
            return Ok(None);
        }
        Ok(Some(self.pos.checked_add(off).ok_or_else(|| bad("field offset"))?))
    }
    fn u8_or(&self, i: usize, d: u8) -> Result<u8, Error> {
        match self.field(i)? {
            Some(p) => self.buf.u8_at(p),
            None => Ok(d),
        }
    }
    fn i32_or(&self, i: usize, d: i32) -> Result<i32, Error> {
        match self.field(i)? {
            Some(p) => self.buf.i32_at(p),
            None => Ok(d),
        }
    }
    fn u32_or(&self, i: usize, d: u32) -> Result<u32, Error> {
        match self.field(i)? {
            Some(p) => self.buf.u32_at(p),
            None => Ok(d),
        }
    }
    fn u64_or(&self, i: usize, d: u64) -> Result<u64, Error> {
        match self.field(i)? {
            Some(p) => self.buf.u64_at(p),
            None => Ok(d),
        }
    }
    fn table(&self, i: usize) -> Result<Option<Table<'a>>, Error> {
        match self.field(i)? {
            Some(p) => Ok(Some(Table::at(self.buf, self.buf.deref(p)?)?)),
            None => Ok(None),
        }
    }
    /// (position of the first element, length) of a vector field.
    fn vector(&self, i: usize, elem: usize) -> Result<Option<(usize, usize)>, Error> {
        let Some(p) = self.field(i)? else { return Ok(None) };
        let v = self.buf.deref(p)?;
        let n = self.buf.u32_at(v)? as usize;
        if n > MAX_VEC {
            return Err(bad("vector too long"));
        }
        let start = v + 4;
        let end = n.checked_mul(elem).and_then(|b| b.checked_add(start)).ok_or_else(|| bad("vector overflow"))?;
        if end > self.buf.0.len() {
            return Err(bad("vector out of range"));
        }
        Ok(Some((start, n)))
    }
    fn bytes(&self, i: usize) -> Result<&'a [u8], Error> {
        match self.vector(i, 1)? {
            Some((s, n)) => self.buf.0.get(s..s + n).ok_or_else(|| bad("vector out of range")),
            None => Ok(&[]),
        }
    }
    fn string(&self, i: usize) -> Result<String, Error> {
        Ok(String::from_utf8_lossy(self.bytes(i)?).into_owned())
    }
    fn i32s(&self, i: usize) -> Result<Vec<i32>, Error> {
        let Some((s, n)) = self.vector(i, 4)? else { return Ok(Vec::new()) };
        (0..n).map(|k| self.buf.i32_at(s + 4 * k)).collect()
    }
    fn f32s(&self, i: usize) -> Result<Vec<f32>, Error> {
        Ok(self.i32s(i)?.into_iter().map(|b| f32::from_bits(b as u32)).collect())
    }
    fn tables(&self, i: usize) -> Result<Vec<Table<'a>>, Error> {
        let Some((s, n)) = self.vector(i, 4)? else { return Ok(Vec::new()) };
        (0..n).map(|k| Table::at(self.buf, self.buf.deref(s + 4 * k)?)).collect()
    }
}

/// TFLite `TensorType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    I32,
    U8,
    I8,
    I64,
    Other(u8),
}

impl DType {
    fn from_code(c: u8) -> Self {
        match c {
            0 => DType::F32,
            1 => DType::F16,
            2 => DType::I32,
            3 => DType::U8,
            4 => DType::I64,
            9 => DType::I8,
            o => DType::Other(o),
        }
    }
}

/// A tensor of the graph.
#[derive(Clone, Debug)]
pub struct TensorDef {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// Index into [`Model::buffers`] (0 = no data: an activation).
    pub buffer: usize,
    /// Per-tensor affine quantisation (scale, zero point), when present.
    pub quant: Option<(f32, i64)>,
}

/// Padding of a convolution or pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Padding {
    Same,
    Valid,
}

/// Fused activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    None,
    Relu,
    Relu1,
    Relu6,
    Tanh,
}

impl Act {
    fn from_code(c: u8) -> Result<Self, Error> {
        Ok(match c {
            0 => Act::None,
            1 => Act::Relu,
            2 => Act::Relu1,
            3 => Act::Relu6,
            4 => Act::Tanh,
            o => return Err(bad(format!("unsupported fused activation {o}"))),
        })
    }
}

/// The operators the interpreter runs, with their options.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Add {
        act: Act,
    },
    Mul {
        act: Act,
    },
    Sub {
        act: Act,
    },
    Conv2d {
        pad: Padding,
        stride: (usize, usize),
        dilation: (usize, usize),
        act: Act,
    },
    DepthwiseConv2d {
        pad: Padding,
        stride: (usize, usize),
        dilation: (usize, usize),
        multiplier: usize,
        act: Act,
    },
    MaxPool2d {
        pad: Padding,
        stride: (usize, usize),
        filter: (usize, usize),
        act: Act,
    },
    AvgPool2d {
        pad: Padding,
        stride: (usize, usize),
        filter: (usize, usize),
        act: Act,
    },
    Concat {
        axis: i32,
        act: Act,
    },
    Reshape,
    Pad,
    Prelu,
    Relu,
    Relu6,
    Logistic,
    Tanh,
    Dequantize,
    ResizeBilinear {
        align_corners: bool,
        half_pixel: bool,
    },
    /// Any other builtin (by code) or custom op (by name): the model can't run.
    Unsupported(String),
}

/// One node of the graph.
#[derive(Clone, Debug)]
pub struct OpDef {
    pub op: Op,
    /// Tensor indices; -1 = an optional input left out.
    pub inputs: Vec<i32>,
    pub outputs: Vec<i32>,
}

/// A parsed model: its first subgraph and its constant buffers.
#[derive(Clone, Debug)]
pub struct Model {
    pub tensors: Vec<TensorDef>,
    pub ops: Vec<OpDef>,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    /// Constant data per buffer index (empty for activations).
    pub buffers: Vec<Vec<u8>>,
}

fn opt_stride(t: &Option<Table>, w_field: usize, h_field: usize) -> Result<(usize, usize), Error> {
    let Some(t) = t else { return Ok((1, 1)) };
    let w = t.i32_or(w_field, 1)?;
    let h = t.i32_or(h_field, 1)?;
    if !(1..=64).contains(&w) || !(1..=64).contains(&h) {
        return Err(bad("stride/dilation out of range"));
    }
    Ok((h as usize, w as usize))
}

fn opt_pad(t: &Option<Table>, f: usize) -> Result<Padding, Error> {
    Ok(match t.as_ref().map(|t| t.u8_or(f, 0)).transpose()?.unwrap_or(0) {
        0 => Padding::Same,
        _ => Padding::Valid,
    })
}

fn opt_act(t: &Option<Table>, f: usize) -> Result<Act, Error> {
    Act::from_code(t.as_ref().map(|t| t.u8_or(f, 0)).transpose()?.unwrap_or(0))
}

fn builtin_name(code: i32) -> String {
    format!("builtin operator {code}")
}

fn parse_op(code: i32, custom: &str, o: Option<Table>) -> Result<Op, Error> {
    Ok(match code {
        0 => Op::Add { act: opt_act(&o, 0)? },
        1 => Op::AvgPool2d {
            pad: opt_pad(&o, 0)?,
            stride: opt_stride(&o, 1, 2)?,
            filter: opt_stride(&o, 3, 4).map_err(|_| bad("pool filter out of range"))?,
            act: opt_act(&o, 5)?,
        },
        2 => Op::Concat { axis: o.as_ref().map(|t| t.i32_or(0, 0)).transpose()?.unwrap_or(0), act: opt_act(&o, 1)? },
        3 => Op::Conv2d { pad: opt_pad(&o, 0)?, stride: opt_stride(&o, 1, 2)?, act: opt_act(&o, 3)?, dilation: opt_stride(&o, 4, 5)? },
        4 => {
            let m = o.as_ref().map(|t| t.i32_or(3, 1)).transpose()?.unwrap_or(1);
            Op::DepthwiseConv2d {
                pad: opt_pad(&o, 0)?,
                stride: opt_stride(&o, 1, 2)?,
                multiplier: usize::try_from(m).ok().filter(|m| (1..=64).contains(m)).ok_or_else(|| bad("depth multiplier"))?,
                act: opt_act(&o, 4)?,
                dilation: opt_stride(&o, 5, 6)?,
            }
        }
        6 => Op::Dequantize,
        14 => Op::Logistic,
        17 => Op::MaxPool2d {
            pad: opt_pad(&o, 0)?,
            stride: opt_stride(&o, 1, 2)?,
            filter: opt_stride(&o, 3, 4).map_err(|_| bad("pool filter out of range"))?,
            act: opt_act(&o, 5)?,
        },
        18 => Op::Mul { act: opt_act(&o, 0)? },
        19 => Op::Relu,
        21 => Op::Relu6,
        22 => Op::Reshape,
        23 => Op::ResizeBilinear {
            align_corners: o.as_ref().map(|t| t.u8_or(2, 0)).transpose()?.unwrap_or(0) != 0,
            half_pixel: o.as_ref().map(|t| t.u8_or(3, 0)).transpose()?.unwrap_or(0) != 0,
        },
        28 => Op::Tanh,
        32 => Op::Unsupported(format!("custom operator `{custom}`")),
        34 => Op::Pad,
        41 => Op::Sub { act: opt_act(&o, 0)? },
        54 => Op::Prelu,
        c => Op::Unsupported(builtin_name(c)),
    })
}

impl Model {
    /// Parse a `.tflite` file.
    pub fn parse(bytes: &[u8]) -> Result<Model, Error> {
        if bytes.len() < 16 {
            return Err(bad("file too short"));
        }
        let buf = Buf(bytes);
        let root = buf.root()?;
        // operator codes: (builtin code, custom name)
        let mut codes = Vec::new();
        for oc in root.tables(1)? {
            let dep = i32::from(oc.u8_or(0, 0)?);
            let b = oc.i32_or(3, 0)?;
            // newer files store codes > 127 in `builtin_code`; older ones only in the deprecated byte
            codes.push((dep.max(b), oc.string(1)?));
        }
        let subgraphs = root.tables(2)?;
        let sg = subgraphs.first().ok_or_else(|| bad("no subgraph"))?;
        let mut buffers = Vec::new();
        for b in root.tables(4)? {
            let data = b.bytes(0)?;
            if !data.is_empty() {
                buffers.push(data.to_vec());
                continue;
            }
            // large models keep buffers after the flatbuffer (offset from the file start)
            let (off, size) = (b.u64_or(1, 0)?, b.u64_or(2, 0)?);
            if off > 1 && size > 0 {
                let s = usize::try_from(off).map_err(|_| bad("buffer offset"))?;
                let e = usize::try_from(size).ok().and_then(|n| s.checked_add(n)).ok_or_else(|| bad("buffer size"))?;
                buffers.push(bytes.get(s..e).ok_or_else(|| bad("buffer out of range"))?.to_vec());
            } else {
                buffers.push(Vec::new());
            }
        }
        let mut tensors = Vec::new();
        for t in sg.tables(0)? {
            let shape = t
                .i32s(0)?
                .into_iter()
                .map(|d| usize::try_from(d).ok().filter(|d| *d <= 1 << 20).ok_or_else(|| bad("tensor dimension out of range")))
                .collect::<Result<Vec<_>, _>>()?;
            let quant = match t.table(4)? {
                Some(q) => {
                    let scale = q.f32s(2)?;
                    let zp = match q.vector(3, 8)? {
                        Some((s, n)) if n > 0 => buf.u64_at(s)? as i64,
                        _ => 0,
                    };
                    scale.first().map(|s| (*s, zp))
                }
                None => None,
            };
            tensors.push(TensorDef { name: t.string(3)?, shape, dtype: DType::from_code(t.u8_or(1, 0)?), buffer: t.u32_or(2, 0)? as usize, quant });
        }
        let idx = |v: Vec<i32>| -> Result<Vec<usize>, Error> {
            v.into_iter().map(|i| usize::try_from(i).ok().filter(|i| *i < tensors.len()).ok_or_else(|| bad("tensor index out of range"))).collect()
        };
        let inputs = idx(sg.i32s(1)?)?;
        let outputs = idx(sg.i32s(2)?)?;
        let mut ops = Vec::new();
        for o in sg.tables(3)? {
            let oi = o.u32_or(0, 0)? as usize;
            let (code, custom) = codes.get(oi).cloned().ok_or_else(|| bad("operator code index out of range"))?;
            let opts = o.table(4)?;
            let ins = o.i32s(1)?;
            let outs = o.i32s(2)?;
            for i in ins.iter().chain(&outs) {
                if *i < -1 || *i >= i32::try_from(tensors.len()).unwrap_or(i32::MAX) {
                    return Err(bad("operator tensor index out of range"));
                }
            }
            ops.push(OpDef { op: parse_op(code, &custom, opts)?, inputs: ins, outputs: outs });
        }
        for t in &tensors {
            if t.buffer >= buffers.len() && t.buffer != 0 {
                return Err(bad(format!("tensor `{}` names a missing buffer", t.name)));
            }
        }
        Ok(Model { tensors, ops, inputs, outputs, buffers })
    }

    /// The operators this interpreter can't run (empty when the model is runnable).
    pub fn unsupported(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .ops
            .iter()
            .filter_map(|o| match &o.op {
                Op::Unsupported(n) => Some(n.clone()),
                _ => None,
            })
            .collect();
        v.dedup();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hostile_bytes_never_panic() {
        assert!(Model::parse(&[]).is_err());
        assert!(Model::parse(&[0xff; 64]).is_err());
        let mut seed = 0x1234_5678u32;
        for len in [16usize, 17, 64, 200, 1000] {
            for _ in 0..200 {
                let v: Vec<u8> = (0..len)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 17;
                        seed ^= seed << 5;
                        seed as u8
                    })
                    .collect();
                let _ = Model::parse(&v);
            }
        }
    }
}

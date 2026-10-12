//! A small float interpreter for the TFLite operator subset of mobile CNNs (BlazeFace, Face Mesh,
//! Iris): NHWC tensors, batch 1, f32 arithmetic (f16 / int8 / uint8 weights are dequantised when
//! the model is loaded).
//!
//! Every shape comes from the model file and is checked before it is used: a model that doesn't
//! fit these rules fails to load or to run with an [`Error::Model`], never a panic.

use crate::Error;
use crate::tflite::{Act, DType, Model, Op, OpDef, Padding};

/// Largest tensor (elements) the interpreter will allocate.
const MAX_ELEMS: usize = 1 << 26;

fn bad(msg: impl Into<String>) -> Error {
    Error::Model(msg.into())
}

/// A dense f32 tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    /// Every tensor keeps `data.len() == product(shape)`; the kernels index on that.
    fn consistent(&self) -> bool {
        numel(&self.shape).is_ok_and(|n| n == self.data.len())
    }
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, Error> {
        if numel(&shape)? != data.len() {
            return Err(bad(format!("tensor of shape {shape:?} given {} values", data.len())));
        }
        Ok(Tensor { shape, data })
    }
    fn zeros(shape: Vec<usize>) -> Result<Self, Error> {
        let n = numel(&shape)?;
        Ok(Tensor { shape, data: vec![0.0; n] })
    }
    /// (h, w, c) of an NHWC tensor with batch 1.
    fn hwc(&self) -> Result<(usize, usize, usize), Error> {
        match self.shape.as_slice() {
            [1, h, w, c] => Ok((*h, *w, *c)),
            s => Err(bad(format!("expected a [1,h,w,c] tensor, got {s:?}"))),
        }
    }
}

fn numel(shape: &[usize]) -> Result<usize, Error> {
    let n = shape.iter().try_fold(1usize, |a, d| a.checked_mul(*d)).ok_or_else(|| bad("tensor too large"))?;
    if n > MAX_ELEMS {
        return Err(bad("tensor too large"));
    }
    Ok(n)
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            // subnormal: normalise
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (man << 13),
        e => sign | ((e + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// Decode a constant buffer to f32 values.
fn const_f32(data: &[u8], dtype: DType, quant: Option<(f32, i64)>, n: usize) -> Result<Vec<f32>, Error> {
    let need = |w: usize| n.checked_mul(w).filter(|b| *b <= data.len()).ok_or_else(|| bad("constant buffer shorter than its tensor"));
    Ok(match dtype {
        DType::F32 => {
            let b = need(4)?;
            data[..b].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        }
        DType::F16 => {
            let b = need(2)?;
            data[..b].chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
        }
        DType::I32 => {
            let b = need(4)?;
            data[..b].chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32).collect()
        }
        DType::U8 | DType::I8 => {
            need(1)?;
            let (s, zp) = quant.unwrap_or((1.0, 0));
            data[..n].iter().map(|b| (if dtype == DType::I8 { i64::from(*b as i8) } else { i64::from(*b) } - zp) as f32 * s).collect()
        }
        DType::I64 => {
            let b = need(8)?;
            data[..b].chunks_exact(8).map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32).collect()
        }
        DType::Other(c) => return Err(bad(format!("unsupported tensor type {c}"))),
    })
}

/// A loaded, runnable network.
#[derive(Debug)]
pub struct Net {
    model: Model,
    /// Constant tensors (weights, biases, folded dequantisations).
    consts: Vec<Option<Tensor>>,
    /// Operators left to run (dequantisations of constants are folded away).
    ops: Vec<OpDef>,
    /// Per tensor: the index of the last op that reads it (activations are freed after it).
    last_use: Vec<usize>,
}

impl Net {
    /// Parse and prepare a `.tflite` model.
    pub fn load(bytes: &[u8]) -> Result<Net, Error> {
        let model = Model::parse(bytes)?;
        let bad_ops = model.unsupported();
        if !bad_ops.is_empty() {
            return Err(bad(format!("unsupported operators: {}", bad_ops.join(", "))));
        }
        let mut consts: Vec<Option<Tensor>> = Vec::with_capacity(model.tensors.len());
        for t in &model.tensors {
            let data = model.buffers.get(t.buffer).filter(|b| t.buffer != 0 && !b.is_empty());
            consts.push(match data {
                Some(d) => {
                    let n = numel(&t.shape)?;
                    Some(Tensor { shape: t.shape.clone(), data: const_f32(d, t.dtype, t.quant, n)? })
                }
                None => None,
            });
        }
        // fold Dequantize(const) into constants
        let mut ops = Vec::new();
        for o in &model.ops {
            if o.op == Op::Dequantize
                && let (Some(i), Some(out)) = (o.inputs.first(), o.outputs.first())
                && let (Ok(i), Ok(out)) = (usize::try_from(*i), usize::try_from(*out))
                && let Some(Some(c)) = consts.get(i).cloned()
                && let Some(slot) = consts.get_mut(out)
            {
                *slot = Some(c);
                continue;
            }
            ops.push(o.clone());
        }
        let mut last_use = vec![0usize; model.tensors.len()];
        for (k, o) in ops.iter().enumerate() {
            for i in &o.inputs {
                if let Ok(i) = usize::try_from(*i)
                    && let Some(l) = last_use.get_mut(i)
                {
                    *l = k;
                }
            }
        }
        for o in &model.outputs {
            if let Some(l) = last_use.get_mut(*o) {
                *l = usize::MAX;
            }
        }
        Ok(Net { model, consts, ops, last_use })
    }

    /// Shape of input `i`.
    pub fn input_shape(&self, i: usize) -> Option<&[usize]> {
        self.model.inputs.get(i).and_then(|t| self.model.tensors.get(*t)).map(|t| t.shape.as_slice())
    }

    /// Run the network on its (single) input; returns the outputs in model order.
    pub fn run(&self, input: Tensor) -> Result<Vec<Tensor>, Error> {
        let n = self.model.tensors.len();
        let mut vals: Vec<Option<Tensor>> = vec![None; n];
        let &[inp] = self.model.inputs.as_slice() else { return Err(bad("the model must have exactly one input")) };
        let want = self.input_shape(0).ok_or_else(|| bad("no input"))?;
        if want != input.shape.as_slice() {
            return Err(bad(format!("input shape {:?}, the model takes {want:?}", input.shape)));
        }
        if let Some(v) = vals.get_mut(inp) {
            *v = Some(input);
        }
        for (k, o) in self.ops.iter().enumerate() {
            let out = self.exec(o, &vals)?;
            let oi = o.outputs.first().and_then(|i| usize::try_from(*i).ok()).ok_or_else(|| bad("operator without output"))?;
            if let Some(v) = vals.get_mut(oi) {
                *v = Some(out);
            }
            // free activations nothing reads any more
            for i in &o.inputs {
                if let Ok(i) = usize::try_from(*i)
                    && self.last_use.get(i) == Some(&k)
                    && let Some(v) = vals.get_mut(i)
                {
                    *v = None;
                }
            }
        }
        self.model
            .outputs
            .iter()
            .map(|o| vals.get(*o).cloned().flatten().or_else(|| self.consts.get(*o).cloned().flatten()).ok_or_else(|| bad("an output was never computed")))
            .collect()
    }

    fn get<'v>(&'v self, vals: &'v [Option<Tensor>], o: &OpDef, k: usize) -> Result<&'v Tensor, Error> {
        let i = o.inputs.get(k).and_then(|i| usize::try_from(*i).ok()).ok_or_else(|| bad("missing operator input"))?;
        let t = vals
            .get(i)
            .and_then(Option::as_ref)
            .or_else(|| self.consts.get(i).and_then(Option::as_ref))
            .ok_or_else(|| bad("operator input not computed yet"))?;
        if !t.consistent() {
            return Err(bad("inconsistent tensor"));
        }
        Ok(t)
    }

    fn opt<'v>(&'v self, vals: &'v [Option<Tensor>], o: &OpDef, k: usize) -> Option<&'v Tensor> {
        let i = usize::try_from(*o.inputs.get(k)?).ok()?;
        vals.get(i).and_then(Option::as_ref).or_else(|| self.consts.get(i).and_then(Option::as_ref)).filter(|t| t.consistent())
    }

    fn out_shape(&self, o: &OpDef) -> Result<Vec<usize>, Error> {
        let i = o.outputs.first().and_then(|i| usize::try_from(*i).ok()).ok_or_else(|| bad("operator without output"))?;
        Ok(self.model.tensors.get(i).ok_or_else(|| bad("output tensor index"))?.shape.clone())
    }

    fn exec(&self, o: &OpDef, vals: &[Option<Tensor>]) -> Result<Tensor, Error> {
        let x = self.get(vals, o, 0)?;
        let r = match &o.op {
            Op::Add { act } => binary(x, self.get(vals, o, 1)?, |a, b| a + b, *act),
            Op::Sub { act } => binary(x, self.get(vals, o, 1)?, |a, b| a - b, *act),
            Op::Mul { act } => binary(x, self.get(vals, o, 1)?, |a, b| a * b, *act),
            Op::Prelu => binary(x, self.get(vals, o, 1)?, |a, s| if a >= 0.0 { a } else { a * s }, Act::None),
            Op::Relu => Ok(map(x, |v| v.max(0.0))),
            Op::Relu6 => Ok(map(x, |v| v.clamp(0.0, 6.0))),
            Op::Logistic => Ok(map(x, |v| 1.0 / (1.0 + (-v).exp()))),
            Op::Tanh => Ok(map(x, f32::tanh)),
            Op::Dequantize => Ok(x.clone()),
            Op::Reshape => {
                let shape = self.out_shape(o)?;
                if numel(&shape)? != x.data.len() {
                    return Err(bad(format!("reshape of {:?} to {shape:?}", x.shape)));
                }
                Ok(Tensor { shape, data: x.data.clone() })
            }
            Op::Pad => pad(x, self.get(vals, o, 1)?),
            Op::Concat { axis, act } => {
                let ins: Vec<&Tensor> = (0..o.inputs.len()).map(|k| self.get(vals, o, k)).collect::<Result<_, _>>()?;
                concat(&ins, *axis).map(|t| activate(t, *act))
            }
            Op::Conv2d { pad, stride, dilation, act } => conv(x, self.get(vals, o, 1)?, self.opt(vals, o, 2), *pad, *stride, *dilation, *act),
            Op::DepthwiseConv2d { pad, stride, dilation, multiplier, act } => {
                depthwise(x, self.get(vals, o, 1)?, self.opt(vals, o, 2), *pad, *stride, *dilation, *multiplier, *act)
            }
            Op::MaxPool2d { pad, stride, filter, act } => pool(x, *pad, *stride, *filter, true).map(|t| activate(t, *act)),
            Op::AvgPool2d { pad, stride, filter, act } => pool(x, *pad, *stride, *filter, false).map(|t| activate(t, *act)),
            Op::ResizeBilinear { align_corners, half_pixel } => resize(x, &self.out_shape(o)?, *align_corners, *half_pixel),
            Op::Unsupported(n) => Err(bad(format!("unsupported operator {n}"))),
        }?;
        // trust the computed shape, but it must agree with the model's declaration
        let declared = self.out_shape(o)?;
        if !declared.is_empty() && numel(&declared)? != r.data.len() {
            return Err(bad(format!("operator {:?} produced {:?}, the model declares {declared:?}", o.op, r.shape)));
        }
        Ok(r)
    }
}

fn act_fn(a: Act) -> fn(f32) -> f32 {
    match a {
        Act::None => |v| v,
        Act::Relu => |v| v.max(0.0),
        Act::Relu1 => |v| v.clamp(-1.0, 1.0),
        Act::Relu6 => |v| v.clamp(0.0, 6.0),
        Act::Tanh => f32::tanh,
    }
}

fn activate(mut t: Tensor, a: Act) -> Tensor {
    if a != Act::None {
        let f = act_fn(a);
        t.data.iter_mut().for_each(|v| *v = f(*v));
    }
    t
}

fn map(x: &Tensor, f: impl Fn(f32) -> f32) -> Tensor {
    Tensor { shape: x.shape.clone(), data: x.data.iter().map(|v| f(*v)).collect() }
}

/// Elementwise with NumPy broadcasting (up to 6 dimensions).
fn binary(a: &Tensor, b: &Tensor, f: impl Fn(f32, f32) -> f32, act: Act) -> Result<Tensor, Error> {
    let g = act_fn(act);
    if a.shape == b.shape {
        return Ok(Tensor { shape: a.shape.clone(), data: a.data.iter().zip(&b.data).map(|(x, y)| g(f(*x, *y))).collect() });
    }
    let rank = a.shape.len().max(b.shape.len());
    if rank > 6 {
        return Err(bad("broadcast rank > 6"));
    }
    let pad_to = |s: &[usize]| -> Vec<usize> { std::iter::repeat_n(1, rank - s.len()).chain(s.iter().copied()).collect() };
    let (sa, sb) = (pad_to(&a.shape), pad_to(&b.shape));
    let mut out = Vec::with_capacity(rank);
    for (x, y) in sa.iter().zip(&sb) {
        out.push(match (*x, *y) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return Err(bad(format!("can't broadcast {:?} with {:?}", a.shape, b.shape))),
        });
    }
    let strides = |s: &[usize]| -> Vec<usize> {
        let mut st = vec![0usize; rank];
        let mut acc = 1usize;
        for d in (0..rank).rev() {
            st[d] = if s[d] == 1 { 0 } else { acc };
            acc = acc.saturating_mul(s[d]);
        }
        st
    };
    let (ta, tb) = (strides(&sa), strides(&sb));
    let n = numel(&out)?;
    let mut data = Vec::with_capacity(n);
    let mut idx = vec![0usize; rank];
    for _ in 0..n {
        let (mut ia, mut ib) = (0usize, 0usize);
        for d in 0..rank {
            ia += idx[d] * ta[d];
            ib += idx[d] * tb[d];
        }
        let (x, y) = (a.data.get(ia).copied().unwrap_or(0.0), b.data.get(ib).copied().unwrap_or(0.0));
        data.push(g(f(x, y)));
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < out[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(Tensor { shape: out, data })
}

fn pad(x: &Tensor, p: &Tensor) -> Result<Tensor, Error> {
    let rank = x.shape.len();
    if p.data.len() != rank * 2 || rank == 0 || rank > 6 {
        return Err(bad("pad: paddings don't match the input rank"));
    }
    let mut before = Vec::with_capacity(rank);
    let mut out = Vec::with_capacity(rank);
    for d in 0..rank {
        let (b, a) = (p.data[2 * d], p.data[2 * d + 1]);
        if !(0.0..=4096.0).contains(&b) || !(0.0..=4096.0).contains(&a) {
            return Err(bad("pad: padding out of range"));
        }
        before.push(b as usize);
        out.push(x.shape[d] + b as usize + a as usize);
    }
    let mut y = Tensor::zeros(out.clone())?;
    // copy row by row along the last axis
    let last = x.shape[rank - 1];
    let rows = x.data.len().checked_div(last).unwrap_or(0);
    let mut idx = vec![0usize; rank.saturating_sub(1)];
    for r in 0..rows {
        let mut off = 0usize;
        for d in 0..rank - 1 {
            off = off * out[d] + idx[d] + before[d];
        }
        off = off * out[rank - 1] + before[rank - 1];
        let src = x.data.get(r * last..(r + 1) * last).ok_or_else(|| bad("pad: input too short"))?;
        y.data.get_mut(off..off + last).ok_or_else(|| bad("pad: output too short"))?.copy_from_slice(src);
        for d in (0..rank - 1).rev() {
            idx[d] += 1;
            if idx[d] < x.shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(y)
}

fn concat(ins: &[&Tensor], axis: i32) -> Result<Tensor, Error> {
    let first = ins.first().ok_or_else(|| bad("concat of nothing"))?;
    let rank = first.shape.len();
    let ax = if axis < 0 { rank as i64 + i64::from(axis) } else { i64::from(axis) };
    let ax = usize::try_from(ax).ok().filter(|a| *a < rank).ok_or_else(|| bad("concat axis out of range"))?;
    let mut shape = first.shape.clone();
    shape[ax] = 0;
    for t in ins {
        if t.shape.len() != rank || t.shape.iter().enumerate().any(|(d, s)| d != ax && *s != first.shape[d]) {
            return Err(bad("concat of mismatched shapes"));
        }
        shape[ax] += t.shape[ax];
    }
    let outer: usize = first.shape[..ax].iter().product();
    let mut data = Vec::with_capacity(numel(&shape)?);
    for o in 0..outer {
        for t in ins {
            let inner: usize = t.shape[ax..].iter().product();
            data.extend_from_slice(t.data.get(o * inner..(o + 1) * inner).ok_or_else(|| bad("concat input too short"))?);
        }
    }
    Tensor::new(shape, data)
}

/// Output size and leading padding of a window op along one axis.
fn window(n: usize, k: usize, stride: usize, dil: usize, pad: Padding) -> Result<(usize, usize), Error> {
    let eff = (k.saturating_sub(1)).saturating_mul(dil).saturating_add(1);
    match pad {
        Padding::Same => {
            let out = n.div_ceil(stride.max(1));
            let need = (out.saturating_sub(1) * stride + eff).saturating_sub(n);
            Ok((out, need / 2))
        }
        Padding::Valid => {
            if n < eff {
                return Err(bad("window larger than its input"));
            }
            Ok(((n - eff) / stride.max(1) + 1, 0))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn conv(x: &Tensor, w: &Tensor, b: Option<&Tensor>, pad: Padding, stride: (usize, usize), dil: (usize, usize), act: Act) -> Result<Tensor, Error> {
    let (h, wd, ci) = x.hwc()?;
    let [co, kh, kw, wci] = *w.shape.as_slice() else { return Err(bad("conv weights must be [o,h,w,i]")) };
    if wci != ci {
        return Err(bad(format!("conv: {ci} input channels, weights take {wci}")));
    }
    let bias = bias_of(b, co)?;
    let (oh, pt) = window(h, kh, stride.0, dil.0, pad)?;
    let (ow, pl) = window(wd, kw, stride.1, dil.1, pad)?;
    let mut y = Tensor::zeros(vec![1, oh, ow, co])?;
    let f = act_fn(act);
    let mut acc = vec![0f32; co];
    for oy in 0..oh {
        for ox in 0..ow {
            acc.copy_from_slice(&bias);
            for ky in 0..kh {
                let iy = (oy * stride.0 + ky * dil.0) as isize - pt as isize;
                if iy < 0 || iy as usize >= h {
                    continue;
                }
                for kx in 0..kw {
                    let ix = (ox * stride.1 + kx * dil.1) as isize - pl as isize;
                    if ix < 0 || ix as usize >= wd {
                        continue;
                    }
                    let base = (iy as usize * wd + ix as usize) * ci;
                    let inp = &x.data[base..base + ci];
                    for (o, a) in acc.iter_mut().enumerate() {
                        let wb = ((o * kh + ky) * kw + kx) * ci;
                        let wr = &w.data[wb..wb + ci];
                        let mut s = 0f32;
                        for (p, q) in inp.iter().zip(wr) {
                            s += p * q;
                        }
                        *a += s;
                    }
                }
            }
            let ob = (oy * ow + ox) * co;
            for (d, a) in y.data[ob..ob + co].iter_mut().zip(&acc) {
                *d = f(*a);
            }
        }
    }
    Ok(y)
}

fn bias_of(b: Option<&Tensor>, n: usize) -> Result<Vec<f32>, Error> {
    match b {
        Some(b) if b.data.len() == n => Ok(b.data.clone()),
        Some(_) => Err(bad("bias length doesn't match the output channels")),
        None => Ok(vec![0.0; n]),
    }
}

#[allow(clippy::too_many_arguments)]
fn depthwise(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    pad: Padding,
    stride: (usize, usize),
    dil: (usize, usize),
    mult: usize,
    act: Act,
) -> Result<Tensor, Error> {
    let (h, wd, ci) = x.hwc()?;
    let [1, kh, kw, co] = *w.shape.as_slice() else { return Err(bad("depthwise weights must be [1,h,w,c]")) };
    if co != ci.saturating_mul(mult) {
        return Err(bad("depthwise: channel count doesn't match the multiplier"));
    }
    let bias = bias_of(b, co)?;
    let (oh, pt) = window(h, kh, stride.0, dil.0, pad)?;
    let (ow, pl) = window(wd, kw, stride.1, dil.1, pad)?;
    let mut y = Tensor::zeros(vec![1, oh, ow, co])?;
    let f = act_fn(act);
    let mut acc = vec![0f32; co];
    for oy in 0..oh {
        for ox in 0..ow {
            acc.copy_from_slice(&bias);
            for ky in 0..kh {
                let iy = (oy * stride.0 + ky * dil.0) as isize - pt as isize;
                if iy < 0 || iy as usize >= h {
                    continue;
                }
                for kx in 0..kw {
                    let ix = (ox * stride.1 + kx * dil.1) as isize - pl as isize;
                    if ix < 0 || ix as usize >= wd {
                        continue;
                    }
                    let base = (iy as usize * wd + ix as usize) * ci;
                    let wb = (ky * kw + kx) * co;
                    let wr = &w.data[wb..wb + co];
                    if mult == 1 {
                        for ((a, p), q) in acc.iter_mut().zip(&x.data[base..base + ci]).zip(wr) {
                            *a += p * q;
                        }
                    } else {
                        for (o, a) in acc.iter_mut().enumerate() {
                            *a += x.data[base + o / mult] * wr[o];
                        }
                    }
                }
            }
            let ob = (oy * ow + ox) * co;
            for (d, a) in y.data[ob..ob + co].iter_mut().zip(&acc) {
                *d = f(*a);
            }
        }
    }
    Ok(y)
}

fn pool(x: &Tensor, pad: Padding, stride: (usize, usize), filter: (usize, usize), max: bool) -> Result<Tensor, Error> {
    let (h, wd, c) = x.hwc()?;
    let (oh, pt) = window(h, filter.0, stride.0, 1, pad)?;
    let (ow, pl) = window(wd, filter.1, stride.1, 1, pad)?;
    let mut y = Tensor::zeros(vec![1, oh, ow, c])?;
    for oy in 0..oh {
        for ox in 0..ow {
            let ob = (oy * ow + ox) * c;
            for ch in 0..c {
                let (mut m, mut s, mut n) = (f32::NEG_INFINITY, 0f32, 0u32);
                for ky in 0..filter.0 {
                    let iy = (oy * stride.0 + ky) as isize - pt as isize;
                    if iy < 0 || iy as usize >= h {
                        continue;
                    }
                    for kx in 0..filter.1 {
                        let ix = (ox * stride.1 + kx) as isize - pl as isize;
                        if ix < 0 || ix as usize >= wd {
                            continue;
                        }
                        let v = x.data[(iy as usize * wd + ix as usize) * c + ch];
                        m = m.max(v);
                        s += v;
                        n += 1;
                    }
                }
                y.data[ob + ch] = if n == 0 {
                    0.0
                } else if max {
                    m
                } else {
                    s / n as f32
                };
            }
        }
    }
    Ok(y)
}

fn resize(x: &Tensor, out: &[usize], align: bool, half: bool) -> Result<Tensor, Error> {
    let (h, w, c) = x.hwc()?;
    let [1, oh, ow, oc] = *out else { return Err(bad("resize: output must be [1,h,w,c]")) };
    if oc != c || h == 0 || w == 0 {
        return Err(bad("resize: bad shape"));
    }
    let scale = |i: usize, o: usize| -> f32 { if align && o > 1 { (i as f32 - 1.0) / (o as f32 - 1.0) } else { i as f32 / o.max(1) as f32 } };
    let (sy, sx) = (scale(h, oh), scale(w, ow));
    let mut y = Tensor::zeros(vec![1, oh, ow, c])?;
    for oy in 0..oh {
        let fy = if half { ((oy as f32 + 0.5) * sy - 0.5).max(0.0) } else { oy as f32 * sy };
        let y0 = (fy.floor() as usize).min(h - 1);
        let y1 = (y0 + 1).min(h - 1);
        let dy = fy - y0 as f32;
        for ox in 0..ow {
            let fx = if half { ((ox as f32 + 0.5) * sx - 0.5).max(0.0) } else { ox as f32 * sx };
            let x0 = (fx.floor() as usize).min(w - 1);
            let x1 = (x0 + 1).min(w - 1);
            let dx = fx - x0 as f32;
            for ch in 0..c {
                let p = |yy: usize, xx: usize| x.data[(yy * w + xx) * c + ch];
                let top = p(y0, x0) + (p(y0, x1) - p(y0, x0)) * dx;
                let bot = p(y1, x0) + (p(y1, x1) - p(y1, x0)) * dx;
                y.data[(oy * ow + ox) * c + ch] = top + (bot - top) * dy;
            }
        }
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(shape: &[usize], data: &[f32]) -> Tensor {
        Tensor::new(shape.to_vec(), data.to_vec()).unwrap()
    }

    #[test]
    fn f16_decodes() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-12);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
    }

    #[test]
    fn conv_same_padding_and_bias() {
        // 3x3 input, one 3x3 kernel of ones, bias 1: each output = 1 + sum of its neighbourhood
        let x = t(&[1, 3, 3, 1], &[1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let w = t(&[1, 3, 3, 1], &[1.0; 9]);
        let b = t(&[1], &[1.0]);
        let y = conv(&x, &w, Some(&b), Padding::Same, (1, 1), (1, 1), Act::None).unwrap();
        assert_eq!(y.shape, vec![1, 3, 3, 1]);
        assert_eq!(y.data, vec![13., 22., 17., 28., 46., 34., 25., 40., 29.]);
        let y = conv(&x, &w, Some(&b), Padding::Same, (2, 2), (1, 1), Act::None).unwrap();
        assert_eq!(y.shape, vec![1, 2, 2, 1]);
        assert_eq!(y.data, vec![13., 17., 25., 29.]);
    }

    #[test]
    fn depthwise_and_pool_and_pad() {
        let x = t(&[1, 2, 2, 2], &[1., 10., 2., 20., 3., 30., 4., 40.]);
        let w = t(&[1, 1, 1, 2], &[2.0, -1.0]);
        let y = depthwise(&x, &w, None, Padding::Valid, (1, 1), (1, 1), 1, Act::Relu).unwrap();
        assert_eq!(y.data, vec![2., 0., 4., 0., 6., 0., 8., 0.]);
        let m = pool(&x, Padding::Valid, (2, 2), (2, 2), true).unwrap();
        assert_eq!(m.data, vec![4., 40.]);
        let a = pool(&x, Padding::Valid, (2, 2), (2, 2), false).unwrap();
        assert_eq!(a.data, vec![2.5, 25.]);
        let p = pad(&t(&[1, 1, 1, 2], &[5., 6.]), &t(&[4, 2], &[0., 0., 1., 0., 0., 1., 0., 1.])).unwrap();
        assert_eq!(p.shape, vec![1, 2, 2, 3]);
        assert_eq!(p.data, vec![0.; 6].into_iter().chain([5., 6., 0., 0., 0., 0.]).collect::<Vec<_>>());
    }

    #[test]
    fn broadcast_prelu_and_concat() {
        let x = t(&[1, 1, 2, 2], &[-1., 2., -3., 4.]);
        let s = t(&[1, 1, 2], &[0.5, 0.25]);
        let y = binary(&x, &s, |a, s| if a >= 0.0 { a } else { a * s }, Act::None).unwrap();
        assert_eq!(y.data, vec![-0.5, 2., -1.5, 4.]);
        let c = concat(&[&t(&[1, 2, 1], &[1., 2.]), &t(&[1, 2, 2], &[3., 4., 5., 6.])], -1).unwrap();
        assert_eq!(c.shape, vec![1, 2, 3]);
        assert_eq!(c.data, vec![1., 3., 4., 2., 5., 6.]);
        assert!(binary(&t(&[2], &[1., 2.]), &t(&[3], &[1., 2., 3.]), |a, b| a + b, Act::None).is_err());
    }

    #[test]
    fn hostile_shapes_are_errors() {
        let x = t(&[1, 2, 2, 1], &[1.0; 4]);
        assert!(conv(&x, &t(&[1, 3, 3, 2], &[0.0; 18]), None, Padding::Same, (1, 1), (1, 1), Act::None).is_err());
        assert!(conv(&x, &t(&[1, 3, 3, 1], &[0.0; 9]), None, Padding::Valid, (1, 1), (1, 1), Act::None).is_err());
        assert!(pad(&x, &t(&[1], &[1.0])).is_err());
        assert!(pad(&x, &t(&[4, 2], &[1e9; 8])).is_err());
        assert!(Tensor::new(vec![usize::MAX, 2], vec![]).is_err());
        assert!(concat(&[], 0).is_err());
    }
}

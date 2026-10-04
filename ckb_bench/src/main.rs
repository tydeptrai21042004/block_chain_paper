#![no_std]
#![no_main]

mod merkle_vectors;
mod lenet_vectors;

use ckb_hash::blake2b_256;
use ckb_std::{default_alloc, entry};
use core::ptr::{addr_of_mut, write_volatile};
use merkle_vectors::{
    CANONICAL_DEPTH, CANONICAL_LEAF_COUNT, CANONICAL_LEAVES, CANONICAL_PROOFS,
    CANONICAL_ROOT, GENERIC_SIBLINGS,
};
use lenet_vectors::{
    checkpoint_matches, load_checkpoint, CONV1_B, CONV1_W, CONV2_B, CONV2_W,
    FC1_B, FC1_W, FC2_B, FC2_W, FC3_B, FC3_W, LENET_MAX_STATE, LENET_STATE_LENS,
};

entry!(program_entry);
default_alloc!();

static mut SINK: i64 = 0;

fn parse_usize(s: &str, default_value: usize) -> usize {
    let mut value = 0usize;
    let mut seen = false;
    for b in s.as_bytes() {
        if *b < b'0' || *b > b'9' {
            return default_value;
        }
        seen = true;
        value = value.saturating_mul(10).saturating_add((*b - b'0') as usize);
    }
    if seen { value } else { default_value }
}

#[inline(always)]
fn value_a(i: usize) -> i64 {
    ((i.wrapping_mul(37).wrapping_add(11) % 257) as i64) - 128
}

#[inline(always)]
fn value_b(i: usize) -> i64 {
    ((i.wrapping_mul(29).wrapping_add(7) % 113) as i64) - 56
}

#[inline(never)]
fn bench_relu(n: usize) -> i64 {
    let mut acc = 0i64;
    for i in 0..n {
        let x = value_a(i);
        let y = if x > 0 { x } else { 0 };
        acc = acc.wrapping_add(y.wrapping_mul((i as i64) + 1));
    }
    acc
}

#[inline(never)]
fn bench_dot(n: usize) -> i64 {
    let mut acc = 17i64;
    for i in 0..n {
        acc = acc.wrapping_add(value_a(i).wrapping_mul(value_b(i)));
    }
    acc
}

#[inline(never)]
fn bench_gemm(dim: usize) -> i64 {
    let d = if dim == 0 { 1 } else if dim > 64 { 64 } else { dim };
    let mut checksum = 0i64;
    for i in 0..d {
        for j in 0..d {
            let mut acc = 0i64;
            for k in 0..d {
                let a = value_a(i * d + k);
                let b = value_b(k * d + j);
                acc = acc.wrapping_add(a.wrapping_mul(b));
            }
            checksum = checksum.wrapping_add(acc.wrapping_mul((i + j + 1) as i64));
        }
    }
    checksum
}

#[inline(never)]
fn bench_conv(side: usize) -> i64 {
    let s = if side < 3 { 3 } else if side > 64 { 64 } else { side };
    let kernel: [i64; 9] = [1, -2, 1, 0, 3, 0, -1, 2, -1];
    let mut checksum = 0i64;
    for y in 0..(s - 2) {
        for x in 0..(s - 2) {
            let mut acc = 0i64;
            for ky in 0..3 {
                for kx in 0..3 {
                    let p = value_a((y + ky) * s + (x + kx));
                    acc = acc.wrapping_add(p.wrapping_mul(kernel[ky * 3 + kx]));
                }
            }
            checksum = checksum.wrapping_add(acc.wrapping_mul((y + x + 1) as i64));
        }
    }
    checksum
}

#[inline(always)]
fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut pair = [0u8; 64];
    pair[..32].copy_from_slice(left);
    pair[32..].copy_from_slice(right);
    blake2b_256(&pair)
}

/// Verify an already-hashed Merkle leaf against a supplied authentication path.
///
/// Sibling digests are inputs to verification.  The verifier therefore performs
/// exactly one parent hash per proof level.  Left/right ordering follows the
/// actual leaf-index bit at each level.
#[inline(never)]
fn verify_merkle_path(
    leaf_hash: [u8; 32],
    leaf_index: usize,
    siblings: &[[u8; 32]],
) -> [u8; 32] {
    let mut current = leaf_hash;
    for (level, sibling) in siblings.iter().enumerate() {
        current = if ((leaf_index >> level) & 1) == 0 {
            hash_pair(&current, sibling)
        } else {
            hash_pair(sibling, &current)
        };
    }
    current
}

#[inline(never)]
fn bench_merkle_access(depth: usize, leaf_index: usize) -> i64 {
    // Separate in-memory proof-access microbenchmark.  This intentionally does
    // no hashing, so it can be reported independently from authentication work.
    // It is not a substitute for measuring real witness syscalls/decoding.
    let d = if depth == 0 {
        1
    } else if depth > GENERIC_SIBLINGS.len() {
        GENERIC_SIBLINGS.len()
    } else {
        depth
    };

    let mut checksum = 0i64;
    if d == CANONICAL_DEPTH && leaf_index < CANONICAL_LEAF_COUNT {
        for (level, sibling) in CANONICAL_PROOFS[leaf_index].iter().enumerate() {
            for (byte_index, byte) in sibling.iter().enumerate() {
                let weight = ((level * 32 + byte_index + 1) as i64).wrapping_mul(17);
                checksum = checksum.wrapping_add((*byte as i64).wrapping_mul(weight));
            }
        }
    } else {
        for (level, sibling) in GENERIC_SIBLINGS[..d].iter().enumerate() {
            for (byte_index, byte) in sibling.iter().enumerate() {
                let weight = ((level * 32 + byte_index + 1) as i64).wrapping_mul(17);
                checksum = checksum.wrapping_add((*byte as i64).wrapping_mul(weight));
            }
        }
    }
    checksum
}

#[inline(never)]
fn bench_merkle(depth: usize, leaf_index: usize) -> i64 {
    // The publication experiment uses depth=4 because 13 committed states are
    // padded to 16 leaves.  For that case, validate against a known correct root.
    // Other depths are a hash-scaling microbenchmark with pre-supplied siblings.
    let d = if depth == 0 {
        1
    } else if depth > GENERIC_SIBLINGS.len() {
        GENERIC_SIBLINGS.len()
    } else {
        depth
    };

    let current = if d == CANONICAL_DEPTH && leaf_index < CANONICAL_LEAF_COUNT {
        let root = verify_merkle_path(
            CANONICAL_LEAVES[leaf_index],
            leaf_index,
            &CANONICAL_PROOFS[leaf_index],
        );
        if root != CANONICAL_ROOT {
            // A failed canonical proof must never be reported as a valid timing.
            return i64::MIN;
        }
        root
    } else {
        let synthetic_leaf = [0x42u8; 32];
        verify_merkle_path(synthetic_leaf, leaf_index, &GENERIC_SIBLINGS[..d])
    };

    let mut out = [0u8; 8];
    out.copy_from_slice(&current[..8]);
    i64::from_le_bytes(out)
}

#[inline(always)]
fn quantize_i64(acc: i64, shift: u32) -> i32 {
    let shifted = acc >> shift;
    if shifted > 32767 {
        32767
    } else if shifted < -32768 {
        -32768
    } else {
        shifted as i32
    }
}

#[inline(never)]
fn qconv2d(
    input: &[i32],
    cin: usize,
    h: usize,
    w: usize,
    cout: usize,
    kernel: usize,
    weights: &[i8],
    bias: &[i32],
    shift: u32,
    output: &mut [i32],
) -> usize {
    let oh = h - kernel + 1;
    let ow = w - kernel + 1;
    let out_len = cout * oh * ow;
    if input.len() != cin * h * w
        || weights.len() != cout * cin * kernel * kernel
        || bias.len() != cout
        || output.len() < out_len
    {
        return 0;
    }

    for oc in 0..cout {
        for y in 0..oh {
            for x in 0..ow {
                let mut acc = bias[oc] as i64;
                for ic in 0..cin {
                    for ky in 0..kernel {
                        for kx in 0..kernel {
                            let src = ic * h * w + (y + ky) * w + (x + kx);
                            let wi = ((oc * cin + ic) * kernel + ky) * kernel + kx;
                            acc += (input[src] as i64) * (weights[wi] as i64);
                        }
                    }
                }
                output[oc * oh * ow + y * ow + x] = quantize_i64(acc, shift);
            }
        }
    }
    out_len
}

#[inline(never)]
fn qrelu(input: &[i32], output: &mut [i32]) -> usize {
    if output.len() < input.len() {
        return 0;
    }
    for (dst, src) in output[..input.len()].iter_mut().zip(input.iter()) {
        *dst = if *src > 0 { *src } else { 0 };
    }
    input.len()
}

#[inline(never)]
fn qmaxpool2x2(
    input: &[i32],
    channels: usize,
    h: usize,
    w: usize,
    output: &mut [i32],
) -> usize {
    if input.len() != channels * h * w || h % 2 != 0 || w % 2 != 0 {
        return 0;
    }
    let oh = h / 2;
    let ow = w / 2;
    let out_len = channels * oh * ow;
    if output.len() < out_len {
        return 0;
    }

    let mut out_index = 0usize;
    for c in 0..channels {
        let base = c * h * w;
        for y in 0..oh {
            for x in 0..ow {
                let i0 = base + (2 * y) * w + 2 * x;
                let mut m = input[i0];
                let v1 = input[i0 + 1];
                let v2 = input[i0 + w];
                let v3 = input[i0 + w + 1];
                if v1 > m { m = v1; }
                if v2 > m { m = v2; }
                if v3 > m { m = v3; }
                output[out_index] = m;
                out_index += 1;
            }
        }
    }
    out_len
}

#[inline(never)]
fn qcopy(input: &[i32], output: &mut [i32]) -> usize {
    if output.len() < input.len() {
        return 0;
    }
    output[..input.len()].copy_from_slice(input);
    input.len()
}

#[inline(never)]
fn qfc(
    input: &[i32],
    outputs: usize,
    weights: &[i8],
    bias: &[i32],
    shift: u32,
    output: &mut [i32],
) -> usize {
    let inputs = input.len();
    if weights.len() != outputs * inputs || bias.len() != outputs || output.len() < outputs {
        return 0;
    }
    for o in 0..outputs {
        let mut acc = bias[o] as i64;
        let base = o * inputs;
        for i in 0..inputs {
            acc += (input[i] as i64) * (weights[base + i] as i64);
        }
        output[o] = quantize_i64(acc, shift);
    }
    outputs
}

#[inline(never)]
fn apply_lenet_op(op: usize, input: &[i32], output: &mut [i32]) -> usize {
    match op {
        1 => qconv2d(input, 1, 32, 32, 6, 5, &CONV1_W, &CONV1_B, 2, output),
        2 => qrelu(input, output),
        3 => qmaxpool2x2(input, 6, 28, 28, output),
        4 => qconv2d(input, 6, 14, 14, 16, 5, &CONV2_W, &CONV2_B, 4, output),
        5 => qrelu(input, output),
        6 => qmaxpool2x2(input, 16, 10, 10, output),
        7 => qcopy(input, output),
        8 => qfc(input, 120, &FC1_W, &FC1_B, 7, output),
        9 => qrelu(input, output),
        10 => qfc(input, 84, &FC2_W, &FC2_B, 6, output),
        11 => qrelu(input, output),
        12 => qfc(input, 10, &FC3_W, &FC3_B, 5, output),
        _ => 0,
    }
}

#[inline(never)]
fn checksum_state(data: &[i32]) -> i64 {
    let mut checksum = 0i64;
    for (index, value) in data.iter().enumerate() {
        checksum = checksum.wrapping_add(
            (*value as i64).wrapping_mul(((index % 97) + 1) as i64),
        );
    }
    checksum
}

#[inline(never)]
fn bench_lenet_block(start: usize, end: usize) -> i64 {
    // A genuine state-chained interval verifier.  It loads the exact committed
    // checkpoint S_start generated by generate_lenet_vectors.py, re-executes
    // f_{start+1}..f_end using deterministic quantized LeNet-5 dimensions, and
    // accepts the benchmark only when the result exactly equals S_end.
    //
    // This benchmark includes checkpoint memory copy, arithmetic, and final
    // state comparison.  It still excludes transaction/witness syscalls and
    // Merkle authentication, which are measured separately.
    if start >= end || end > 12 {
        return i64::MIN;
    }

    let mut a = [0i32; LENET_MAX_STATE];
    let mut b = [0i32; LENET_MAX_STATE];
    let mut len = load_checkpoint(start, &mut a);
    if len != LENET_STATE_LENS[start] {
        return i64::MIN;
    }

    let mut current_in_a = true;
    let mut op = start + 1;
    while op <= end {
        let next_len = if current_in_a {
            apply_lenet_op(op, &a[..len], &mut b)
        } else {
            apply_lenet_op(op, &b[..len], &mut a)
        };
        if next_len == 0 || next_len != LENET_STATE_LENS[op] {
            return i64::MIN;
        }
        len = next_len;
        current_in_a = !current_in_a;
        op += 1;
    }

    let final_state = if current_in_a { &a[..len] } else { &b[..len] };
    if !checkpoint_matches(end, final_state, len) {
        return i64::MIN;
    }
    checksum_state(final_state)
}

fn program_entry() -> i8 {
    let kind = option_env!("BENCH_KIND").unwrap_or("relu");
    let size = parse_usize(option_env!("BENCH_SIZE").unwrap_or("64"), 64);
    let start = parse_usize(option_env!("BENCH_START").unwrap_or("0"), 0);
    let end = parse_usize(option_env!("BENCH_END").unwrap_or("1"), 1);
    let leaf_index = parse_usize(option_env!("BENCH_LEAF_INDEX").unwrap_or("6"), 6);

    let checksum = match kind {
        "relu" => bench_relu(size),
        "dot" => bench_dot(size),
        "gemm" => bench_gemm(size),
        "conv" => bench_conv(size),
        "merkle" => bench_merkle(size, leaf_index),
        "merkle_access" => bench_merkle_access(size, leaf_index),
        "lenet" => bench_lenet_block(start, end),
        _ => return 2,
    };

    unsafe {
        write_volatile(addr_of_mut!(SINK), checksum);
    }
    0
}

#![no_std]
#![no_main]

use ckb_hash::blake2b_256;
use ckb_std::entry;
use core::ptr::{addr_of_mut, write_volatile};

entry!(program_entry);

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

#[inline(never)]
fn bench_merkle(depth: usize) -> i64 {
    let d = if depth == 0 { 1 } else if depth > 32 { 32 } else { depth };
    let mut current = blake2b_256(b"cellvg-leaf");
    for level in 0..d {
        let mut sibling_seed = [0u8; 16];
        for (i, b) in sibling_seed.iter_mut().enumerate() {
            *b = (level as u8).wrapping_mul(17).wrapping_add(i as u8);
        }
        let sibling = blake2b_256(&sibling_seed);
        let mut pair = [0u8; 64];
        if level % 2 == 0 {
            pair[..32].copy_from_slice(&current);
            pair[32..].copy_from_slice(&sibling);
        } else {
            pair[..32].copy_from_slice(&sibling);
            pair[32..].copy_from_slice(&current);
        }
        current = blake2b_256(&pair);
    }
    let mut out = [0u8; 8];
    out.copy_from_slice(&current[..8]);
    i64::from_le_bytes(out)
}

#[inline(never)]
fn conv_work(out_elements: usize, macs_per_output: usize, seed: usize) -> i64 {
    let mut checksum = 0i64;
    for out in 0..out_elements {
        let mut acc = 0i64;
        for m in 0..macs_per_output {
            let idx = seed.wrapping_add(out.wrapping_mul(macs_per_output)).wrapping_add(m);
            acc = acc.wrapping_add(value_a(idx).wrapping_mul(value_b(idx + 19)));
        }
        checksum = checksum.wrapping_add(acc.wrapping_mul(((out + seed) % 31 + 1) as i64));
    }
    checksum
}

#[inline(never)]
fn pool2x2_work(out_elements: usize, seed: usize) -> i64 {
    let mut checksum = 0i64;
    for out in 0..out_elements {
        let base = seed + out * 4;
        let mut m = value_a(base);
        let b = value_a(base + 1);
        let c = value_a(base + 2);
        let d = value_a(base + 3);
        if b > m { m = b; }
        if c > m { m = c; }
        if d > m { m = d; }
        checksum = checksum.wrapping_add(m.wrapping_mul((out + 1) as i64));
    }
    checksum
}

#[inline(never)]
fn flatten_work(n: usize, seed: usize) -> i64 {
    let mut checksum = 0i64;
    for i in 0..n {
        checksum = checksum.wrapping_add(value_a(seed + i).wrapping_mul((i + 1) as i64));
    }
    checksum
}

#[inline(never)]
fn fc_work(inputs: usize, outputs: usize, seed: usize) -> i64 {
    let mut checksum = 0i64;
    for o in 0..outputs {
        let mut acc = value_b(seed + o);
        for i in 0..inputs {
            let idx = seed + o * inputs + i;
            acc = acc.wrapping_add(value_a(idx).wrapping_mul(value_b(idx + 23)));
        }
        checksum = checksum.wrapping_add(acc.wrapping_mul((o + 1) as i64));
    }
    checksum
}

#[inline(never)]
fn bench_lenet_op(op: usize) -> i64 {
    // Canonical LeNet-5 arithmetic dimensions (input padded to 32x32):
    // Conv1: 6*28*28 outputs, 1*5*5 MACs/output.
    // Conv2: 16*10*10 outputs, 6*5*5 MACs/output.
    match op {
        1 => conv_work(6 * 28 * 28, 25, 101),
        2 => bench_relu(6 * 28 * 28),
        3 => pool2x2_work(6 * 14 * 14, 301),
        4 => conv_work(16 * 10 * 10, 6 * 25, 401),
        5 => bench_relu(16 * 10 * 10),
        6 => pool2x2_work(16 * 5 * 5, 601),
        7 => flatten_work(16 * 5 * 5, 701),
        8 => fc_work(400, 120, 801),
        9 => bench_relu(120),
        10 => fc_work(120, 84, 1001),
        11 => bench_relu(84),
        12 => fc_work(84, 10, 1201),
        _ => 0,
    }
}

#[inline(never)]
fn bench_lenet_block(start: usize, end: usize) -> i64 {
    // start/end are 0-based transition indices with end exclusive.
    if start >= end || end > 12 {
        return 0;
    }
    let mut checksum = 0i64;
    let mut t = start;
    while t < end {
        checksum = checksum.wrapping_add(bench_lenet_op(t + 1).rotate_left((t % 31) as u32));
        t += 1;
    }
    checksum
}

fn program_entry() -> i8 {
    let kind = option_env!("BENCH_KIND").unwrap_or("relu");
    let size = parse_usize(option_env!("BENCH_SIZE").unwrap_or("64"), 64);
    let start = parse_usize(option_env!("BENCH_START").unwrap_or("0"), 0);
    let end = parse_usize(option_env!("BENCH_END").unwrap_or("1"), 1);

    let checksum = match kind {
        "relu" => bench_relu(size),
        "dot" => bench_dot(size),
        "gemm" => bench_gemm(size),
        "conv" => bench_conv(size),
        "merkle" => bench_merkle(size),
        "lenet" => bench_lenet_block(start, end),
        _ => return 2,
    };

    unsafe {
        write_volatile(addr_of_mut!(SINK), checksum);
    }
    0
}

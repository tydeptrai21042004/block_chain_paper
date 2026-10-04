fn main() {
    // Benchmark selection is compile-time (`option_env!` / cfg). Explicit rerun
    // directives prevent Cargo from reusing a binary built for another case.
    for key in [
        "BENCH_KIND",
        "BENCH_SIZE",
        "BENCH_START",
        "BENCH_END",
        "BENCH_LEAF_INDEX",
        "BENCH_TRACE",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }

    println!("cargo:rustc-check-cfg=cfg(trace_conv_heavy)");
    println!("cargo:rustc-check-cfg=cfg(trace_gemm_heavy)");
    match std::env::var("BENCH_TRACE").ok().as_deref() {
        Some("conv_heavy") => println!("cargo:rustc-cfg=trace_conv_heavy"),
        Some("gemm_heavy") => println!("cargo:rustc-cfg=trace_gemm_heavy"),
        _ => {}
    }
}

fn main() {
    // The benchmark selection is compiled through option_env! in src/main.rs.
    // Explicit rerun directives prevent Cargo from reusing a binary built for
    // a previous benchmark interval/size when these environment variables change.
    for key in ["BENCH_KIND", "BENCH_SIZE", "BENCH_START", "BENCH_END", "BENCH_LEAF_INDEX"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
}

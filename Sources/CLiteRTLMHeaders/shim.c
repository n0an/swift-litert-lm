// SwiftPM requires a C target to have at least one source file. The target's
// purpose is its `include/` folder: it declares the `CLiteRTLM` module for the
// macOS build, whose xcframework ships the dylib without headers (see
// Package.swift for why).
void clitertlm_headers_anchor(void) {}

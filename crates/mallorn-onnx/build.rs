fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile ONNX protobuf definitions
    prost_build::Config::new()
        .btree_map(["."])
        .compile_protos(&["proto/onnx.proto"], &["proto/"])?;

    println!("cargo:rerun-if-changed=proto/onnx.proto");

    Ok(())
}

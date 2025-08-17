fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("src/proto")
        .compile(
            &[
                "../proto/common.proto",
                "../proto/trajectory.proto",
                "../proto/defect_detection.proto",
                "../proto/criticality.proto",
                "../proto/reporting.proto",
                "../proto/gateway.proto",
            ],
            &["../proto"],
        )?;
    Ok(())
}
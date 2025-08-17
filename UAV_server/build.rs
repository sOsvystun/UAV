fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(false) // We only need the server part for this project
        .out_dir("src/protofbufs") // Output to a sub-directory for cleanliness
        .compile(
            &[
                "../../protos/services.proto",
                "../../protos/common.proto",
                "../../protos/trajectory.proto",
                "../../protos/defect_detection.proto",
                "../../protos/criticality.proto",
                "../../protos/reporting.proto",
            ],
            &["../../protos"], // Specify the include path
        )?;
    Ok(())
}

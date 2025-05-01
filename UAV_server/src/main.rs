mod trajectory_service;
mod detection_service;
mod compose_images;
mod criticality_assessor_service;
mod defect_expert_params;

use kube::{Api, Client, api::{PostParams, ObjectMeta}};
use kube::api::ResourceExt;
use serde::{Deserialize, Serialize};
use tokio;
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone, Debug)]
struct AppConfig {
    name: String,
    image: String,
    replicas: i32,
    port: i32,
    environment_variables: HashMap<String, String>,
}

async fn deploy_to_kubernetes(config: AppConfig) -> anyhow::Result<()> {
    let client = Client::try_default().await?;

    let deployments: Api<kube::api::Deployment> = Api::namespaced(client, "default");

    let deployment = kube::api::Deployment {
        metadata: ObjectMeta {
            name: Some(config.name.clone()),
            labels: Some(HashMap::from([
                ("app".into(), config.name.clone()),
            ])),
            ..Default::default()
        },
        spec: Some(kube::api::DeploymentSpec {
            replicas: Some(config.replicas),
            selector: kube::api::LabelSelector {
                match_labels: Some(HashMap::from([
                    ("app".into(), config.name.clone()),
                ])),
                ..Default::default()
            },
            template: kube::api::PodTemplateSpec {
                metadata: Some(ObjectMeta {
                    labels: Some(HashMap::from([
                        ("app".into(), config.name.clone()),
                    ])),
                    ..Default::default()
                }),
                spec: Some(kube::api::PodSpec {
                    containers: vec![kube::api::Container {
                        name: config.name.clone(),
                        image: Some(config.image.clone()),
                        ports: Some(vec![kube::api::ContainerPort {
                            container_port: config.port,
                            ..Default::default()
                        }]),
                        env: Some(config.environment_variables.iter().map(|(key, value)| {
                            kube::api::EnvVar {
                                name: key.clone(),
                                value: Some(value.clone()),
                                ..Default::default()
                            }
                        }).collect()),
                        resources: Some(kube::api::ResourceRequirements {
                            limits: Some(HashMap::from([
                                ("cpu".into(), "500m".into()),
                                ("memory".into(), "512Mi".into()),
                            ])),
                            requests: Some(HashMap::from([
                                ("cpu".into(), "250m".into()),
                                ("memory".into(), "256Mi".into()),
                            ])),
                        }),
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
            },
            ..Default::default()
        }),
        status: None,
    };

    deployments.create(&PostParams::default(), &deployment).await?;

    println!("Deployment created: {}", config.name);

    Ok(())
}

async fn initialize_system() -> anyhow::Result<()> {
    println!("Initializing Trajectory Service...");
    trajectory_service::start().await?;

    println!("Initializing Detection Service...");
    detection_service::start().await?;

    println!("Initializing Image Composition Module...");
    compose_images::start().await?;

    println!("Initializing Criticality Assessor Service...");
    criticality_assessor_service::start().await?;

    println!("Loading Expert Parameters...");
    defect_expert_params::load("expert_data.json");

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Starting Kubernetes deployment and system initialization...");

    let mut environment_variables = HashMap::new();
    environment_variables.insert("ENV".into(), "production".into());
    environment_variables.insert("LOG_LEVEL".into(), "info".into());

    let config = AppConfig {
        name: "wind-turbine-monitor".into(),
        image: "wind-turbine-monitor-image:v1.0".into(),
        replicas: 3,
        port: 8080,
        environment_variables,
    };

    deploy_to_kubernetes(config).await?;

    println!("Deployment completed successfully.");

    initialize_system().await?;

    trajectory_service::run().await?;
    detection_service::run().await?;
    compose_images::run().await?;
    criticality_assessor_service::run().await?;
    defect_expert_params::run().await?;

    println!("All system components are running.");

    Ok(())
}

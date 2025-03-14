# Kubernetes Deployments for ML Pipeline

This directory contains Kubernetes manifests and Helm charts for deploying the ML pipeline components.

## Architecture

The ML deployment follows a GitOps approach, separating infrastructure provisioning (Terraform) from application deployment (Kubernetes manifests).

### Directory Structure

kubernetes
    - base          --> Base configurations shared across environments 
    - ml-api        --> ML API Service 
    - mlflow        --> MLFlow experiment tracking
    - monitoring    -->  Prometheus / Grafana 
    - training      --> Model training jobs
    - overlays      --> customize overlays for diff envs
        - dev
        - staging
        - production
        - helm-charts 

## Deployment Approach

We use a combination of:

1. **Kustomize** for our own applications (ML API, training jobs)
2. **Helm** for third-party components (MLflow, Prometheus, etc.)
3. **ArgoCD/Flux** for GitOps-based continuous deployment

## Multi-Cloud Setup

The Kubernetes configurations are designed to be cloud-agnostic. Cloud-specific settings are injected through:

- ConfigMaps and Secrets managed by external systems
- Environment-specific overlays (for Kustomize)
- Values files (for Helm)

### Cloud Storage Integration

Each cloud provider's storage is integrated differently:

- **AWS**: S3 buckets accessed via IAM roles for service accounts
- **Azure**: Blob Storage accessed via Workload Identity
- **GCP**: GCS buckets accessed via Workload Identity

## Usage

### Manual Deployment

```bash
# Deploy base applications with Kustomize
kubectl apply -k overlays/dev

# Deploy MLflow using Helm
helm upgrade --install mlflow ./helm-charts/mlflow -f values/dev/mlflow.yaml

# Multi-Cloud ML Deployment - Additional Configuration

Let's complete your production-grade multi-cloud ML deployment configuration with these essential files:

## 1. Complete GCP Implementation: main.tf

```terraform
# ------------------------------------------------------------------------------
# GCP IMPLEMENTATION
# ------------------------------------------------------------------------------

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

locals {
  name_prefix = "${var.project_name}-${var.environment}"
  region      = var.region
  
  common_tags = {
    project     = var.project_name
    environment = var.environment
    terraform   = "true"
    owner       = var.owner
    cost_center = var.cost_center
  }
  
  # Machine type mappings
  machine_types = {
    "small"      = "e2-standard-2"
    "medium"     = "e2-standard-4"
    "large"      = "e2-standard-8"
    "xlarge"     = "e2-standard-16"
    "gpu-small"  = "n1-standard-4"
    "gpu-medium" = "n1-standard-8"
    "gpu-large"  = "n1-standard-16"
  }
  
  # Database tier mappings
  db_tiers = {
    "small"  = "db-custom-2-4096"
    "medium" = "db-custom-4-8192"
    "large"  = "db-custom-8-16384"
    "xlarge" = "db-custom-16-32768"
  }
  
  # Memory store tier mappings
  redis_tiers = {
    "small"  = "basic"
    "medium" = "standard_ha"
    "large"  = "standard_ha"
  }
  
  # Network configuration
  network_name          = "${local.name_prefix}-network"
  subnet_name           = "${local.name_prefix}-subnet"
  subnet_range          = var.vpc_cidr
  secondary_ranges = {
    pods     = "${local.name_prefix}-pods"
    services = "${local.name_prefix}-services"
  }
  pods_range     = "10.100.0.0/16"
  services_range = "10.200.0.0/20"
}

# ------------------------------------------------------------------------------
# VPC & NETWORK INFRASTRUCTURE
# ------------------------------------------------------------------------------
resource "google_compute_network" "vpc" {
  name                    = local.network_name
  auto_create_subnetworks = false
  routing_mode            = "GLOBAL"
}

resource "google_compute_subnetwork" "subnet" {
  name          = local.subnet_name
  ip_cidr_range = local.subnet_range
  region        = local.region
  network       = google_compute_network.vpc.id
  
  secondary_ip_range {
    range_name    = local.secondary_ranges.pods
    ip_cidr_range = local.pods_range
  }
  
  secondary_ip_range {
    range_name    = local.secondary_ranges.services
    ip_cidr_range = local.services_range
  }
  
  private_ip_google_access = true
  
  log_config {
    aggregation_interval = "INTERVAL_10_MIN"
    flow_sampling        = 0.5
    metadata             = "INCLUDE_ALL_METADATA"
  }
}

resource "google_compute_router" "router" {
  name    = "${local.name_prefix}-router"
  region  = local.region
  network = google_compute_network.vpc.id
}

resource "google_compute_router_nat" "nat" {
  name                               = "${local.name_prefix}-nat"
  router                             = google_compute_router.router.name
  region                             = local.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
  
  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# ------------------------------------------------------------------------------
# GKE CLUSTER
# ------------------------------------------------------------------------------
resource "google_container_cluster" "primary" {
  name     = "${local.name_prefix}-cluster"
  location = local.region
  
  # We can't create a cluster with no node pool defined, but we want to use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1
  
  # Enable VPC native networking
  networking_mode = "VPC_NATIVE"
  
  network    = google_compute_network.vpc.self_link
  subnetwork = google_compute_subnetwork.subnet.self_link
  
  ip_allocation_policy {
    cluster_secondary_range_name  = local.secondary_ranges.pods
    services_secondary_range_name = local.secondary_ranges.services
  }
  
  # Enable workload identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  # Enable Autopilot mode for serverless GKE
  enable_autopilot = var.environment == "production" ? false : true
  
  # For non-Autopilot clusters, configure advanced security options
  dynamic "binary_authorization" {
    for_each = var.environment == "production" ? [1] : []
    content {
      evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
    }
  }
  
  # Private cluster configuration for production
  dynamic "private_cluster_config" {
    for_each = var.environment == "production" ? [1] : []
    content {
      enable_private_nodes    = true
      enable_private_endpoint = false
      master_ipv4_cidr_block  = "172.16.0.0/28"
    }
  }
  
  # Release channel for automatic upgrades
  release_channel {
    channel = var.environment == "production" ? "STABLE" : "REGULAR"
  }
  
  # Enable shielded nodes
  node_config {
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }
  
  # Disable legacy Auth
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }
  
  maintenance_policy {
    recurring_window {
      start_time = "2022-01-01T00:00:00Z"
      end_time   = "2022-01-01T04:00:00Z"
      recurrence = "FREQ=WEEKLY;BYDAY=SA,SU"
    }
  }
  
  # Enable Network Policy
  network_policy {
    enabled  = true
    provider = "CALICO"
  }
  
  # Enable all needed addons
  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
    network_policy_config {
      disabled = false
    }
    gcp_filestore_csi_driver_config {
      enabled = true
    }
  }
}

# ------------------------------------------------------------------------------
# GKE NODE POOLS
# ------------------------------------------------------------------------------
resource "google_container_node_pool" "general" {
  count      = var.environment == "production" ? 1 : 0
  name       = "general"
  location   = local.region
  cluster    = google_container_cluster.primary.name
  node_count = var.min_nodes
  
  autoscaling {
    min_node_count = var.min_nodes
    max_node_count = var.max_nodes
  }
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  node_config {
    machine_type = local.machine_types["medium"]
    disk_type    = "pd-ssd"
    disk_size_gb = 100
    
    # Needed for gVisor
    sandbox_config {
      sandbox_type = "gvisor"
    }
    
    # GCE default service account
    service_account = google_service_account.gke.email
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    labels = {
      environment = var.environment
      nodepool   = "general"
    }
    
    # Enable workload identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}

resource "google_container_node_pool" "ml_compute" {
  count      = var.environment == "production" ? 1 : 0
  name       = "ml-compute"
  location   = local.region
  cluster    = google_container_cluster.primary.name
  
  autoscaling {
    min_node_count = 0
    max_node_count = 10
  }
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  node_config {
    machine_type = local.machine_types["large"]
    disk_type    = "pd-ssd"
    disk_size_gb = 200
    
    # GPU configuration
    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
      
      # Configure GPU driver installation on startup
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }
    
    # GCE default service account
    service_account = google_service_account.gke.email
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    taint {
      key    = "workload"
      value  = "ml-training"
      effect = "NO_SCHEDULE"
    }
    
    labels = {
      environment = var.environment
      nodepool    = "ml-compute"
      workload    = "ml-training"
    }
    
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}

# ------------------------------------------------------------------------------
# SERVICE ACCOUNTS
# ------------------------------------------------------------------------------
resource "google_service_account" "gke" {
  account_id   = "${local.name_prefix}-gke"
  display_name = "Service Account for GKE Nodes"
}

resource "google_service_account" "ml_model" {
  account_id   = "${local.name_prefix}-ml-model"
  display_name = "Service Account for ML Model Service"
}

# Grant required roles to GKE service account
resource "google_project_iam_member" "gke_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.gke.email}"
}

resource "google_project_iam_member" "gke_metrics_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.gke.email}"
}

resource "google_project_iam_member" "gke_artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.gke.email}"
}

# Grant model service account access to storage
resource "google_project_iam_member" "ml_storage_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.ml_model.email}"
}

# Configure workload identity for model service
resource "google_service_account_iam_binding" "ml_model_workload_identity" {
  service_account_id = google_service_account.ml_model.name
  role               = "roles/iam.workloadIdentityUser"
  
  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[ml-model/model-api]"
  ]
}

# ------------------------------------------------------------------------------
# DATABASE (CLOUD SQL)
# ------------------------------------------------------------------------------
resource "google_sql_database_instance" "postgres" {
  name             = "${local.name_prefix}-postgres"
  database_version = "POSTGRES_14"
  region           = local.region
  
  settings {
    tier = local.db_tiers[var.db_instance_type]
    
    backup_configuration {
      enabled                        = true
      binary_log_enabled             = false
      start_time                     = "02:00"
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 7
      }
    }
    
    maintenance_window {
      day          = 7  # Sunday
      hour         = 2
      update_track = "stable"
    }
    
    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = false
    }
    
    ip_configuration {
      ipv4_enabled    = var.environment != "production"
      private_network = var.environment == "production" ? google_compute_network.vpc.self_link : null
      require_ssl     = true
    }
    
    database_flags {
      name  = "log_min_duration_statement"
      value = "1000"
    }
    
    database_flags {
      name  = "max_connections"
      value = "100"
    }
    
    user_labels = local.common_tags
  }
  
  deletion_protection = var.environment == "production" ? true : false
}

resource "google_sql_database" "database" {
  name     = "mlmodels"
  instance = google_sql_database_instance.postgres.name
}

resource "random_password" "database_password" {
  length  = 16
  special = false
}

resource "google_sql_user" "user" {
  name     = "mluser"
  instance = google_sql_database_instance.postgres.name
  password = random_password.database_password.result
}

# ------------------------------------------------------------------------------
# REDIS (MEMORYSTORE)
# ------------------------------------------------------------------------------
resource "google_redis_instance" "redis" {
  name           = "${local.name_prefix}-redis"
  display_name   = "ML Model Cache"
  tier           = local.redis_tiers[var.redis_node_type]
  memory_size_gb = 1
  
  region             = local.region
  authorized_network = google_compute_network.vpc.id
  
  redis_version     = "REDIS_6_X"
  transit_encryption_mode = "SERVER_AUTHENTICATION"
  
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 2
        minutes = 0
      }
    }
  }
  
  labels = local.common_tags
}

# ------------------------------------------------------------------------------
# STORAGE (GCS)
# ------------------------------------------------------------------------------
resource "google_storage_bucket" "model_artifacts" {
  name     = "${local.name_prefix}-model-artifacts"
  location = local.region
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
  
  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD", "PUT", "POST", "DELETE"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
  
  labels = local.common_tags
}

resource "google_storage_bucket" "mlflow_artifacts" {
  name     = "${local.name_prefix}-mlflow-artifacts"
  location = local.region
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD", "PUT", "POST", "DELETE"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
  
  labels = local.common_tags
}

# ------------------------------------------------------------------------------
# CONTAINER REGISTRY
# ------------------------------------------------------------------------------
resource "google_artifact_registry_repository" "repository" {
  location      = local.region
  repository_id = "${local.name_prefix}-repo"
  format        = "DOCKER"
  
  labels = local.common_tags
}

# ------------------------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------------------------
output "kubernetes_cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.primary.name
}

output "kubernetes_cluster_host" {
  description = "GKE cluster endpoint"
  value       = "https://${google_container_cluster.primary.endpoint}"
  sensitive   = true
}

output "db_connection_name" {
  description = "Cloud SQL connection name"
  value       = google_sql_database_instance.postgres.connection_name
}

output "db_name" {
  description = "Database name"
  value       = google_sql_database.database.name
}

output "db_user" {
  description = "Database user"
  value       = google_sql_user.user.name
  sensitive   = true
}

output "db_password" {
  description = "Database password"
  value       = google_sql_user.user.password
  sensitive   = true
}

output "redis_host" {
  description = "Redis hostname"
  value       = google_redis_instance.redis.host
}

output "model_artifacts_bucket" {
  description = "GCS bucket for model artifacts"
  value       = google_storage_bucket.model_artifacts.name
}

output "mlflow_artifacts_bucket" {
  description = "GCS bucket for MLflow artifacts"
  value       = google_storage_bucket.mlflow_artifacts.name
}

output "container_registry" {
  description = "Artifact Registry URL for Docker images"
  value       = "${local.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.repository.repository_id}"
}

output "ml_model_service_account_email" {
  description = "Service account email for ML model service"
  value       = google_service_account.ml_model.email
}
```

## 2. GCP Variables: variables.tf

```terraform
# ------------------------------------------------------------------------------
# REQUIRED VARIABLES
# ------------------------------------------------------------------------------
variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for all resources"
  type        = string
  default     = "us-central1"
}

variable "vpc_cidr" {
  description = "CIDR for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# ------------------------------------------------------------------------------
# COMMON VARIABLES (passed from root module)
# ------------------------------------------------------------------------------
variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, production)"
  type        = string
}

variable "owner" {
  description = "Owner of the resources"
  type        = string
}

variable "cost_center" {
  description = "Cost center for billing"
  type        = string
}

variable "kubernetes_version" {
  description = "Kubernetes version for GKE cluster"
  type        = string
}

variable "min_nodes" {
  description = "Minimum number of GKE nodes"
  type        = number
}

variable "max_nodes" {
  description = "Maximum number of GKE nodes"
  type        = number
}

variable "desired_nodes" {
  description = "Desired number of GKE nodes"
  type        = number
}

variable "db_instance_type" {
  description = "Database instance type"
  type        = string
  default     = "small"
}

variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "small"
}

variable "management_ips" {
  description = "List of CIDR blocks that should have management access"
  type        = list(string)
  default     = []
}
```

## 3. GitOps Kubernetes Deployment: `kubernetes/README.md`

```markdown
# Kubernetes Deployments for ML Pipeline

This directory contains Kubernetes manifests and Helm charts for deploying the ML pipeline components.

## Architecture

The ML deployment follows a GitOps approach, separating infrastructure provisioning (Terraform) from application deployment (Kubernetes manifests).

### Directory Structure

```
kubernetes/
├── base/                   # Base configurations shared across environments
│   ├── ml-api/             # ML API service
│   ├── mlflow/             # MLflow experiment tracking
│   ├── monitoring/         # Prometheus, Grafana, etc.
│   └── training/           # Model training jobs
├── overlays/               # Kustomize overlays for different environments
│   ├── dev/
│   ├── staging/
│   └── production/
└── helm-charts/            # Helm charts for third-party components
```

## Deployment Approach

We use a combination of:

1. **Kustomize** for our own applications (ML API, training jobs)
2. **Helm** for third-party components (MLflow, Prometheus, etc.)
3. **ArgoCD/Flux** for GitOps-based continuous deployment

## Multi-Cloud Setup

The Kubernetes configurations are designed to be cloud-agnostic. Cloud-specific settings are injected through:

- ConfigMaps and Secrets managed by external systems
- Environment-specific overlays (for Kustomize)
- Values files (for Helm)

### Cloud Storage Integration

Each cloud provider's storage is integrated differently:

- **AWS**: S3 buckets accessed via IAM roles for service accounts
- **Azure**: Blob Storage accessed via Workload Identity
- **GCP**: GCS buckets accessed via Workload Identity

## Usage

### Manual Deployment

```bash
# Deploy base applications with Kustomize
kubectl apply -k overlays/dev

# Deploy MLflow using Helm
helm upgrade --install mlflow ./helm-charts/mlflow -f values/dev/mlflow.yaml
```

### GitOps Deployment

Configure your GitOps tool (ArgoCD/Flux) to sync from this directory, specifying the appropriate environment overlay.

## Security Notes

- All deployments use non-root users
- Resource limits are enforced on all pods
- Network policies restrict communication between components
- Secrets are managed externally and injected at runtime
```

## 4. Kubernetes Custom Resource for Model Training: `kubernetes/base/training/cronjob.yaml`

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: model-training
  labels:
    app: model-training
    component: ml-pipeline
spec:
  schedule: "0 2 * * *"  # Run at 2 AM every day
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      backoffLimit: 2
      activeDeadlineSeconds: 10800  # 3 hours timeout
      template:
        metadata:
          labels:
            app: model-training
            component: ml-pipeline
        spec:
          restartPolicy: Never
          serviceAccountName: model-training
          containers:
            - name: model-trainer
              image: ${REGISTRY_URL}/model-trainer:${TAG}
              imagePullPolicy: Always
              args:
                - "--config=/config/training_config.json"
                - "--output-dir=/models"
                - "--log-level=info"
              env:
                - name: MLFLOW_TRACKING_URI
                  value: http://mlflow.mlops:5000
                - name: PYTHONUNBUFFERED
                  value: "1"
                # Cloud-specific environment variables will be injected through ConfigMap
              envFrom:
                - configMapRef:
                    name: ml-training-config
                - secretRef:
                    name: ml-training-secrets
              resources:
                limits:
                  cpu: 4
                  memory: 16Gi
                  # GPU resources if needed
                  # nvidia.com/gpu: 1
                requests:
                  cpu: 1
                  memory: 8Gi
              volumeMounts:
                - name: models-volume
                  mountPath: /models
                - name: config-volume
                  mountPath: /config
                - name: data-volume
                  mountPath: /data
                  readOnly: true
              securityContext:
                allowPrivilegeEscalation: false
                runAsUser: 1000
                runAsGroup: 1000
                readOnlyRootFilesystem: false
          volumes:
            - name: models-volume
              persistentVolumeClaim:
                claimName: model-storage
            - name: config-volume
              configMap:
                name: training-config
            - name: data-volume
              persistentVolumeClaim:
                claimName: training-data
          affinity:
            nodeAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                nodeSelectorTerms:
                - matchExpressions:
                  - key: workload
                    operator: In
                    values:
                    - ml-training
```

## 5. Multi-Cloud ConfigMaps: `kubernetes/overlays/dev/kustomization.yaml`

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

bases:
  - ../../base

namespace: ml-deploy-dev

commonLabels:
  environment: dev

resources:
  - namespace.yaml
  - cloud-configmap.yaml
  - storage.yaml

configMapGenerator:
  - name: ml-api-config
    literals:
      - LOG_LEVEL=INFO
      - MODEL_CONFIG_PATH=/app/config/model_config.json
      - ENABLE_AUTH=false
      - API_WORKERS=2
      - MAX_PAYLOAD_SIZE=10MB
  
  - name: ml-training-config
    literals:
      - DATA_DIR=/data
      - OUTPUT_DIR=/models
      - RESOURCE_UTILIZATION=0.8
      - ENABLE_GPU=false

patchesStrategicMerge:
  - ml-api-deployment-patch.yaml
  - ml-api-service-patch.yaml
  - model-training-patch.yaml

images:
  - name: model-api
    newName: ${REGISTRY_URL}/model-api
    newTag: ${API_VERSION}
  - name: model-trainer
    newName: ${REGISTRY_URL}/model-trainer
    newTag: ${TRAINER_VERSION}

vars:
  - name: CLOUD_PROVIDER
    objRef:
      kind: ConfigMap
      name: cloud-config
      apiVersion: v1
    fieldRef:
      fieldPath: data.CLOUD_PROVIDER
```

## 6. Cloud-Specific Config: `kubernetes/overlays/dev/cloud-configmap.yaml`

```yaml
# This ConfigMap provides cloud-provider specific settings
# This is populated by the CI/CD pipeline based on the target deployment environment
apiVersion: v1
kind: ConfigMap
metadata:
  name: cloud-config
data:
  CLOUD_PROVIDER: "aws"  # aws, azure, or gcp
  
  # AWS specific configuration
  AWS_REGION: "us-west-2"
  S3_MODEL_BUCKET: "ml-deploy-dev-model-artifacts"
  S3_LOGS_BUCKET: "ml-deploy-dev-logs"
  
  # Azure specific configuration
  AZURE_REGION: "eastus"
  AZURE_MODEL_CONTAINER: "model-artifacts"
  AZURE_STORAGE_ACCOUNT: "mldeploydevstorage"
  
  # GCP specific configuration
  GCP_REGION: "us-central1"
  GCP_PROJECT_ID: "ml-deploy-dev-project"
  GCS_MODEL_BUCKET: "ml-deploy-dev-model-artifacts"
  
  # Cloud storage path template that works across providers
  # The application code uses this template with the appropriate cloud provider
  # variables injected at runtime
  MODEL_STORAGE_URI_TEMPLATE: >
    aws:s3://${S3_MODEL_BUCKET}/models/;
    azure:https://${AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/${AZURE_MODEL_CONTAINER}/models/;
    gcp:gs://${GCS_MODEL_BUCKET}/models/
```

## 7. CI/CD Complete GitHub Actions: github-actions-workflow.yml

```yaml
name: ML Deploy Pipeline

on:
  push:
    branches: [ main ]
    paths:
      - 'src/**'
      - 'terraform/**'
      - 'deployment/**'
      - 'kubernetes/**'
      - '.github/workflows/**'
  pull_request:
    branches: [ main ]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy to'
        required: true
        default: 'dev'
        type: choice
        options:
          - dev
          -# filepath: c:\Users\adilm\repositories\Go\ml-deploy-pipeline\terraform\ci-cd\github-actions-workflow.yml

name: ML Deploy Pipeline

on:
  push:
    branches: [ main ]
    paths:
      - 'src/**'
      - 'terraform/**'
      - 'deployment/**'
      - 'kubernetes/**'
      - '.github/workflows/**'
  pull_request:
    branches: [ main ]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy to'
        required: true
        default: 'dev'
        type: choice
        options:
          - dev
          -
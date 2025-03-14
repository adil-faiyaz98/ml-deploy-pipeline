# ------------------------------------------------------------------------------
# CLOUD PROVIDER SELECTORION
# ------------------------------------------------------------------------------
provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

module "aws" {
  source = "./aws"
  count  = var.cloud_provider == "aws" ? 1 : 0
  
  # Pass common variables
  project_name        = var.project_name
  environment         = var.environment
  owner               = var.owner
  cost_center         = var.cost_center
  kubernetes_version  = var.kubernetes_version
  min_nodes           = var.min_nodes
  max_nodes           = var.max_nodes
  desired_nodes       = var.desired_nodes
  db_instance_type    = var.db_instance_type
  redis_node_type     = var.redis_node_type
  management_ips      = var.management_ips
  # AWS specific variables
  region              = var.aws_region
  vpc_cidr            = var.aws_vpc_cidr
  eks_admin_roles     = var.aws_eks_admin_roles
}

module "azure" {
  source = "./azure"
  count  = var.cloud_provider == "azure" ? 1 : 0
  
  # Pass common variables
  project_name        = var.project_name
  environment         = var.environment
  owner               = var.owner
  cost_center         = var.cost_center
  kubernetes_version  = var.kubernetes_version
  min_nodes           = var.min_nodes
  max_nodes           = var.max_nodes
  desired_nodes       = var.desired_nodes
  db_instance_type    = var.db_instance_type
  redis_node_type     = var.redis_node_type
  management_ips      = var.management_ips
  # Azure specific variables
  location            = var.azure_location
  subscription_id     = var.azure_subscription_id
  resource_group_name = local.azure_resource_group_name
  admin_group_ids     = var.azure_admin_group_ids
}

module "gcp" {
  source = "./gcp"
  count  = var.cloud_provider == "gcp" ? 1 : 0
  
  # Pass common variables
  project_name        = var.project_name
  environment         = var.environment
  owner               = var.owner
  cost_center         = var.cost_center
  kubernetes_version  = var.kubernetes_version
  min_nodes           = var.min_nodes
  max_nodes           = var.max_nodes
  desired_nodes       = var.desired_nodes
  db_instance_type    = var.db_instance_type
  redis_node_type     = var.redis_node_type
  management_ips      = var.management_ips
  # GCP specific variables
  project_id          = var.gcp_project_id
  region              = var.gcp_region
  zone                = var.gcp_zone
  network_name        = var.gcp_network_name
  subnetwork_name     = var.gcp_subnetwork_name
}

# ------------------------------------------------------------------------------
# LOCAL VARIABLES
# ------------------------------------------------------------------------------

locals {
  name_prefix = "${var.project_name}-${var.environment}"
  location    = var.region
  
  common_labels = {
    project     = var.project_name
    environment = var.environment
    terraform   = "true"
    owner       = var.owner
    cost_center = var.cost_center
  }
  
  # Machine type mappings
  machine_types = {
    "small"  = "n2-standard-2"
    "medium" = "n2-standard-4"
    "large"  = "n2-standard-8"
    "xlarge" = "n2-standard-16"
    "gpu-small" = "n1-standard-4"
    "gpu-medium" = "n1-standard-8"
    "gpu-large" = "n1-standard-16"
  }
  
  # This provides a deterministic name for Azure resource groups
  azure_resource_group_name = "${var.project_name}-${var.environment}-rg"
  
  # Extract cluster endpoint depending on cloud provider
  cluster_endpoint = (
    var.cloud_provider == "aws" ? module.aws[0].cluster_endpoint :
    var.cloud_provider == "azure" ? module.azure[0].cluster_endpoint :
    var.cloud_provider == "gcp" ? module.gcp[0].cluster_endpoint :
    null
  )
  
  # Extract kubeconfig depending on cloud provider
  kubeconfig = (
    var.cloud_provider == "aws" ? module.aws[0].kubeconfig :
    var.cloud_provider == "azure" ? module.azure[0].kubeconfig :
    var.cloud_provider == "gcp" ? module.gcp[0].kubeconfig :
    null
  )
}

# ------------------------------------------------------------------------------
# VPC NETWORK
# ------------------------------------------------------------------------------
resource "google_compute_network" "vpc" {
  name                    = "${local.name_prefix}-vpc"
  auto_create_subnetworks = false
  description             = "VPC network for ${var.project_name} ${var.environment}"
}

resource "google_compute_subnetwork" "private" {
  name                     = "${local.name_prefix}-private"
  ip_cidr_range            = var.private_subnet_cidr
  region                   = var.region
  network                  = google_compute_network.vpc.id
  private_ip_google_access = true
  
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.pods_ip_range
  }
  
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.services_ip_range
  }
  
  log_config {
    aggregation_interval = "INTERVAL_5_MIN"
    flow_sampling        = 0.5
    metadata             = "INCLUDE_ALL_METADATA"
  }
}

# ------------------------------------------------------------------------------
# GKE CLUSTER
# ------------------------------------------------------------------------------
resource "google_container_cluster" "primary" {
  name     = "${local.name_prefix}-gke"
  location = var.regional_cluster ? var.region : "${var.region}-a"
  
  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1
  
  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.private.name
  
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }
  
  # Enable Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  # Enable network policy
  network_policy {
    enabled  = true
    provider = "CALICO"
  }
  
  # Enable Shielded Nodes
  release_channel {
    channel = "REGULAR"
  }
  
  # Minimum GKE version
  min_master_version = var.kubernetes_version
  
  # Setup private cluster
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = var.master_ipv4_cidr_block
  }
  
  # Binary authorization
  binary_authorization {
    evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  }
  
  # Node config defaults
  node_config {
    # Google recommends custom service accounts with minimal permissions
    service_account = google_service_account.gke_sa.email
    
    # Use COS for better security
    image_type = "COS_CONTAINERD"
    
    # Enable Workload Identity on all nodes
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    # Enable Secure Boot for nodes
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }
  
  maintenance_policy {
    recurring_window {
      start_time = "2022-01-01T00:00:00Z"
      end_time   = "2022-01-01T04:00:00Z"
      recurrence = "FREQ=WEEKLY;BYDAY=SA"
    }
  }
  
  # Enable VPA
  vertical_pod_autoscaling {
    enabled = true
  }
  
  # Monitoring
  logging_service    = "logging.googleapis.com/kubernetes"
  monitoring_service = "monitoring.googleapis.com/kubernetes"
}

# ------------------------------------------------------------------------------
# NODE POOLS
# ------------------------------------------------------------------------------
resource "google_container_node_pool" "ml_compute" {
  name       = "ml-compute"
  location   = google_container_cluster.primary.location
  cluster    = google_container_cluster.primary.name
  node_count = var.desired_nodes
  
  autoscaling {
    min_node_count = var.min_nodes
    max_node_count = var.max_nodes
  }
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  node_config {
    preemptible  = var.environment != "production"
    machine_type = local.machine_types["large"]
    
    labels = {
      workload-type = "ml-training"
    }
    
    service_account = google_service_account.gke_sa.email
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    # Specify disk settings
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    
    # Enable Workload Identity on all nodes
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    # Shielded nodes
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }
}

resource "google_container_node_pool" "api_serving" {
  name       = "api-serving"
  location   = google_container_cluster.primary.location
  cluster    = google_container_cluster.primary.name
  node_count = 2
  
  autoscaling {
    min_node_count = 2
    max_node_count = 10
  }
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  node_config {
    preemptible  = false
    machine_type = local.machine_types["medium"]
    
    labels = {
      workload-type = "model-serving"
    }
    
    service_account = google_service_account.gke_sa.email
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    # Specify disk settings
    disk_size_gb = 50
    disk_type    = "pd-ssd"
    
    # Enable Workload Identity on all nodes
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    # Shielded nodes
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }
}

resource "google_container_node_pool" "monitoring" {
  name       = "monitoring"
  location   = google_container_cluster.primary.location
  cluster    = google_container_cluster.primary.name
  node_count = 1
  
  autoscaling {
    min_node_count = 1
    max_node_count = 3
  }
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  node_config {
    preemptible  = false
    machine_type = local.machine_types["small"]
    
    labels = {
      workload-type = "monitoring"
    }
    
    service_account = google_service_account.gke_sa.email
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    # Specify disk settings
    disk_size_gb = 50
    disk_type    = "pd-standard"
    
    # Enable Workload Identity on all nodes
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}

# ------------------------------------------------------------------------------
# SERVICE ACCOUNTS
# ------------------------------------------------------------------------------
resource "google_service_account" "gke_sa" {
  account_id   = "${local.name_prefix}-gke-sa"
  display_name = "GKE Service Account for ${var.project_name} ${var.environment}"
}

resource "google_project_iam_member" "gke_sa_roles" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/artifactregistry.reader"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_sa.email}"
}

# ------------------------------------------------------------------------------
# CLOUD SQL
# ------------------------------------------------------------------------------
resource "google_sql_database_instance" "postgres" {
  name             = "${local.name_prefix}-postgres"
  database_version = "POSTGRES_14"
  region           = var.region
  
  settings {
    tier = var.db_instance_type
    
    availability_type = var.environment == "production" ? "REGIONAL" : "ZONAL"
    
    backup_configuration {
      enabled            = true
      binary_log_enabled = false
      start_time         = "02:00"
      
      backup_retention_settings {
        retained_backups = var.environment == "production" ? 30 : 7
        retention_unit   = "COUNT"
      }
    }
    
    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = true
    }
    
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc.id
      
      # Allow access from the GKE control plane
      authorized_networks {
        name  = "GKE Control Plane"
        value = var.master_ipv4_cidr_block
      }
    }
    
    database_flags {
      name  = "max_connections"
      value = "500"
    }
    
    database_flags {
      name  = "shared_buffers"
      value = "4096MB"
    }
    
    maintenance_window {
      day          = 7  # Sunday
      hour         = 2  # 2 AM
      update_track = "stable"
    }
  }
  
  deletion_protection = var.environment == "production"
}

resource "google_sql_database" "mlmodels" {
  name     = "mlmodels"
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_user" "postgres_user" {
  name     = "postgres"
  instance = google_sql_database_instance.postgres.name
  password = random_password.db_password.result
}

# ------------------------------------------------------------------------------
# REDIS
# ------------------------------------------------------------------------------
resource "google_redis_instance" "cache" {
  name           = "${local.name_prefix}-redis"
  tier           = var.environment == "production" ? "STANDARD_HA" : "BASIC"
  memory_size_gb = var.environment == "production" ? 16 : 4
  
  region              = var.region
  authorized_network  = google_compute_network.vpc.id
  connect_mode        = "PRIVATE_SERVICE_ACCESS"
  transit_encryption_mode = "SERVER_AUTHENTICATION"
  
  redis_version      = "REDIS_6_X"
  display_name       = "ML Pipeline Redis Cache"
  redis_configs = {
    "maxmemory-policy" = "allkeys-lru"
  }
  
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 2
        minutes = 0
      }
    }
  }
}

# ------------------------------------------------------------------------------
# CLOUD STORAGE
# ------------------------------------------------------------------------------
resource "google_storage_bucket" "model_artifacts" {
  name     = "${local.name_prefix}-model-artifacts"
  location = var.region
  
  uniform_bucket_level_access = true
  force_destroy               = var.environment != "production"
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }
}

# ------------------------------------------------------------------------------
# SECRETS
# ------------------------------------------------------------------------------
resource "random_password" "db_password" {
  length           = 16
  special          = true
  override_special = "!#$%&*-_=+:?"
}

resource "google_secret_manager_secret" "db_password" {
  secret_id = "${local.name_prefix}-db-password"
  
  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "db_password" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = random_password.db_password.result
}

resource "random_password" "redis_password" {
  length           = 16
  special          = true
  override_special = "!#$%&*-_=+:?"
}

resource "google_secret_manager_secret" "redis_password" {
  secret_id = "${local.name_prefix}-redis-password"
  
  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "redis_password" {
  secret      = google_secret_manager_secret.redis_password.id
  secret_data = random_password.redis_password.result
}

# ------------------------------------------------------------------------------
# ARTIFACT REGISTRY
# ------------------------------------------------------------------------------
resource "google_artifact_registry_repository" "ml_registry" {
  location      = var.region
  repository_id = "${local.name_prefix}-registry"
  description   = "ML model container registry"
  format        = "DOCKER"
}

# ------------------------------------------------------------------------------
# IAM FOR WORKLOAD IDENTITY
# ------------------------------------------------------------------------------
resource "google_service_account" "ml_api_sa" {
  account_id   = "${local.name_prefix}-ml-api-sa"
  display_name = "ML API Service Account"
}

resource "google_project_iam_member" "ml_api_roles" {
  for_each = toset([
    "roles/secretmanager.secretAccessor",
    "roles/storage.objectUser",
    "roles/monitoring.metricWriter"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.ml_api_sa.email}"
}

resource "google_service_account_iam_binding" "ml_api_workload_identity" {
  service_account_id = google_service_account.ml_api_sa.name
  role               = "roles/iam.workloadIdentityUser"
  
  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[default/model-api]"
  ]
}

# ------------------------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------------------------
output "gke_cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.primary.name
}

output "gke_cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.primary.endpoint
}

output "postgres_connection_name" {
  description = "Cloud SQL connection name"
  value       = google_sql_database_instance.postgres.connection_name
}

output "postgres_private_ip" {
  description = "Cloud SQL private IP"
  value       = google_sql_database_instance.postgres.private_ip_address
}

output "redis_host" {
  description = "Redis hostname"
  value       = google_redis_instance.cache.host
}

output "model_artifacts_bucket" {
  description = "GCS bucket for model artifacts"
  value       = google_storage_bucket.model_artifacts.name
}

output "ml_api_service_account" {
  description = "Service account for ML API"
  value       = google_service_account.ml_api_sa.email
}

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
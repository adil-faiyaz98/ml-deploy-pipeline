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
  
  # Database instance mappings
  db_instance_types = {
    "small"  = "db-g1-small"
    "medium" = "db-custom-2-4096"
    "large"  = "db-custom-4-8192"
    "xlarge" = "db-custom-8-16384"
  }
  
  # Redis tier mappings
  redis_tiers = {
    "small"  = "BASIC"
    "medium" = "STANDARD_HA"
    "large"  = "STANDARD_HA"
  }
  
  # GPU types and counts
  gpu_types = {
    "gpu-small"  = "nvidia-tesla-t4"
    "gpu-medium" = "nvidia-tesla-t4"
    "gpu-large"  = "nvidia-tesla-v100"
  }
  
  gpu_counts = {
    "gpu-small"  = 1
    "gpu-medium" = 2
    "gpu-large"  = 4
  }
  
  # Network settings
  network_name = "${local.name_prefix}-network"
  subnet_name  = "${local.name_prefix}-subnet"
  subnet_cidr  = "10.0.0.0/16"
  
  # Node pools configuration
  node_pools = [
    {
      name               = "ml-compute"
      machine_type       = var.environment == "production" ? "n2-standard-8" : "n2-standard-4"
      node_count         = var.desired_nodes
      min_count          = var.min_nodes
      max_count          = var.max_nodes
      disk_size_gb       = 100
      disk_type          = "pd-ssd"
      auto_repair        = true
      auto_upgrade       = true
      preemptible        = var.environment != "production"
      accelerator_type   = var.environment == "production" ? "nvidia-tesla-t4" : null
      accelerator_count  = var.environment == "production" ? 1 : 0
      labels             = { "workload-type" = "ml-training" }
      taints             = []
    },
    {
      name               = "api-serving"
      machine_type       = var.environment == "production" ? "n2-standard-4" : "n2-standard-2"
      node_count         = 2
      min_count          = 2
      max_count          = 10
      disk_size_gb       = 50
      disk_type          = "pd-ssd"
      auto_repair        = true
      auto_upgrade       = true
      preemptible        = false
      accelerator_type   = null
      accelerator_count  = 0
      labels             = { "workload-type" = "model-serving" }
      taints             = []
    },
    {
      name               = "monitoring"
      machine_type       = "n2-standard-2"
      node_count         = 1
      min_count          = 1
      max_count          = 3
      disk_size_gb       = 50
      disk_type          = "pd-standard"
      auto_repair        = true
      auto_upgrade       = true
      preemptible        = var.environment != "production"
      accelerator_type   = null
      accelerator_count  = 0
      labels             = { "workload-type" = "monitoring" }
      taints             = []
    }
  ]
}

# ------------------------------------------------------------------------------
# VPC & NETWORK INFRASTRUCTURE
# ------------------------------------------------------------------------------
resource "google_compute_network" "vpc_network" {
  name                    = local.network_name
  auto_create_subnetworks = false
  routing_mode            = "GLOBAL"
}

resource "google_compute_subnetwork" "subnetwork" {
  name                     = local.subnet_name
  ip_cidr_range            = local.subnet_cidr
  region                   = var.region
  network                  = google_compute_network.vpc_network.id
  private_ip_google_access = true
  
  log_config {
    aggregation_interval = "INTERVAL_5_SEC"
    flow_sampling        = 0.5
    metadata             = "INCLUDE_ALL_METADATA"
  }
}

resource "google_compute_router" "router" {
  name    = "${local.name_prefix}-router"
  region  = var.region
  network = google_compute_network.vpc_network.id
}

resource "google_compute_router_nat" "nat" {
  name                               = "${local.name_prefix}-nat"
  router                             = google_compute_router.router.name
  region                             = var.region
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
  location = var.region
  
  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1
  
  network    = google_compute_network.vpc_network.id
  subnetwork = google_compute_subnetwork.subnetwork.id
  
  release_channel {
    channel = var.environment == "production" ? "REGULAR" : "RAPID"
  }
  
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }
  
  ip_allocation_policy {
    cluster_ipv4_cidr_block  = "/16"
    services_ipv4_cidr_block = "/22"
  }
  
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }
  
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }
  
  logging_service    = "logging.googleapis.com/kubernetes"
  monitoring_service = "monitoring.googleapis.com/kubernetes"
  
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
  
  network_policy {
    enabled = true
    provider = "CALICO"
  }
  
  timeouts {
    create = "30m"
    update = "40m"
  }
}

resource "google_container_node_pool" "node_pools" {
  for_each = { for i, pool in local.node_pools : pool.name => pool }
  
  name       = each.value.name
  location   = var.region
  cluster    = google_container_cluster.primary.name
  node_count = each.value.node_count
  
  autoscaling {
    min_node_count = each.value.min_count
    max_node_count = each.value.max_count
  }
  
  management {
    auto_repair  = each.value.auto_repair
    auto_upgrade = each.value.auto_upgrade
  }
  
  node_config {
    preemptible  = each.value.preemptible
    machine_type = each.value.machine_type
    disk_size_gb = each.value.disk_size_gb
    disk_type    = each.value.disk_type
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    labels = each.value.labels
    
    dynamic "taint" {
      for_each = each.value.taints
      content {
        key    = taint.value.key
        value  = taint.value.value
        effect = taint.value.effect
      }
    }
    
    dynamic "guest_accelerator" {
      for_each = each.value.accelerator_count > 0 ? [1] : []
      content {
        type  = each.value.accelerator_type
        count = each.value.accelerator_count
      }
    }
  }
}

# ------------------------------------------------------------------------------
# CLOUD SQL (PostgreSQL)
# ------------------------------------------------------------------------------
resource "google_sql_database_instance" "postgres" {
  name             = "${local.name_prefix}-postgres"
  database_version = "POSTGRES_14"
  region           = var.region
  
  deletion_protection = var.environment == "production"
  
  settings {
    tier              = local.db_instance_types[var.db_instance_type]
    availability_type = var.environment == "production" ? "REGIONAL" : "ZONAL"
    
    disk_size = 100
    disk_type = "PD_SSD"
    disk_autoresize = true
    disk_autoresize_limit = 500
    
    backup_configuration {
      enabled            = true
      start_time         = "02:00"
      binary_log_enabled = false
      
      backup_retention_settings {
        retained_backups = var.environment == "production" ? 30 : 7
        retention_unit   = "COUNT"
      }
    }
    
    maintenance_window {
      day          = 1  # Monday
      hour         = 3
      update_track = "stable"
    }
    
    insights_config {
      query_insights_enabled  = true
      query_string_length     = 4096
      record_application_tags = true
      record_client_address   = true
    }
    
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc_network.id
      require_ssl     = true
    }
    
    database_flags {
      name  = "max_connections"
      value = "500"
    }
    
    database_flags {
      name  = "shared_buffers"
      value = "4096MB"
    }
    
    database_flags {
      name  = "work_mem"
      value = "64MB"
    }
  }
  
  depends_on = [google_service_networking_connection.private_vpc_connection]
}

resource "google_sql_database" "mlflow_db" {
  name     = "mlmodels"
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_user" "postgres_user" {
  name     = "postgres"
  instance = google_sql_database_instance.postgres.name
  password = random_password.postgres_password.result
}

resource "random_password" "postgres_password" {
  length           = 16
  special          = true
  override_special = "_%@"
}

resource "google_compute_global_address" "private_ip_address" {
  name          = "${local.name_prefix}-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc_network.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.vpc_network.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}

# ------------------------------------------------------------------------------
# REDIS
# ------------------------------------------------------------------------------
resource "google_redis_instance" "cache" {
  name           = "${local.name_prefix}-redis"
  tier           = local.redis_tiers[var.redis_node_type]
  memory_size_gb = 4
  location_id    = var.region
  
  redis_version      = "REDIS_6_X"
  redis_configs      = {}
  display_name       = "ML Model API Cache"
  auth_enabled       = true
  transit_encryption_mode = "SERVER_AUTHENTICATION"
  
  authorized_network = google_compute_network.vpc_network.id
  
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
# GCS BUCKETS
# ------------------------------------------------------------------------------
resource "google_storage_bucket" "model_artifacts" {
  name                        = "${local.name_prefix}-model-artifacts"
  location                    = var.region
  storage_class               = "STANDARD"
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
  
  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }
  
  dynamic "lifecycle_rule" {
    for_each = var.environment == "production" ? [] : [1]
    content {
      condition {
        age = 30
      }
      action {
        type = "Delete"
      }
    }
  }
  
  labels = local.common_labels
}

resource "google_storage_bucket_iam_binding" "model_artifacts_admin" {
  bucket = google_storage_bucket.model_artifacts.name
  role   = "roles/storage.admin"
  members = [
    "serviceAccount:${google_service_account.gke_sa.email}"
  ]
}

# ------------------------------------------------------------------------------
# IAM & SERVICE ACCOUNTS
# ------------------------------------------------------------------------------
resource "google_service_account" "gke_sa" {
  account_id   = "${local.name_prefix}-gke-sa"
  display_name = "GKE Service Account for ML workloads"
}

resource "google_project_iam_member" "gke_sa_roles" {
  for_each = toset([
    "roles/storage.admin",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/artifactregistry.reader",
    "roles/cloudsql.client"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_sa.email}"
}

# ------------------------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------------------------
output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.primary.name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "db_instance_name" {
  description = "Cloud SQL instance name"
  value       = google_sql_database_instance.postgres.name
}

output "db_name" {
  description = "Database name"
  value       = google_sql_database.mlflow_db.name
}

output "db_user" {
  description = "Database user"
  value       = google_sql_user.postgres_user.name
  sensitive   = true
}

output "db_password" {
  description = "Database password"
  value       = google_sql_user.postgres_user.password
  sensitive   = true
}

output "redis_host" {
  description = "Redis hostname"
  value       = google_redis_instance.cache.host
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
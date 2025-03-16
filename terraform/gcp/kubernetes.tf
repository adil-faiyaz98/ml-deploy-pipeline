# ------------------------------------------------------------------------------
# GKE CLUSTER
# ------------------------------------------------------------------------------
resource "google_container_cluster" "primary" {
  name     = "${local.name_prefix}-gke"
  location = var.region
  
  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1
  
  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.private.name
  
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = var.environment == "production"
    master_ipv4_cidr_block  = "172.16.0.0/28"
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
  
  # Enable master authorized networks
  master_authorized_networks_config {
    dynamic "cidr_blocks" {
      for_each = var.authorized_networks
      content {
        cidr_block   = cidr_blocks.value.cidr_block
        display_name = cidr_blocks.value.display_name
      }
    }
  }
  
  # Enable Kubernetes Binary Authorization
  binary_authorization {
    evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  }
  
  # Enable shielded nodes
  release_channel {
    channel = "REGULAR"
  }
  
  # Enable VPA
  vertical_pod_autoscaling {
    enabled = true
  }
  
  # IP allocation policy for VPC-native
  ip_allocation_policy {
    cluster_ipv4_cidr_block  = "/14"
    services_ipv4_cidr_block = "/20"
  }
  
  # Maintenance policy
  maintenance_policy {
    recurring_window {
      start_time = "2022-01-01T00:00:00Z"
      end_time   = "2022-01-02T03:00:00Z"
      recurrence = "FREQ=WEEKLY;BYDAY=SU"
    }
  }
  
  # Enable Dataplane V2 (GKE Dataplane V2 with Cilium)
  datapath_provider = "ADVANCED_DATAPATH"
  
  addons_config {
    horizontal_pod_autoscaling {
      disabled = false
    }
    http_load_balancing {
      disabled = false
    }
    gcp_filestore_csi_driver_config {
      enabled = true
    }
    config_connector_config {
      enabled = true
    }
  }
}

# ------------------------------------------------------------------------------
# NODE POOLS
# ------------------------------------------------------------------------------

# Node Pool for ML Training
resource "google_container_node_pool" "ml_compute" {
  name       = "ml-compute-pool"
  location   = var.region
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
    machine_type = local.vm_sizes["large"]
    
    # Google recommended minimum of 1:8 for ML workloads
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    
    # Enable workload identity on the node pool
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    # Enable secure boot for GKE nodes
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
    
    labels = {
      workload = "ml-training"
    }
    
    taint {
      key    = "workload"
      value  = "ml-training"
      effect = "NO_SCHEDULE"
    }
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}

# Node Pool for API Serving
resource "google_container_node_pool" "api_serving" {
  name       = "api-serving-pool"
  location   = var.region
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
    machine_type = local.vm_sizes["medium"]
    
    disk_size_gb = 50
    disk_type    = "pd-ssd"
    
    # Enable workload identity on the node pool
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    # Enable secure boot for GKE nodes
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
    
    labels = {
      workload = "model-serving"
    }
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}

# Node Pool for Monitoring
resource "google_container_node_pool" "monitoring" {
  name       = "monitoring-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  node_count = 1
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  node_config {
    preemptible  = false
    machine_type = local.vm_sizes["small"]
    
    disk_size_gb = 50
    disk_type    = "pd-standard"
    
    # Enable workload identity on the node pool
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    labels = {
      workload = "monitoring"
    }
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}

# ------------------------------------------------------------------------------
# KUBERNETES PROVIDER CONFIGURATION
# ------------------------------------------------------------------------------
provider "kubernetes" {
  host                   = "https://${google_container_cluster.primary.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth.0.cluster_ca_certificate)
}

provider "helm" {
  kubernetes {
    host                   = "https://${google_container_cluster.primary.endpoint}"
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth.0.cluster_ca_certificate)
  }
}

data "google_client_config" "default" {}
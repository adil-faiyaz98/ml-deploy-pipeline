# ------------------------------------------------------------------------------
# CLOUD PROVIDER SELECTOR
# ------------------------------------------------------------------------------

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
# OUTPUT SELECTOR
# ------------------------------------------------------------------------------

locals {
  # Helper logic to select the appropriate module output based on chosen cloud provider
  cloud_outputs = {
    aws   = module.aws[0]
    azure = module.azure[0]
    gcp   = module.gcp[0]
  }
  
  # Define a local variable for the Azure resource group name
  azure_resource_group_name = "${var.project_name}-${var.environment}-rg"
}

# Cloud-agnostic outputs that will work regardless of chosen provider
output "kubernetes_cluster_name" {
  description = "Kubernetes cluster name"
  value       = var.cloud_provider == "aws" ? module.aws[0].kubernetes_cluster_name : (
                var.cloud_provider == "azure" ? module.azure[0].kubernetes_cluster_name : 
                module.gcp[0].kubernetes_cluster_name)
}

output "kubernetes_endpoint" {
  description = "Kubernetes cluster endpoint"
  value       = var.cloud_provider == "aws" ? module.aws[0].kubernetes_endpoint : (
                var.cloud_provider == "azure" ? module.azure[0].kubernetes_endpoint : 
                module.gcp[0].kubernetes_endpoint)
  sensitive   = true
}

output "database_endpoint" {
  description = "Database endpoint"
  value       = var.cloud_provider == "aws" ? module.aws[0].database_endpoint : (
                var.cloud_provider == "azure" ? module.azure[0].database_endpoint : 
                module.gcp[0].database_endpoint)
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis endpoint"
  value       = var.cloud_provider == "aws" ? module.aws[0].redis_endpoint : (
                var.cloud_provider == "azure" ? module.azure[0].redis_endpoint : 
                module.gcp[0].redis_endpoint)
  sensitive   = true
}

output "storage_bucket" {
  description = "Storage bucket for model artifacts"
  value       = var.cloud_provider == "aws" ? module.aws[0].storage_bucket : (
                var.cloud_provider == "azure" ? module.azure[0].storage_account : 
                module.gcp[0].storage_bucket)
}

output "ingress_ip" {
  description = "Ingress controller IP address/hostname"
  value       = var.cloud_provider == "aws" ? module.aws[0].ingress_hostname : (
                var.cloud_provider == "azure" ? module.azure[0].ingress_ip : 
                module.gcp[0].ingress_ip)
}
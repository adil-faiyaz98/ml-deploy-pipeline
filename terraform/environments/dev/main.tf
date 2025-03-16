# ------------------------------------------------------------------------------
# DEV ENVIRONMENT CONFIGURATION
# ------------------------------------------------------------------------------

# Use specific versions for development environment
provider "aws" {
  region = var.aws_region
}

provider "azurerm" {
  features {}
}

provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
}

module "ml_infrastructure" {
  source = "../../"
  
  # Common variables
  cloud_provider = var.cloud_provider
  project_name   = var.project_name
  environment    = "dev"
  owner          = var.owner
  cost_center    = var.cost_center
  
  # Development-specific variables
  min_nodes      = 2
  max_nodes      = 4
  desired_nodes  = 2
  
  # Database variables
  db_instance_type = "small"
  redis_node_type  = "small"
  
  # Network variables
  management_ips   = var.management_ips
  
  # AWS specific
  aws_region       = var.aws_region
  aws_vpc_cidr     = "10.0.0.0/16"
  
  # Azure specific
  azure_location       = var.azure_location
  azure_subscription_id = var.azure_subscription_id
  
  # GCP specific
  gcp_project = var.gcp_project
  gcp_region  = var.gcp_region
}

# Output the endpoints
output "model_api_endpoint" {
  description = "Endpoint for model API"
  value       = module.ml_infrastructure.model_api_endpoint
}

output "mlflow_endpoint" {
  description = "Endpoint for MLflow"
  value       = module.ml_infrastructure.mlflow_endpoint
}
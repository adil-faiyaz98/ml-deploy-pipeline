# This module manages multi-region deployments for high availability

variable "primary_region" {
  description = "Primary region for deployment"
  type        = string
}

variable "secondary_regions" {
  description = "List of secondary regions for failover"
  type        = list(string)
  default     = []
}

variable "cloud_provider" {
  description = "Cloud provider (aws, azure, gcp)"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "project_name" {
  description = "Project name"
  type        = string
}

locals {
  is_multi_region = length(var.secondary_regions) > 0
  
  # Only enable multi-region for production by default
  enable_multi_region = var.environment == "production" && local.is_multi_region
  
  # Region specific configurations
  name_prefix = "${var.project_name}-${var.environment}"
  
  # DNS failover configuration based on cloud provider
  dns_failover_config = {
    aws   = "route53"
    azure = "traffic-manager"
    gcp   = "cloud-dns"
  }
}

# AWS Implementation
module "aws_multi_region" {
  count  = var.cloud_provider == "aws" && local.enable_multi_region ? 1 : 0
  source = "./aws"
  
  primary_region    = var.primary_region
  secondary_regions = var.secondary_regions
  name_prefix       = local.name_prefix
  
  # Enable Route 53 health checks and failover routing
  enable_health_checks = true
  health_check_path    = "/health"
  health_check_port    = 80
  
  # Primary region resources (would be passed in a real module)
  # primary_lb_arn = var.primary_lb_arn
  # primary_db_arn = var.primary_db_arn
}

# Azure Implementation
module "azure_multi_region" {
  count  = var.cloud_provider == "azure" && local.enable_multi_region ? 1 : 0
  source = "./azure"
  
  primary_region    = var.primary_region
  secondary_regions = var.secondary_regions
  name_prefix       = local.name_prefix
  
  # Enable Traffic Manager for global routing
  traffic_manager_profile_name = "${local.name_prefix}-tm"
  health_check_path            = "/health"
  health_check_port            = 80
  
  # Primary region resources (would be passed in a real module)
  # primary_app_service_id = var.primary_app_service_id
  # primary_db_id         = var.primary_db_id
}

# GCP Implementation
module "gcp_multi_region" {
  count  = var.cloud_provider == "gcp" && local.enable_multi_region ? 1 : 0
  source = "./gcp"
  
  primary_region    = var.primary_region
  secondary_regions = var.secondary_regions
  name_prefix       = local.name_prefix
  
  # Enable Cloud DNS and global load balancing
  enable_global_lb = true
  health_check_path = "/health"
  health_check_port = 80
  
  # Primary region resources (would be passed in a real module)
  # primary_lb_name = var.primary_lb_name
  # primary_db_name = var.primary_db_name
}

output "endpoint" {
  description = "Global endpoint for the multi-region deployment"
  value = var.cloud_provider == "aws" && local.enable_multi_region ? module.aws_multi_region[0].endpoint :
         var.cloud_provider == "azure" && local.enable_multi_region ? module.azure_multi_region[0].endpoint :
         var.cloud_provider == "gcp" && local.enable_multi_region ? module.gcp_multi_region[0].endpoint :
         null
}
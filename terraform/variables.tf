# ------------------------------------------------------------------------------
# COMMON VARIABLES
# ------------------------------------------------------------------------------

variable "cloud_provider" {
  description = "Cloud provider to use (aws, azure, gcp)"
  type        = string
  default     = "aws"
  
  validation {
    condition     = contains(["aws", "azure", "gcp"], var.cloud_provider)
    error_message = "Valid values for cloud_provider are: aws, azure, gcp"
  }
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "ml-deploy"
}

variable "environment" {
  description = "Environment (dev, staging, production)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production"
  }
}

variable "owner" {
  description = "Owner of the resources"
  type        = string
  default     = "data-science-team"
}

variable "cost_center" {
  description = "Cost center for billing"
  type        = string
  default     = "ml-research"
}

variable "kubernetes_version" {
  description = "Kubernetes version for cluster"
  type        = string
  default     = "1.28"
}

variable "min_nodes" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 2
}

variable "max_nodes" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 8
}

variable "desired_nodes" {
  description = "Desired number of worker nodes"
  type        = number
  default     = 3
}

variable "db_instance_type" {
  description = "Database instance type"
  type        = string
  default     = "medium"  # Will be mapped to provider-specific instance types
}

variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "medium"  # Will be mapped to provider-specific instance types
}

variable "management_ips" {
  description = "List of CIDR blocks that should have management access"
  type        = list(string)
  default     = []
}

variable "deploy_monitoring" {
  description = "Whether to deploy the monitoring stack"
  type        = bool
  default     = true
}

variable "deploy_mlflow" {
  description = "Whether to deploy MLflow"
  type        = bool
  default     = true
}

variable "deploy_elastic_stack" {
  description = "Whether to deploy the ELK stack"
  type        = bool
  default     = false
}

variable "grafana_admin_password" {
  description = "Password for Grafana admin user"
  type        = string
  default     = "admin"
  sensitive   = true
}

# ------------------------------------------------------------------------------
# AWS SPECIFIC VARIABLES
# ------------------------------------------------------------------------------

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "aws_vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "aws_eks_admin_roles" {
  description = "List of IAM roles with admin access to the EKS cluster"
  type        = list(string)
  default     = []
}

# ------------------------------------------------------------------------------
# AZURE SPECIFIC VARIABLES
# ------------------------------------------------------------------------------

variable "azure_location" {
  description = "Azure region"
  type        = string
  default     = "eastus"
}

variable "azure_subscription_id" {
  description = "Azure Subscription ID"
  type        = string
  default     = null
}

variable "azure_resource_prefix" {
  description = "Prefix for Azure resource names"
  type        = string
  default     = "mlops"
}

# ------------------------------------------------------------------------------
# GCP SPECIFIC VARIABLES
# ------------------------------------------------------------------------------

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "gcp_project_id" {
  description = "GCP Project ID"
  type        = string
  default     = null
}
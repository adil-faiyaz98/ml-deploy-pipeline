# ------------------------------------------------------------------------------
# COMMON VARIABLES (CLOUD-AGNOSTIC)
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# KUBERNETES VARIABLES
# ------------------------------------------------------------------------------
variable "kubernetes_version" {
  description = "Kubernetes version for cluster"
  type        = string
  default     = "1.28"
}

variable "min_nodes" {
  description = "Minimum number of cluster nodes"
  type        = number
  default     = 2
}

variable "max_nodes" {
  description = "Maximum number of cluster nodes"
  type        = number
  default     = 8
}

variable "desired_nodes" {
  description = "Desired number of cluster nodes"
  type        = number
  default     = 3
}

# ------------------------------------------------------------------------------
# SERVICE OPTIONS
# ------------------------------------------------------------------------------
variable "deploy_mlflow" {
  description = "Whether to deploy MLflow"
  type        = bool
  default     = true
}

variable "deploy_monitoring" {
  description = "Whether to deploy the monitoring stack"
  type        = bool
  default     = true
}

variable "deploy_elastic_stack" {
  description = "Whether to deploy the ELK stack"
  type        = bool
  default     = true
}

# ------------------------------------------------------------------------------
# MONITORING CONFIGURATION 
# ------------------------------------------------------------------------------
variable "grafana_admin_password" {
  description = "Password for Grafana admin user"
  type        = string
  default     = "admin"
  sensitive   = true
}

# ------------------------------------------------------------------------------
# CLOUD SELECTION
# ------------------------------------------------------------------------------
variable "cloud_provider" {
  description = "Cloud provider to use (aws, azure, gcp)"
  type        = string
  default     = "aws"
  
  validation {
    condition     = contains(["aws", "azure", "gcp"], var.cloud_provider)
    error_message = "Cloud provider must be one of: aws, azure, gcp"
  }
}
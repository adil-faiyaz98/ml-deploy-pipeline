# ------------------------------------------------------------------------------
# INPUT VARIABLES
# ------------------------------------------------------------------------------

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "vpc_cidr" {
  description = "CIDR range for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "pods_cidr" {
  description = "Secondary CIDR range for pods"
  type        = string
  default     = "10.1.0.0/16"
}

variable "services_cidr" {
  description = "Secondary CIDR range for services"
  type        = string
  default     = "10.2.0.0/16"
}

variable "master_cidr" {
  description = "CIDR range for GKE master"
  type        = string
  default     = "172.16.0.0/28"
}

# Pass-through variables from root module
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
  description = "Database instance type (small, medium, large, xlarge)"
  type        = string
  default     = "small"
}

variable "redis_node_type" {
  description = "Redis instance type (small, medium, large)"
  type        = string
  default     = "small"
}

variable "management_ips" {
  description = "List of CIDR blocks that should have management access"
  type        = list(string)
  default     = []
}
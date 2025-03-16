# ------------------------------------------------------------------------------
# REQUIRED PARAMETERS
# ------------------------------------------------------------------------------

variable "region" {
  description = "AWS region"
  type        = string
}

# ------------------------------------------------------------------------------
# COMMON PARAMETERS
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

# ------------------------------------------------------------------------------
# KUBERNETES PARAMETERS
# ------------------------------------------------------------------------------

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
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
  default     = 10
}

variable "desired_nodes" {
  description = "Desired number of worker nodes"
  type        = number
  default     = 3
}

# ------------------------------------------------------------------------------
# DATABASE PARAMETERS
# ------------------------------------------------------------------------------

variable "db_instance_type" {
  description = "Size of the database instance (small, medium, large, xlarge)"
  type        = string
  default     = "small"
  
  validation {
    condition     = contains(["small", "medium", "large", "xlarge"], var.db_instance_type)
    error_message = "Valid values for db_instance_type: small, medium, large, xlarge"
  }
}

variable "redis_node_type" {
  description = "Size of the Redis nodes (small, medium, large, xlarge)"
  type        = string
  default     = "small"
  
  validation {
    condition     = contains(["small", "medium", "large", "xlarge"], var.redis_node_type)
    error_message = "Valid values for redis_node_type: small, medium, large, xlarge"
  }
}
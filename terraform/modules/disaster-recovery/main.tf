# Module to set up disaster recovery capabilities

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
}

variable "cloud_provider" {
  description = "Cloud provider (aws, azure, gcp)"
  type        = string
}

variable "project_name" {
  description = "Project name"
  type        = string
}

variable "backup_enabled" {
  description = "Whether to enable automated backups"
  type        = bool
  default     = true
}

locals {
  is_production = var.environment == "production"
  name_prefix   = "${var.project_name}-${var.environment}"
  
  # Backup retention based on environment
  backup_retention_days = {
    dev        = 7
    staging    = 14
    production = 30
  }
  
  # Backup frequency based on environment
  backup_frequency = {
    dev        = "daily"
    staging    = "12h"
    production = "6h"
  }
  
  # DR replication config based on environment
  dr_replication = {
    dev        = false
    staging    = local.is_production
    production = true
  }
}

# AWS Implementation
module "aws_backups" {
  count  = var.cloud_provider == "aws" && var.backup_enabled ? 1 : 0
  source = "./aws"
  
  name_prefix          = local.name_prefix
  backup_retention_days = local.backup_retention_days[var.environment]
  backup_frequency     = local.backup_frequency[var.environment]
  enable_cross_region  = local.dr_replication[var.environment]
  
  # Uncomment these when creating the actual AWS module
  # db_instance_arn      = var.db_instance_arn
  # eks_cluster_id       = var.eks_cluster_id
  # s3_buckets           = var.s3_buckets
}

# Azure Implementation
module "azure_backups" {
  count  = var.cloud_provider == "azure" && var.backup_enabled ? 1 : 0
  source = "./azure"
  
  name_prefix          = local.name_prefix
  backup_retention_days = local.backup_retention_days[var.environment]
  backup_frequency     = local.backup_frequency[var.environment]
  enable_geo_redundancy = local.dr_replication[var.environment]
  
  # Uncomment these when creating the actual Azure module
  # postgres_server_id   = var.postgres_server_id
  # aks_cluster_id       = var.aks_cluster_id
  # storage_accounts     = var.storage_accounts
}

# GCP Implementation
module "gcp_backups" {
  count  = var.cloud_provider == "gcp" && var.backup_enabled ? 1 : 0
  source = "./gcp"
  
  name_prefix          = local.name_prefix
  backup_retention_days = local.backup_retention_days[var.environment]
  backup_frequency     = local.backup_frequency[var.environment]
  enable_multi_regional = local.dr_replication[var.environment]
  
  # Uncomment these when creating the actual GCP module
  # database_instance_name = var.database_instance_name
  # gke_cluster_name       = var.gke_cluster_name
  # storage_buckets        = var.storage_buckets
}
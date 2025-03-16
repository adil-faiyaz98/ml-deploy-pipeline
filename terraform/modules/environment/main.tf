# This module configures environment-specific settings

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
}

variable "cloud_provider" {
  description = "Cloud provider to use (aws, azure, gcp)"
  type        = string
}

locals {
  # Environment configurations for different clouds
  aws_configs = {
    dev = {
      instance_types       = ["t3.medium"]
      node_min             = 2
      node_max             = 5
      db_instance          = "db.t3.medium"
      redis_instance       = "cache.t3.medium"
      storage_class        = "gp2"
      use_spot_instances   = true
      multi_az             = false
      backup_retention     = 7
      auto_minor_upgrades  = true
      monitoring_interval  = 60
    }
    staging = {
      instance_types       = ["t3.large", "m5.large"]
      node_min             = 3
      node_max             = 8
      db_instance          = "db.m5.large"
      redis_instance       = "cache.m5.large"
      storage_class        = "gp2"
      use_spot_instances   = true
      multi_az             = true
      backup_retention     = 14
      auto_minor_upgrades  = true
      monitoring_interval  = 30
    }
    production = {
      instance_types       = ["m5.large", "m5.xlarge"]
      node_min             = 5
      node_max             = 20
      db_instance          = "db.m5.xlarge"
      redis_instance       = "cache.m5.xlarge"
      storage_class        = "gp3"
      use_spot_instances   = false
      multi_az             = true
      backup_retention     = 30
      auto_minor_upgrades  = false
      monitoring_interval  = 15
    }
  }
  
  azure_configs = {
    dev = {
      vm_sizes            = ["Standard_D2s_v3"]
      node_min            = 2
      node_max            = 5
      db_sku              = "GP_Gen5_2"
      redis_sku           = "Basic"
      storage_class       = "managed-premium"
      use_spot_instances  = true
      zone_redundant      = false
      backup_retention    = 7
      auto_minor_upgrades = true
    }
    staging = {
      vm_sizes            = ["Standard_D4s_v3"]
      node_min            = 3
      node_max            = 8
      db_sku              = "GP_Gen5_4"
      redis_sku           = "Standard"
      storage_class       = "managed-premium"
      use_spot_instances  = true
      zone_redundant      = true
      backup_retention    = 14
      auto_minor_upgrades = true
    }
    production = {
      vm_sizes            = ["Standard_D4s_v3", "Standard_D8s_v3"]
      node_min            = 5
      node_max            = 20
      db_sku              = "GP_Gen5_8"
      redis_sku           = "Premium"
      storage_class       = "managed-premium"
      use_spot_instances  = false
      zone_redundant      = true
      backup_retention    = 30
      auto_minor_upgrades = false
    }
  }
  
  gcp_configs = {
    dev = {
      machine_types       = ["n1-standard-2"]
      node_min            = 2
      node_max            = 5
      db_tier             = "db-custom-2-8192"
      redis_tier          = "BASIC"
      storage_class       = "standard"
      use_preemptible     = true
      regional            = false
      backup_retention    = 7
      auto_minor_upgrades = true
    }
    staging = {
      machine_types       = ["n1-standard-4"]
      node_min            = 3
      node_max            = 8
      db_tier             = "db-custom-4-15360"
      redis_tier          = "STANDARD_HA"
      storage_class       = "premium-rwo"
      use_preemptible     = true
      regional            = true
      backup_retention    = 14
      auto_minor_upgrades = true
    }
    production = {
      machine_types       = ["n1-standard-4", "n1-standard-8"]
      node_min            = 5
      node_max            = 20
      db_tier             = "db-custom-8-30720"
      redis_tier          = "STANDARD_HA"
      storage_class       = "premium-rwo"
      use_preemptible     = false
      regional            = true
      backup_retention    = 30
      auto_minor_upgrades = false
    }
  }
  
  # Select config based on provider and environment
  config = local.cloud_provider == "aws" ? local.aws_configs[var.environment] :
           local.cloud_provider == "azure" ? local.azure_configs[var.environment] :
           local.gcp_configs[var.environment]
}

output "config" {
  description = "Environment-specific configuration"
  value       = local.config
}
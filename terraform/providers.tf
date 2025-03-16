# ------------------------------------------------------------------------------
# CONFIGURE PROVIDERS
# ------------------------------------------------------------------------------

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      Terraform   = "true"
      Owner       = var.owner
      CostCenter  = var.cost_center
    }
  }
  
  # Only needed when using terraform on local workstation
  # In CI/CD pipeline, these will be provided by GitHub Actions
  # profile = var.aws_profile
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy    = false
      recover_soft_deleted_key_vaults = true
    }
    resource_group {
      prevent_deletion_if_contains_resources = true
    }
    virtual_machine {
      delete_os_disk_on_deletion = true
    }
  }
  
  # Only needed when using terraform on local workstation
  # In CI/CD pipeline, these will be provided by GitHub Actions
  # subscription_id = var.azure_subscription_id
  # tenant_id       = var.azure_tenant_id
  # client_id       = var.azure_client_id
  # client_secret   = var.azure_client_secret
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
  
  # Only needed when using terraform on local workstation
  # In CI/CD pipeline, these will be provided by GitHub Actions
  # credentials = file(var.gcp_credentials_file)
}

provider "kubernetes" {
  # Configuration is sourced dynamically based on cloud provider
  # This empty block allows for provider definition without static configuration
  # Actual connection parameters are injected during CI/CD pipeline
}

provider "helm" {
  kubernetes {
    # Configuration is sourced dynamically based on cloud provider
    # This empty block allows for provider definition without static configuration
  }
}

provider "random" {}

provider "local" {}
# ------------------------------------------------------------------------------
# TERRAFORM BACKEND CONFIGURATION
# ------------------------------------------------------------------------------
# This file configures remote state storage
# Uncomment and configure the appropriate backend for your environment

# Backend configuration for Terraform state management

terraform {
  backend "s3" {
    bucket         = "ml-deploy-terraform-state"
    key            = "terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "ml-deploy-terraform-locks"
    encrypt        = true
  }
}

# Uncomment this block for Azure state storage
# terraform {
#   backend "azurerm" {
#     resource_group_name  = "terraform-state-rg"
#     storage_account_name = "mldeploytfstate"
#     container_name       = "tfstate"
#     key                  = "terraform.tfstate"
#   }
# }

# Uncomment this block for GCP state storage
# terraform {
#   backend "gcs" {
#     bucket  = "ml-deploy-terraform-state"
#     prefix  = "terraform/state"
#   }
# }
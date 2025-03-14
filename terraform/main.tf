# ------------------------------------------------------------------------------
# TERRAFORM CONFIGURATION
# ------------------------------------------------------------------------------
terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.10"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.70"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.80"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }
  
  # Configure remote state - uncomment and configure for production
  # backend "s3" {}           # For AWS
  # backend "azurerm" {}      # For Azure
  # backend "gcs" {}          # For GCP
}

# ------------------------------------------------------------------------------
# PROVIDER CONFIGURATION
# ------------------------------------------------------------------------------

# AWS Provider - only initialized when selected
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      Terraform   = "true"
      Owner       = var.owner
    }
  }
  
  skip_requesting_account_id = var.cloud_provider != "aws"
  skip_credentials_validation = var.cloud_provider != "aws"
  skip_metadata_api_check     = var.cloud_provider != "aws"
}

# Azure Provider - only initialized when selected
provider "azurerm" {
  features {}
  
  subscription_id = var.azure_subscription_id
  tenant_id       = var.azure_tenant_id
  
  skip_provider_registration = var.cloud_provider != "azure"
}

# GCP Provider - only initialized when selected
provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# Kubernetes Provider - dynamically configured based on cloud
provider "kubernetes" {
  host                   = local.cluster_endpoint
  cluster_ca_certificate = local.kubeconfig.cluster_ca_certificate
  token                  = local.kubeconfig.token
  
  # AWS specific config
  dynamic "exec" {
    for_each = var.cloud_provider == "aws" ? [1] : []
    content {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", local.kubeconfig.cluster_name]
    }
  }
}

# Helm Provider - uses the kubernetes provider config
provider "helm" {
  kubernetes {
    host                   = local.cluster_endpoint
    cluster_ca_certificate = local.kubeconfig.cluster_ca_certificate
    token                  = local.kubeconfig.token
    
    # AWS specific config
    dynamic "exec" {
      for_each = var.cloud_provider == "aws" ? [1] : []
      content {
        api_version = "client.authentication.k8s.io/v1beta1"
        command     = "aws"
        args        = ["eks", "get-token", "--cluster-name", local.kubeconfig.cluster_name]
      }
    }
  }
}

# ------------------------------------------------------------------------------
# LOCAL VARIABLES
# ------------------------------------------------------------------------------
locals {
  name_prefix = "${var.project_name}-${var.environment}"
  
  eks_managed_node_groups = {
    ml_compute = {
      name           = "ml-compute"
      instance_types = ["m5.2xlarge", "m5.4xlarge"]
      min_size       = var.min_nodes
      max_size       = var.max_nodes
      desired_size   = var.desired_nodes
      
      # Use mixed instances policy for cost optimization
      capacity_type  = "SPOT"  # Use SPOT for non-prod, ON_DEMAND for prod
      
      labels = {
        workload-type = "ml-training"
      }
      
      taints = []
      
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 100
            volume_type           = "gp3"
            iops                  = 3000
            throughput            = 150
            encrypted             = true
            delete_on_termination = true
          }
        }
      }
    },
    
    api_serving = {
      name           = "api-serving"
      instance_types = ["c5.xlarge", "c5.2xlarge"]
      min_size       = 2
      max_size       = 10
      desired_size   = 2
      
      capacity_type  = "ON_DEMAND"
      
      labels = {
        workload-type = "model-serving"
      }
      
      taints = []
    },
    
    monitoring = {
      name           = "monitoring"
      instance_types = ["t3.large"]
      min_size       = 1
      max_size       = 3
      desired_size   = 1
      
      capacity_type  = "ON_DEMAND"
      
      labels = {
        workload-type = "monitoring"
      }
    }
  }
  
  vpc_cidr = "10.0.0.0/16"
  
  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  
  private_subnets = [
    cidrsubnet(local.vpc_cidr, 4, 0),
    cidrsubnet(local.vpc_cidr, 4, 1),
    cidrsubnet(local.vpc_cidr, 4, 2)
  ]
  
  public_subnets = [
    cidrsubnet(local.vpc_cidr, 8, 0),
    cidrsubnet(local.vpc_cidr, 8, 1),
    cidrsubnet(local.vpc_cidr, 8, 2)
  ]
  
  database_subnets = [
    cidrsubnet(local.vpc_cidr, 8, 100),
    cidrsubnet(local.vpc_cidr, 8, 101),
    cidrsubnet(local.vpc_cidr, 8, 102)
  ]
  
  elastic_cache_subnets = [
    cidrsubnet(local.vpc_cidr, 8, 200),
    cidrsubnet(local.vpc_cidr, 8, 201),
    cidrsubnet(local.vpc_cidr, 8, 202)
  ]
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    Terraform   = "true"
    Owner       = var.owner
    CostCenter  = var.cost_center
  }
}

# ------------------------------------------------------------------------------
# DATA SOURCES
# ------------------------------------------------------------------------------
data "aws_availability_zones" "available" {}

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

# ------------------------------------------------------------------------------
# VPC & NETWORK INFRASTRUCTURE
# ------------------------------------------------------------------------------
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.1"
  
  name = "${local.name_prefix}-vpc"
  cidr = local.vpc_cidr
  
  azs              = local.azs
  private_subnets  = local.private_subnets
  public_subnets   = local.public_subnets
  database_subnets = local.database_subnets
  
  create_database_subnet_group       = true
  create_database_subnet_route_table = true
  
  enable_nat_gateway     = true
  single_nat_gateway     = var.environment != "production"
  one_nat_gateway_per_az = var.environment == "production"
  
  enable_vpn_gateway     = false
  enable_dns_hostnames   = true
  enable_dns_support     = true
  
  # VPC Flow Logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role  = true
  flow_log_max_aggregation_interval    = 60
  
  # Network security
  manage_default_security_group = true
  
  default_security_group_ingress = []
  default_security_group_egress  = []
  
  tags = local.common_tags
}

# ------------------------------------------------------------------------------
# EKS CLUSTER
# ------------------------------------------------------------------------------
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.15"
  
  cluster_name    = "${local.name_prefix}-cluster"
  cluster_version = var.kubernetes_version
  
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = true
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  enable_irsa = true
  
  # EKS Managed Node Groups
  eks_managed_node_groups = local.eks_managed_node_groups
  
  # Encryption for secrets
  cluster_encryption_config = [
    {
      provider_key_arn = aws_kms_key.eks.arn
      resources        = ["secrets"]
    }
  ]
  
  # Allow access from the management network
  cluster_security_group_additional_rules = {
    ingress_management_cidr = {
      description = "Management access from trusted network"
      protocol    = "tcp"
      from_port   = 443
      to_port     = 443
      type        = "ingress"
      cidr_blocks = var.management_ips
    }
  }
  
  node_security_group_additional_rules = {
    ingress_self_all = {
      description = "Allow nodes to communicate with each other"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      self        = true
    }
    egress_all = {
      description = "Allow nodes to communicate to the internet"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "egress"
      cidr_blocks = ["0.0.0.0/0"]
    }
  }
  
  # AWS IAM roles for service accounts (IRSA)
  cluster_identity_providers = {
    sts = {
      enabled = true
    }
  }
  
  tags = merge(local.common_tags, {
    "kubernetes.io/cluster/${local.name_prefix}-cluster" = "owned"
  })
  
  manage_aws_auth_configmap = true
  
  aws_auth_users = var.aws_auth_users
  aws_auth_roles = var.aws_auth_roles
}

# ------------------------------------------------------------------------------
# KMS KEYS
# ------------------------------------------------------------------------------
resource "aws_kms_key" "eks" {
  description             = "EKS Cluster Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  tags = local.common_tags
}

resource "aws_kms_key" "rds" {
  description             = "RDS Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  tags = local.common_tags
}

resource "aws_kms_key" "s3" {
  description             = "S3 Encryption Key for ML artifacts"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  tags = local.common_tags
}

# ------------------------------------------------------------------------------
# RDS POSTGRES DATABASE
# ------------------------------------------------------------------------------
module "db" {
  source  = "terraform-aws-modules/rds/aws"
  version = "~> 6.1"
  
  identifier = "${local.name_prefix}-postgres"
  
  engine               = "postgres"
  engine_version       = "14.7"
  family               = "postgres14"
  major_engine_version = "14"
  instance_class       = var.db_instance_type
  
  allocated_storage     = 100
  max_allocated_storage = 500
  
  db_name  = "mlmodels"
  username = "postgres"
  port     = 5432
  
  # Use a random password for security
  create_random_password = true
  random_password_length = 16
  
  multi_az               = var.environment == "production"
  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [aws_security_group.database.id]
  
  maintenance_window = "Mon:00:00-Mon:03:00"
  backup_window      = "03:00-06:00"
  
  # Enhanced monitoring
  monitoring_interval           = 60
  monitoring_role_name          = "${local.name_prefix}-rds-monitoring-role"
  create_monitoring_role        = true
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  
  # Backups and snapshots
  backup_retention_period = var.environment == "production" ? 30 : 7
  skip_final_snapshot     = var.environment != "production"
  final_snapshot_identifier = var.environment == "production" ? "${local.name_prefix}-db-final-snapshot" : null
  
  # Encryption
  storage_encrypted      = true
  kms_key_id             = aws_kms_key.rds.arn
  performance_insights_enabled = true
  
  # Upgrades
  auto_minor_version_upgrade = true
  apply_immediately          = var.environment != "production"
  
  # Parameters
  parameters = [
    {
      name  = "max_connections"
      value = "500"
    },
    {
      name  = "shared_buffers"
      value = "4096MB"
    },
    {
      name  = "work_mem"
      value = "64MB"
    }
  ]
  
  tags = local.common_tags
}

# ------------------------------------------------------------------------------
# ELASTICACHE REDIS
# ------------------------------------------------------------------------------
module "redis" {
  source  = "cloudposse/elasticache-redis/aws"
  version = "~> 0.52"
  
  name                       = "${local.name_prefix}-redis"
  vpc_id                     = module.vpc.vpc_id
  subnets                    = module.vpc.private_subnets
  availability_zones         = local.azs
  allowed_security_group_ids = [aws_security_group.redis_access.id]
  
  instance_type              = var.redis_node_type
  cluster_mode_enabled       = var.environment == "production"
  cluster_mode_num_node_groups = var.environment == "production" ? 3 : 1
  cluster_mode_replicas_per_node_group = var.environment == "production" ? 1 : 0
  
  automatic_failover_enabled = var.environment == "production"
  multi_az_enabled           = var.environment == "production"
  
  transit_encryption_enabled = true
  auth_token                 = random_password.redis_password.result
  
  parameter_group_name       = "default.redis7.cluster.on"
  at_rest_encryption_enabled = true
  
  # Maintenance and backups
  maintenance_window         = "sun:05:00-sun:07:00"
  snapshot_window            = "03:00-05:00"
  snapshot_retention_limit   = var.environment == "production" ? 7 : 1
  
  tags = local.common_tags
}

# ------------------------------------------------------------------------------
# S3 BUCKETS
# ------------------------------------------------------------------------------
module "model_artifacts_bucket" {
  source  = "terraform-aws-modules/s3-bucket/aws"
  version = "~> 3.15"
  
  bucket = "${local.name_prefix}-model-artifacts"
  
  # Bucket options
  force_destroy = var.environment != "production"
  
  # Versioning
  versioning = {
    enabled = true
  }
  
  # Encryption
  server_side_encryption_configuration = {
    rule = {
      apply_server_side_encryption_by_default = {
        kms_master_key_id = aws_kms_key.s3.arn
        sse_algorithm     = "aws:kms"
      }
    }
  }
  
  # Access logs
  logging = {
    target_bucket = module.log_bucket.s3_bucket_id
    target_prefix = "model-artifacts-logs/"
  }
  
  # S3 bucket-level Public Access Block
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
  
  # Object lifecycle
  lifecycle_rule = [
    {
      id      = "archive"
      enabled = true
      prefix  = "archived/"
      
      transition = [
        {
          days          = 30
          storage_class = "STANDARD_IA"
        },
        {
          days          = 90
          storage_class = "GLACIER"
        }
      ]
    }
  ]
  
  tags = local.common_tags
}

module "log_bucket" {
  source  = "terraform-aws-modules/s3-bucket/aws"
  version = "~> 3.15"
  
  bucket = "${local.name_prefix}-logs"
  acl    = "log-delivery-write"
  
  force_destroy = var.environment != "production"
  
  # Encryption
  server_side_encryption_configuration = {
    rule = {
      apply_server_side_encryption_by_default = {
        kms_master_key_id = aws_kms_key.s3.arn
        sse_algorithm     = "aws:kms"
      }
    }
  }
  
  # Versioning
  versioning = {
    enabled = true
  }
  
  # S3 bucket-level Public Access Block
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
  
  # Lifecycle rules for log rotation
  lifecycle_rule = [
    {
      id      = "log-expiration"
      enabled = true
      
      expiration = {
        days = var.environment == "production" ? 365 : 90
      }
      
      noncurrent_version_expiration = {
        days = 30
      }
    }
  ]
  
  tags = local.common_tags
}

# ------------------------------------------------------------------------------
# SECURITY GROUPS
# ------------------------------------------------------------------------------
resource "aws_security_group" "database" {
  name        = "${local.name_prefix}-database-sg"
  description = "Security group for database access"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    description     = "PostgreSQL from EKS nodes"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }
  
  # Allow VPN/management access if needed
  dynamic "ingress" {
    for_each = length(var.management_ips) > 0 ? [1] : []
    content {
      description = "PostgreSQL from management IPs"
      from_port   = 5432
      to_port     = 5432
      protocol    = "tcp"
      cidr_blocks = var.management_ips
    }
  }
  
  tags = local.common_tags
}

resource "aws_security_group" "redis_access" {
  name        = "${local.name_prefix}-redis-access-sg"
  description = "Security group for Redis access"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    description     = "Redis from EKS nodes"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }
  
  tags = local.common_tags
}

# ------------------------------------------------------------------------------
# RANDOM PASSWORDS
# ------------------------------------------------------------------------------
resource "random_password" "redis_password" {
  length  = 16
  special = false
}

# ------------------------------------------------------------------------------
# KUBERNETES DEPLOYMENTS
# ------------------------------------------------------------------------------

# Deploy Kubernetes resources using Helm
module "mlflow" {
  source = "./modules/helm-release"
  
  enabled      = var.deploy_mlflow
  
  release_name = "mlflow"
  chart        = "mlflow"
  repository   = "https://larribas.me/helm-charts"
  namespace    = "mlops"
  create_namespace = true
  
  values = [
    file("${path.module}/kubernetes/mlflow-values.yaml")
  ]
  
  set = [
    {
      name  = "backendStore.postgres.host"
      value = module.db.db_instance_address
    },
    {
      name  = "backendStore.postgres.port"
      value = module.db.db_instance_port
    },
    {
      name  = "backendStore.postgres.database"
      value = module.db.db_instance_name
    },
    {
      name  = "backendStore.postgres.user"
      value = module.db.db_instance_username
    },
    {
      name  = "artifactRoot.s3.bucket"
      value = module.model_artifacts_bucket.s3_bucket_id
    },
    {
      name  = "artifactRoot.s3.region"
      value = var.aws_region
    }
  ]
  
  set_sensitive = [
    {
      name  = "backendStore.postgres.password"
      value = module.db.db_instance_password
    }
  ]
  
  depends_on = [
    module.eks,
    module.db,
    module.model_artifacts_bucket
  ]
}

module "monitoring_stack" {
  source = "./modules/helm-release"
  
  enabled      = var.deploy_monitoring
  
  release_name = "kube-prometheus-stack"
  chart        = "kube-prometheus-stack"
  repository   = "https://prometheus-community.github.io/helm-charts"
  namespace    = "monitoring"
  create_namespace = true
  
  values = [
    file("${path.module}/kubernetes/prometheus-values.yaml")
  ]
  
  set = [
    {
      name  = "grafana.adminPassword"
      value = var.grafana_admin_password
    }
  ]
  
  depends_on = [
    module.eks
  ]
}

module "elastic_stack" {
  source = "./modules/helm-release"
  
  enabled      = var.deploy_elastic_stack
  
  release_name = "elastic-stack"
  chart        = "elastic-stack"
  repository   = "https://helm.elastic.co"
  namespace    = "logging"
  create_namespace = true
  
  values = [
    file("${path.module}/kubernetes/elastic-stack-values.yaml")
  ]
  
  depends_on = [
    module.eks
  ]
}

# ------------------------------------------------------------------------------
# KUBERNETES RESOURCES (CLOUD AGNOSTIC)
# ------------------------------------------------------------------------------

# Deploy MLflow using the Helm module (only if infrastructure is ready)
module "mlflow" {
  source = "./modules/helm-release"
  
  depends_on = [module.aws, module.azure, module.gcp]
  
  enabled = var.deploy_mlflow && (
    (var.cloud_provider == "aws" && length(module.aws) > 0) ||
    (var.cloud_provider == "azure" && length(module.azure) > 0) ||
    (var.cloud_provider == "gcp" && length(module.gcp) > 0)
  )
  
  release_name = "mlflow"
  chart        = "mlflow"
  repository   = "https://larribas.me/helm-charts"
  namespace    = "mlops"
  create_namespace = true
  
  values = [
    file("${path.module}/kubernetes/mlflow-values.yaml")
  ]
  
  # DB connection details passed dynamically based on cloud provider
  set = var.cloud_provider == "aws" ? module.aws[0].mlflow_helm_values :
       var.cloud_provider == "azure" ? module.azure[0].mlflow_helm_values :
       var.cloud_provider == "gcp" ? module.gcp[0].mlflow_helm_values : []
}

# Deploy Prometheus/Grafana monitoring stack
module "monitoring_stack" {
  source = "./modules/helm-release"
  
  depends_on = [module.aws, module.azure, module.gcp]
  
  enabled = var.deploy_monitoring && (
    (var.cloud_provider == "aws" && length(module.aws) > 0) ||
    (var.cloud_provider == "azure" && length(module.azure) > 0) ||
    (var.cloud_provider == "gcp" && length(module.gcp) > 0)
  )
  
  release_name = "kube-prometheus-stack"
  chart        = "kube-prometheus-stack"
  repository   = "https://prometheus-community.github.io/helm-charts"
  namespace    = "monitoring"
  create_namespace = true
  
  values = [
    file("${path.module}/kubernetes/prometheus-values.yaml")
  ]
  
  set = [
    {
      name  = "grafana.adminPassword"
      value = var.grafana_admin_password
    }
  ]
}

# ------------------------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------------------------

output "eks_cluster_id" {
  description = "EKS cluster ID"
  value       = module.eks.cluster_id
}

output "eks_cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "eks_cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "db_instance_endpoint" {
  description = "The connection endpoint for the RDS database"
  value       = module.db.db_instance_endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis primary endpoint"
  value       = module.redis.endpoint
}

output "model_artifacts_bucket" {
  description = "S3 bucket for model artifacts"
  value       = module.model_artifacts_bucket.s3_bucket_id
}

output "eks_oidc_provider_arn" {
  description = "ARN of the OIDC Provider for EKS"
  value       = module.eks.oidc_provider_arn
}

output "eks_node_groups" {
  description = "EKS node groups"
  value       = module.eks.eks_managed_node_groups
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "private_subnets" {
  description = "Private subnets"
  value       = module.vpc.private_subnets
}
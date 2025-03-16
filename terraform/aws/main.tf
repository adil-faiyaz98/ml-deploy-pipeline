# ------------------------------------------------------------------------------
# AWS IMPLEMENTATION
# ------------------------------------------------------------------------------

provider "aws" {
  region = var.region
  
  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      Terraform   = "true"
      Owner       = var.owner
      CostCenter  = var.cost_center
    }
  }
}

locals {
  name_prefix = "${var.project_name}-${var.environment}"
  
  vpc_cidr = var.vpc_cidr
  
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
  
  elasticache_subnets = [
    cidrsubnet(local.vpc_cidr, 8, 200),
    cidrsubnet(local.vpc_cidr, 8, 201),
    cidrsubnet(local.vpc_cidr, 8, 202)
  ]
  
  # Instance type mappings based on node size
  instance_types = {
    "small"  = "t3.medium"
    "medium" = "m5.large" 
    "large"  = "m5.2xlarge"
    "xlarge" = "m5.4xlarge"
    "gpu-small" = "g4dn.xlarge"
    "gpu-medium" = "g4dn.2xlarge"
    "gpu-large" = "g4dn.4xlarge"
  }
  
  # RDS instance type mappings
  db_instance_types = {
    "small"  = "db.t3.medium"
    "medium" = "db.m5.large"
    "large"  = "db.m5.xlarge"
    "xlarge" = "db.m5.2xlarge"
  }
  
  # ElastiCache instance type mappings
  redis_node_types = {
    "small"  = "cache.t3.medium"
    "medium" = "cache.m5.large"
    "large"  = "cache.m5.xlarge"
    "xlarge" = "cache.m5.2xlarge"
  }
  
  # EKS managed node groups configuration
  eks_managed_node_groups = {
    ml_compute = {
      name           = "ml-compute"
      instance_types = [local.instance_types["large"]]
      min_size       = var.min_nodes
      max_size       = var.max_nodes
      desired_size   = var.desired_nodes
      
      # Use mixed instances policy for cost optimization
      capacity_type  = var.environment == "production" ? "ON_DEMAND" : "SPOT"
      
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
      instance_types = [local.instance_types["medium"]]
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
      instance_types = [local.instance_types["small"]]
      min_size       = 1
      max_size       = 3
      desired_size   = 1
      
      capacity_type  = "ON_DEMAND"
      
      labels = {
        workload-type = "monitoring"
      }
    }
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
  elasticache_subnets = local.elasticache_subnets
  
  create_database_subnet_group       = true
  create_database_subnet_route_table = true
  create_elasticache_subnet_group    = true
  
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
  
  manage_aws_auth_configmap = true
  
  aws_auth_roles = var.eks_admin_roles
}

# ------------------------------------------------------------------------------
# KMS KEYS
# ------------------------------------------------------------------------------
resource "aws_kms_key" "eks" {
  description             = "EKS Cluster Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
}

resource "aws_kms_key" "rds" {
  description             = "RDS Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
}

resource "aws_kms_key" "s3" {
  description             = "S3 Encryption Key for ML artifacts"
  deletion_window_in_days = 7
  enable_key_rotation     = true
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
  instance_class       = local.db_instance_types[var.db_instance_type]
  
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
}

# ------------------------------------------------------------------------------
# ELASTICACHE REDIS
# ------------------------------------------------------------------------------
module "redis" {
  source = "cloudposse/elasticache-redis/aws"
  version = "~> 0.52"
  
  name                       = "${local.name_prefix}-redis"
  vpc_id                     = module.vpc.vpc_id
  subnets                    = module.vpc.elasticache_subnets
  availability_zones         = local.azs
  allowed_security_group_ids = [aws_security_group.redis_access.id]
  
  instance_type              = local.redis_node_types[var.redis_node_type]
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
}

# ------------------------------------------------------------------------------
# IAM ROLES FOR SERVICE ACCOUNTS (IRSA)
# ------------------------------------------------------------------------------
module "model_api_irsa_role" {
  source = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.30"

  role_name = "${local.name_prefix}-model-api-role"
  
  role_policy_arns = {
    s3_access = aws_iam_policy.model_api_s3_access.arn
    cloudwatch = aws_iam_policy.model_api_cloudwatch.arn
    secretsmanager = aws_iam_policy.model_api_secretsmanager.arn
  }
  
  oidc_providers = {
    main = {
      provider_arn = module.eks.oidc_provider_arn
      namespace_service_accounts = ["default:model-api"]
    }
  }
}

resource "aws_iam_policy" "model_api_s3_access" {
  name        = "${local.name_prefix}-model-api-s3-access"
  path        = "/"
  description = "IAM policy for Model API S3 access"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
          "s3:PutObject",
        ]
        Effect = "Allow"
        Resource = [
          module.model_artifacts_bucket.s3_bucket_arn,
          "${module.model_artifacts_bucket.s3_bucket_arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_policy" "model_api_cloudwatch" {
  name        = "${local.name_prefix}-model-api-cloudwatch"
  path        = "/"
  description = "IAM policy for Model API CloudWatch access"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Effect = "Allow"
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Effect = "Allow"
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_policy" "model_api_secretsmanager" {
  name        = "${local.name_prefix}-model-api-secretsmanager"
  path        = "/"
  description = "IAM policy for Model API Secrets Manager access"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Effect = "Allow"
        Resource = [
          "arn:aws:secretsmanager:${var.region}:${data.aws_caller_identity.current.account_id}:secret:${local.name_prefix}-*"
        ]
      }
    ]
  })
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
}

# ------------------------------------------------------------------------------
# SECRET MANAGER FOR CREDENTIALS
# ------------------------------------------------------------------------------
resource "aws_secretsmanager_secret" "model_api_credentials" {
  name        = "${local.name_prefix}-model-api-credentials"
  description = "Credentials for Model API"
  
  recovery_window_in_days = var.environment == "production" ? 30 : 0
}

resource "aws_secretsmanager_secret_version" "model_api_credentials" {
  secret_id = aws_secretsmanager_secret.model_api_credentials.id
  
  secret_string = jsonencode({
    db_host     = module.db.db_instance_address
    db_port     = module.db.db_instance_port
    db_name     = module.db.db_instance_name
    db_username = module.db.db_instance_username
    db_password = module.db.db_instance_password
    redis_host  = module.redis.endpoint
    redis_port  = 6379
    redis_auth  = random_password.redis_password.result
  })
}

# ------------------------------------------------------------------------------
# RANDOM PASSWORDS
# ------------------------------------------------------------------------------
resource "random_password" "redis_password" {
  length  = 16
  special = false
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

output "kubeconfig" {
  description = "kubeconfig for the cluster"
  value       = module.eks.kubeconfig
  sensitive   = true
}

output "cluster_endpoint" {
  description = "Endpoint for your Kubernetes API server"
  value       = module.eks.cluster_endpoint
}
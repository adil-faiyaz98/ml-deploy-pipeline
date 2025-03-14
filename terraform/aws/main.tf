# ------------------------------------------------------------------------------
# AWS INFRASTRUCTURE
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
  }
  
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
  
  # Map generic instance types to AWS-specific ones
  db_instance_type_map = {
    "small"  = "db.t3.medium",
    "medium" = "db.r5.large",
    "large"  = "db.r5.xlarge"
  }
  
  redis_node_type_map = {
    "small"  = "cache.t3.medium",
    "medium" = "cache.m5.large",
    "large"  = "cache.m5.xlarge"
  }
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    Terraform   = "true"
    Owner       = var.owner
    CostCenter  = var.cost_center
    Cloud       = "aws"
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
  
  # AWS IAM roles for service accounts (IRSA)
  cluster_identity_providers = {
    sts = {
      enabled = true
    }
  }
  
  tags = local.common_tags
  
  manage_aws_auth_configmap = true
  aws_auth_roles = concat(
    var.eks_admin_roles,
    [{
      rolearn  = module.eks_admin_iam_role.iam_role_arn
      username = "admin"
      groups   = ["system:masters"]
    }]
  )
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
  instance_class       = lookup(local.db_instance_type_map, var.db_instance_type, local.db_instance_type_map["small"])
  
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
  
  instance_type              = lookup(local.redis_node_type_map, var.redis_node_type, local.redis_node_type_map["small"])
  cluster_mode_enabled       = var.environment == "production"
  cluster_mode_num_node_groups = var.environment == "production" ? 3 : 1
  cluster_mode_replicas_per_node_group = var.environment == "production" ? 1 : 0
  
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
  
  tags = local.common_tags
}

resource "aws_kms_key" "s3" {
  description             = "S3 Encryption Key for ML artifacts"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
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
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
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
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = local.common_tags
}

# ------------------------------------------------------------------------------
# IAM ROLES
# ------------------------------------------------------------------------------
module "eks_admin_iam_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-assumable-role"
  version = "~> 5.30"
  
  create_role = true
  role_name   = "${local.name_prefix}-eks-admin"
  
  trusted_role_arns = [
    "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
  ]
  
  custom_role_policy_arns = [
    "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  ]
  
  tags = local.common_tags
}

# ------------------------------------------------------------------------------
# MLFLOW HELM VALUES FOR AWS
# ------------------------------------------------------------------------------
locals {
  mlflow_helm_values = [
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
      name
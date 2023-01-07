terraform {
  backend "s3" {
    bucket = "terraform-scripts-state"
    key    = "sagemaker.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = local.region
}

resource "aws_s3_bucket" "sagemaker" {
  bucket = local.data_bucket_name
  force_destroy = true
}


resource "aws_s3_bucket_public_access_block" "example" {
  bucket = aws_s3_bucket.sagemaker.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_sagemaker_code_repository" "example" {
  code_repository_name = local.repo_name

  git_config {
    repository_url = var.notebook_repo
  }
}

resource "aws_sagemaker_notebook_instance" "notebook_instance" {
  name                    = local.notebook_name
  role_arn                = aws_iam_role.sm_role.arn
  instance_type           = var.instance_type
  default_code_repository = aws_sagemaker_code_repository.example.code_repository_name

}

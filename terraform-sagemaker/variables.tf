variable "instance_type" {
  type        = string
  description = "Notebook instance type"
  default     = "ml.m5.2xlarge"
}


variable "notebook_repo" {
  type        = string
  description = "git repo for sagemaker notebook endpoint"
  default     = "https://github.com/ryankarlos/AWS-Sagemaker.git"
  validation {
    condition     = can(regex("^https://github.com/", var.notebook_repo)) && substr(var.notebook_repo, -4, -1) == ".git"
    error_message = "url should begin with https://github.com/ and end with .git"
  }
}


locals {
  data_bucket_name    = "sagemaker-experiments"
  backend_bucket_name = "terraform-scripts-state"
  backend_bucket_key  = "sagemaker.tfstate"
  region              = "us-east-1"
}

locals {
  notebook_name   = join("-", [local.data_bucket_name, "notebook"])
  repo_name       = join("-", [local.data_bucket_name, "repo"])
  iam_role_name   = join("-", [local.data_bucket_name, "role"])
  iam_policy_name = join("-", [local.data_bucket_name, "policy"])
}

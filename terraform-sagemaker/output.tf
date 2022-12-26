output "s3_bucket_arn" {
  description = "The ARN of the bucket. Will be of format arn:aws:s3:::bucketname."
  value       = aws_s3_bucket.sagemaker.arn
}


output "iam_role_arn" {
  description = "ARN of sagemaker IAM role"
  value       = aws_iam_role.sm_role.arn
}

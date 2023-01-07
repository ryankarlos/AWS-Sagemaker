data "aws_iam_policy_document" "foo" {
  statement {
    effect = "Allow"
    actions = [
      "sagemaker:*",
      "iam:GetRole",
      "iam:PassRole"
    ]
    resources = [
      "*"
    ]
  }
  statement {
    effect = "Allow"
    actions = [
      "cloudwatch:PutMetricData",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:GetLogEvents",
      "logs:CreateLogGroup",
      "logs:DescribeLogStreams",
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage"
    ]
    resources = [
    "*"]
  }
  statement {
    effect = "Allow"
    actions = [
      "s3:*"
    ]
    resources = [
      "arn:aws:s3:::${aws_s3_bucket.sagemaker.bucket}",
      "arn:aws:s3:::${aws_s3_bucket.sagemaker.bucket}/*"
    ]
  }
}

resource "aws_iam_policy" "foo" {
  name        = local.iam_policy_name
  description = "Allow Sagemaker to create model"
  policy      = data.aws_iam_policy_document.foo.json
}



resource "aws_iam_role" "sm_role" {
  name = local.iam_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = ""
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      },
    ]
  })
}


resource "aws_iam_policy_attachment" "sm-attach" {
  name       = "sm-role-policy"
  roles      = ["${aws_iam_role.sm_role.name}"]
  policy_arn = "${aws_iam_policy.foo.arn}"
}
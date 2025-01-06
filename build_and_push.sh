# execution:  build_and_push.sh $account_id $region $ecr_repository_name

ACCOUNT_ID=$1
REGION=$2
REPO_NAME=$3

if [[ $REGION =~ ^cn.* ]]
then
    FULLNAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com.cn/${REPO_NAME}:latest"
else
    FULLNAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest"
fi

echo $FULLNAME

docker build -f Dockerfile -t $REPO_NAME .

docker tag $REPO_NAME $FULLNAME

$(aws ecr get-login --no-include-email --registry-ids $ACCOUNT_ID)

aws ecr describe-repositories --repository-names $REPO_NAME || aws ecr create-repository --repository-name $REPO_NAME

docker push $FULLNAME



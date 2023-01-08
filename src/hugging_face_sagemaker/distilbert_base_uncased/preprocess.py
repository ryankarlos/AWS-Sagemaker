from datasets import load_dataset
from transformers import AutoTokenizer
from datasets.filesystems import S3FileSystem
from sagemaker_config import S3_BUCKET, S3_PREFIX_IMDB


def tokenize(batch):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert_base_uncased")
    return tokenizer(batch["text"], padding="max_length", truncation=True)


def set_dataset_format_for_pytorch(train, test):
    # tokenize train and test datasets
    train_dataset = train.map(tokenize, batched=True)
    test_dataset = test.map(tokenize, batched=True)
    train_dataset = train_dataset.rename_column("label", "labels")
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return train_dataset, test_dataset


def get_bucket_paths():
    training_input_path = f"s3://{S3_BUCKET}/{S3_PREFIX_IMDB}/train"
    test_input_path = f"s3://{S3_BUCKET}/{S3_PREFIX_IMDB}/test"
    return training_input_path, test_input_path


def upload_to_s3(train_dataset, test_dataset):
    s3 = S3FileSystem()
    training_input_path, test_input_path = get_bucket_paths()
    # save train_dataset to S3
    train_dataset.save_to_disk(training_input_path, fs=s3)
    # save test_dataset to S3
    test_dataset.save_to_disk(test_input_path, fs=s3)


if __name__ == "__main__":
    # load dataset
    train_dataset, test_dataset = load_dataset("imdb", split=["train", "test"])
    train, test = set_dataset_format_for_pytorch(train_dataset, test_dataset)
    print(train)
    print(test)
    upload_to_s3(train, test)

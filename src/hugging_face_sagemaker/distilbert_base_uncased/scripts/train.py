from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk
import logging
import sys
import argparse
import os
from src.hugging_face_sagemaker.distilbert_base_uncased.preprocess import (
    get_bucket_paths,
)


# Set up logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def base_command_line_args():
    """hyperparameters sent by the client are passed as command-line
    arguments to the script."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--mode", type=str, default="sagemaker")
    return parser


def addition_command_args(parser, mode=None):
    if mode == "local":
        training_dir, test_dir = get_bucket_paths()
        os.environ["SM_OUTPUT_DATA_DIR"] = "."
        os.environ["SM_MODEL_DIR"] = "."
        os.environ["SM_NUM_GPUS"] = "0"
        os.environ["SM_CHANNEL_TRAIN"] = training_dir
        os.environ["SM_CHANNEL_TEST"] = test_dir
    parser.add_argument(
        "--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument(
        "--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"]
    )
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    return parser


def create_training_model_instance():
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(model)
    print(tokenizer)
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
    )
    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    return trainer


def compute_metrics(pred):
    """
    compute metrics function for binary classification
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


if __name__ == "__main__":
    parser = base_command_line_args()
    args, _ = parser.parse_known_args()
    if args.mode == "local":
        from datasets.filesystems import S3FileSystem

        aws_access_key_id = os.environ["ACCESS_KEY_ID"]
        aws_secret_access_key = os.environ["SECRET_ACCESS_KEY"]
        args, _ = addition_command_args(parser, mode="local").parse_known_args()
        # create S3FileSystem without credentials
        s3 = S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
        # load datasets
        train_dataset = load_from_disk(args.training_dir, fs=s3)
        test_dataset = load_from_disk(args.test_dir, fs=s3)
    elif args.mode == "sagemaker":
        args, _ = addition_command_args(parser).parse_known_args()
        train_dataset = load_from_disk(args.training_dir)
        test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")
    trainer = create_training_model_instance()
    # train model
    trainer.train()
    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)
    # writes eval result to file which can be accessed
    # later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print("***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")
    # Saves the model to s3
    trainer.save_model(args.model_dir)

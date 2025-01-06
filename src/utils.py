from pathlib import Path
import tarfile
import sagemaker

def get_repo_root(filepath):
    for path in Path(filepath).parents:
        if (path / ".git").exists():
            return path


def upload_source_code_to_s3(program, source, session=None, bucket=None):
    # first compress the code and send to S3
    tar = tarfile.open(source, 'w:gz')
    tar.add(program)
    tar.close()

    if session is None:
        sess = sagemaker.session.Session()
    if bucket is None:
        bucket = sess.default_bucket()

    submit_dir = sess.upload_data(
        path=source, 
        bucket=bucket,
        key_prefix="source")

    print(submit_dir)
    return submit_dir

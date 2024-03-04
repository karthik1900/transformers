import json
import os
import subprocess
import boto3
import torch


def get_torch_dtype(dtype: str):
    if dtype == "F32":
        return torch.float32
    elif dtype == "F16":
        return torch.float16
    elif dtype == "F64":
        return torch.float64
    elif dtype == "BF16":
        return torch.bfloat16
    elif dtype == "I64":
        return torch.int64
    elif dtype == "I32":
        return torch.int32
    elif dtype == "I16":
        return torch.int16
    elif dtype == "I8":
        return torch.int8
    elif dtype == "U8":
        return torch.uint8
    elif dtype == "BOOL":
        return torch.bool
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def is_cloud_storage_keys(model):
    try:
        model_id = os.environ.get("MODEL_ID")
        # cloud_storage_keys format is { "stable-diff": "s3://bucket-name/path/to/model",
        # "gpt": "s3://bucket-name/path/to/model"}
        cloud_storage_keys = os.environ.get(f"{model_id}_CLOUD_STORAGE_KEYS")
        model_name = model.split("/")[-1]
        if model_name in cloud_storage_keys:
            return True
    except Exception as e:
        return False
    return False


def is_model_available():
    return os.environ.get("IS_MODEL_AVAILABLE", "false") == "true"


def is_inferless_applicable(model):
    try:
        if not isinstance(model, str):
            return False

        if (
                os.environ.get("INFERLESS_LOADER") == "true" and
                os.environ.get("MODEL_FRAMEWORK") == "pytorch"
        ):
            return True

    except Exception as e:
        print(f"Error checking if model is applicable to Inferless: {e}")
        return False


def is_model_local_path(model):
    if os.path.exists(model):
        return True
    return False


def download_model_from_hf_hub(model):
    model_id = os.environ.get("MODEL_ID")
    model_name = model.split("/")[-1]
    local_path = os.path.join(os.getcwd(), f"inferless_{model_id}/{model_name}")
    # download model from hf hub
    from huggingface_hub import list_repo_files, get_paths_info, hf_hub_download
    files = list_repo_files(model)
    paths_info = get_paths_info(model, files)
    selected_files = []
    for path_info in paths_info:
        if path_info.path.endswith(".safetensors") or path_info.size < 100000000:
            model_path = path_info.path
            selected_files.append(model_path)

    for file in selected_files:
        print(f"Downloading {file} from Hugging Face Hub", flush=True)
        # download and save file to local_path
        hf_hub_download(model, file, local_dir=local_path, local_dir_use_symlinks=False)

    return local_path


def upload_to_s3(model):
    s3_bucket = os.environ.get("S3_BUCKET")
    model_id = os.environ.get("MODEL_ID")
    dest_s3_folder = f"s3://{s3_bucket}/{model_id}/"
    # use s5cmd to upload folder to s3
    resp = subprocess.run(["s5cmd", "cp", model, dest_s3_folder])
    if resp.returncode != 0:
        raise Exception(f"Error uploading {model} to {dest_s3_folder} with s5cmd")
    else:
        print(f"Uploaded {model} to {dest_s3_folder}")


def upload_to_gcs(model):
    pass


def upload_to_azure(model):
    pass


def download_non_weights_from_s3(model):
    # download all files except .safetensors from s3
    s3_bucket = os.environ.get("S3_BUCKET")
    model_id = os.environ.get("MODEL_ID")
    model_name = model.split("/")[-1]
    src_folder = f"s3://{s3_bucket}/{model_id}/{model_name}/*"
    dest_folder = os.path.join(os.getcwd(), f"inferless_{model_id}/{model_name}")
    resp = subprocess.run(["s5cmd", "cp", "--exclude", "*.safetensors", src_folder, dest_folder])
    if resp.returncode != 0:
        raise Exception(f"Error downloading {src_folder} to {dest_folder} with s5cmd")
    else:
        print(f"Downloaded {src_folder} to {dest_folder}")

    if "model.safetensors.index.json" not in os.listdir(dest_folder):
        # create a dummy model.safetensors file
        resp = subprocess.run(["touch", f"{dest_folder}/model.safetensors"])
        if resp.returncode != 0:
            raise Exception(f"Error creating model.safetensors file in {dest_folder}")
        else:
            print(f"Created model.safetensors file in {dest_folder}")

    return dest_folder


def download_model_from_gcs(model):
    pass


def download_model_from_azure(model):
    pass


def cleanup(model):
    resp = subprocess.run(["rm", "-rf", model])
    if resp.returncode != 0:
        raise Exception(f"Error cleaning up {model}")
    else:
        print(f"Cleaned up {model}")


def upload_model_to_inferless(model):
    if is_model_available():
        print("Model is already uploaded to Inferless")
        return
    to_cleanup = False
    if not is_model_local_path(model):
        print("Model is not a local path")
        model = download_model_from_hf_hub(model)
        to_cleanup = True

    cloud = os.environ.get("CLOUD")

    if cloud == "aws":
        upload_to_s3(model)
    elif cloud == "gcp":
        upload_to_gcs(model)
    elif cloud == "azure":
        upload_to_azure(model)

    if to_cleanup:
        cleanup(model)


def get_model_files(model):
    cloud = os.environ.get("CLOUD")

    if cloud == "aws":
        return download_non_weights_from_s3(model)
    elif cloud == "gcp":
        return download_model_from_gcs(model)
    elif cloud == "azure":
        return download_model_from_azure(model)


def get_metadata_single_file(model_file_path, file_name="model.safetensors") -> dict:
    s3_bucket = os.environ.get("S3_BUCKET")
    model_id = os.environ.get("MODEL_ID")
    model_name = model_file_path.split("/")[-1]
    file_name = file_name.split("/")[-1]
    s3_key = f"{model_id}/{model_name}/{file_name}"
    # read first 8 bytes using range request
    s3 = boto3.client("s3")
    resp = s3.get_object(Bucket=s3_bucket, Key=s3_key, Range="bytes=0-7")
    output = resp["Body"].read()
    metadata_size = int.from_bytes(output, byteorder="little")

    # read metadata using range request
    resp = s3.get_object(Bucket=s3_bucket, Key=s3_key, Range=f"bytes=8-{7+metadata_size}")
    metadata = resp["Body"].read()
    metadata = json.loads(metadata)

    return metadata["__metadata__"]


def get_state_dict_single_file(model_file_path, file_name="model.safetensors"):
    s3_bucket = os.environ.get("S3_BUCKET")
    model_id = os.environ.get("MODEL_ID")
    model_name = model_file_path.split("/")[-1]
    file_name = file_name.split("/")[-1]
    s3_key = f"{model_id}/{model_name}/{file_name}"
    from custom_loader import get_s3_weights
    print(f"Getting state dict from {s3_bucket}/{s3_key}")
    return get_s3_weights(s3_bucket, s3_key)


def get_torch_dtype_sharded(model_file_path, resolved_files):
    metadata = get_metadata_single_file(model_file_path, resolved_files[0])
    metadata.pop("__metadata__")
    for k, v in metadata.items():
        if "dtype" in v:
            return get_torch_dtype(v["dtype"])

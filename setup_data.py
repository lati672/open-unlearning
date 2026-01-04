import argparse
import os
import subprocess
import sys
from huggingface_hub import snapshot_download


def download_eval_data():
    snapshot_download(
        repo_id="open-unlearning/eval",
        allow_patterns="*.json",
        repo_type="dataset",
        local_dir="saves/eval",
    )


def download_idk_data():
    snapshot_download(
        repo_id="open-unlearning/idk",
        allow_patterns="*.jsonl",
        repo_type="dataset",
        local_dir="data",
    )


def download_wmdp():
    url = "https://cais-wmdp.s3.us-west-1.amazonaws.com/wmdp-corpora.zip"
    dest_dir = "data/wmdp"
    zip_path = os.path.join(dest_dir, "wmdp-corpora.zip")

    os.makedirs(dest_dir, exist_ok=True)
    subprocess.run(["wget", url, "-O", zip_path], check=True)
    subprocess.run(["unzip", "-P", "wmdpcorpora", zip_path, "-d", dest_dir], check=True)


def generate_tofu_logprobs(splits):
    for split in splits:
        command = [
            sys.executable,
            "src/compute_logprobs.py",
            f"dataset.TOFU_QA_forget.args.hf_args.name={split}",
            f"+dataset_split={split}",
        ]
        subprocess.run(command, check=True)


def generate_muse_logprobs():
    model = os.environ.get("MODEL", "Llama-2-7b-hf")
    batch_size = os.environ.get("BATCH_SIZE", "4")
    data_splits = ("News", "Books")
    forget_split = os.environ.get("FORGET_SPLIT", "forget")

    for muse_split in data_splits:
        dataset_path = f"muse-bench/MUSE-{muse_split}"
        command = [
            sys.executable,
            "src/compute_logprobs.py",
            f"model={model}",
            f"batch_size={batch_size}",
            "data/datasets@dataset=MUSE_forget",
            f"dataset.MUSE_forget.args.hf_args.path={dataset_path}",
            f"dataset.MUSE_forget.args.hf_args.split={forget_split}",
            "dataset.MUSE_forget.args.hf_args.name=raw",
            f"+dataset_split={forget_split}",
        ]
        subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Download and setup evaluation data.")
    parser.add_argument(
        "--eval_logs",
        action="store_true",
        help="Downloads TOFU, MUSE  - retain and finetuned models eval logs and saves them in saves/eval",
    )
    parser.add_argument(
        "--idk",
        action="store_true",
        help="Download idk dataset from HF hub and stores it data/idk.jsonl",
    )
    parser.add_argument(
        "--wmdp",
        action="store_true",
        help="Download and unzip WMDP dataset into data/wmdp",
    )
    parser.add_argument(
        "--logprobs",
        action="store_true",
        help=(
            "Generate TOFU forget01/05/10 and MUSE (News/Books) logprobs via "
            "src/compute_logprobs.py"
        ),
    )
    parser.add_argument(
        "--tofu-logprobs",
        action="store_true",
        help="Generate TOFU forget01/05/10 logprobs via src/compute_logprobs.py",
    )
    parser.add_argument(
        "--muse-logprobs",
        action="store_true",
        help="Generate MUSE (News/Books) logprobs via src/compute_logprobs.py",
    )

    args = parser.parse_args()

    if args.eval_logs:
        download_eval_data()
    if args.idk:
        download_idk_data()
    if args.wmdp:
        download_wmdp()
    if args.logprobs:
        generate_tofu_logprobs(["forget01", "forget05", "forget10"])
        generate_muse_logprobs()
    if args.tofu_logprobs:
        generate_tofu_logprobs(["forget01", "forget05", "forget10"])
    if args.muse_logprobs:
        generate_muse_logprobs()


if __name__ == "__main__":
    main()

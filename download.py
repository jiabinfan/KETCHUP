from huggingface_hub import snapshot_download
snapshot_download(repo_id="google-t5/t5-base", 
                  cache_dir="/home/gluo",
                  ignore_patterns=["*.msgpack", "*.h5"])

import subprocess
import time
import os

from weights_manifest import WeightsManifest

BASE_URL = "https://weights.replicate.delivery/default/comfy-ui"


class WeightsDownloader:
    def __init__(self):
        self.weights_map = WeightsManifest().weights_map

    def download_weights(self, weight_str):
        if weight_str in self.weights_map:
            self.download_if_not_exists(
                weight_str,
                self.weights_map[weight_str]["url"],
                self.weights_map[weight_str]["dest"],
            )
        else:
            raise ValueError(
                f"{weight_str} unavailable. View the list of available weights: https://github.com/fofr/cog-comfyui/blob/main/supported_weights.md"
            )

    def download_if_not_exists(self, weight_str, url, dest):
        if not os.path.exists(f"{dest}/{weight_str}"):
            self.download(weight_str, url, dest)

    def download(self, weight_str, url, dest):
        print(f"⏳ Downloading {weight_str}")
        start = time.time()
        subprocess.check_call(
            ["pget", "--log-level", "warn", "-xf", url, dest], close_fds=False
        )
        elapsed_time = time.time() - start
        file_size_bytes = os.path.getsize(f"{dest}/{weight_str}")
        file_size_megabytes = file_size_bytes / (1024 * 1024)
        print(
            f"⌛️ Download {weight_str} took: {elapsed_time:.2f}s, size: {file_size_megabytes:.2f}MB"
        )

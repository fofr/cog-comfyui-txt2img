import json

WEIGHTS_MANIFEST_PATH = "weights.json"
BASE_URL = "https://weights.replicate.delivery/default/comfy-ui"
BASE_PATH = "ComfyUI/models"


class WeightsManifest:
    def __init__(self):
        self.weights_manifest = self._load_weights_manifest()
        self.weights_map = self._initialize_weights_map()

    def _load_weights_manifest(self):
        with open(WEIGHTS_MANIFEST_PATH, "r") as f:
            return json.load(f)

    def _generate_weights_map(self, keys, dest):
        return {
            key: {
                "url": f"{BASE_URL}/{dest}/{key}.tar",
                "dest": f"{BASE_PATH}/{dest}",
            }
            for key in keys
        }

    def _initialize_weights_map(self):
        weights_map = {}
        for key in self.weights_manifest.keys():
            if key.isupper():
                weights_map.update(
                    self._generate_weights_map(self.weights_manifest[key], key.lower())
                )
        return weights_map

import json


class WorkflowEditor:
    def __init__(self, comfyui):
        self.comfyui = comfyui
        self.load_workflow()

    def load_workflow(self):
        self.workflow = self.load_json_file("examples/txt2img.json")
        self.checkpoints = self.load_json_file("weights.json")["CHECKPOINTS"]

    def find_checkpoint_file(self, checkpoint_name):
        for ckpt_file in self.checkpoints:
            if ckpt_file.startswith(checkpoint_name):
                return ckpt_file
        return None

    @staticmethod
    def load_json_file(file_path):
        with open(file_path, "r") as file:
            return json.load(file)

    def update_workflow(self, **kwargs):
        node_mappings = {
            "seed": ("3", "inputs", "seed"),
            "steps": ("3", "inputs", "steps"),
            "cfg": ("3", "inputs", "cfg"),
            "sampler_name": ("3", "inputs", "sampler_name"),
            "scheduler": ("3", "inputs", "scheduler"),
            "prompt": ("6", "inputs", "text"),
            "negative_prompt": ("7", "inputs", "text"),
            "batch_size": ("5", "inputs", "batch_size"),
            "ckpt_name": ("4", "inputs", "ckpt_name"),
            "width": ("5", "inputs", "width"),
            "height": ("5", "inputs", "height"),
        }

        for key, value in kwargs.items():
            if value is not None:
                if key == "ckpt_name":
                    value = self.find_checkpoint_file(value)
                    if value is None:
                        continue
                elif key in ["width", "height"]:
                    value = 8 * round(value / 8)
                node_id, section, input_name = node_mappings[key]
                self.workflow[node_id][section][input_name] = value

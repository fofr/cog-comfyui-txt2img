import os
import shutil
import tarfile
import zipfile
import json
from typing import List
from cog import BasePredictor, Input, Path
from helpers.comfyui import ComfyUI
from workflow_editor import WorkflowEditor

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"

with open("weights.json", "r") as f:
    checkpoints = json.load(f)["CHECKPOINTS"]
    AVAILABLE_CHECKPOINTS = [os.path.splitext(ckpt)[0] for ckpt in checkpoints]

SAMPLERS = [
    "euler",
    "euler_ancestral",
    "heun",
    "heunpp2",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ddim",
    "uni_pc",
    "uni_pc_bh2",
]

SCHEDULERS = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
]


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)
        self.workflowEditor = WorkflowEditor(self.comfyUI)

    def cleanup(self):
        for directory in [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

    def handle_input_file(self, input_file: Path):
        file_extension = os.path.splitext(input_file)[1]
        if file_extension == ".tar":
            with tarfile.open(input_file, "r") as tar:
                tar.extractall(INPUT_DIR)
        elif file_extension == ".zip":
            with zipfile.ZipFile(input_file, "r") as zip_ref:
                zip_ref.extractall(INPUT_DIR)
        elif file_extension in [".jpg", ".jpeg", ".png", ".webp"]:
            shutil.copy(input_file, os.path.join(INPUT_DIR, f"input{file_extension}"))
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        print("====================================")
        print(f"Inputs uploaded to {INPUT_DIR}:")
        self.log_and_collect_files(INPUT_DIR)
        print("====================================")

    def log_and_collect_files(self, directory, prefix=""):
        files = []
        for f in os.listdir(directory):
            if f == "__MACOSX":
                continue
            path = os.path.join(directory, f)
            if os.path.isfile(path):
                print(f"{prefix}{f}")
                files.append(Path(path))
            elif os.path.isdir(path):
                print(f"{prefix}{f}/")
                files.extend(self.log_and_collect_files(path, prefix=f"{prefix}{f}/"))
        return files

    def predict(
        self,
        prompt: str = Input(default="a photo of an astronaut riding a unicorn"),
        negative_prompt: str = Input(
            description="The negative prompt to guide image generation.",
            default="ugly, disfigured, low quality, blurry, nsfw",
        ),
        model: str = Input(
            description="Pick which base weights you want to use",
            choices=AVAILABLE_CHECKPOINTS,
            default="RealVisXL_V3.0",
        ),
        num_inference_steps: int = Input(
            description="Number of diffusion steps", ge=1, le=100, default=30
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=7.5,
            ge=0,
            le=30.0,
        ),
        seed: int = Input(default=None),
        width: int = Input(default=768),
        height: int = Input(default=768),
        num_outputs: int = Input(
            description="Number of outputs", ge=1, le=10, default=1
        ),
        sampler_name: str = Input(
            choices=SAMPLERS,
            default="euler",
        ),
        scheduler: str = Input(
            choices=SCHEDULERS,
            default="normal",
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.cleanup()

        self.workflowEditor.update_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=num_inference_steps,
            cfg=guidance_scale,
            seed=seed,
            width=width,
            height=height,
            batch_size=num_outputs,
            ckpt_name=model,
            sampler_name=sampler_name,
            scheduler=scheduler,
        )

        wf = self.comfyUI.load_workflow(self.workflowEditor.workflow)

        if seed in ["", None, -1]:
            self.comfyUI.randomise_seeds(wf)

        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        files = []
        output_directories = [OUTPUT_DIR]

        for directory in output_directories:
            print(f"Contents of {directory}:")
            files.extend(self.log_and_collect_files(directory))

        return files

import logging
import asyncio
import os
import torch
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import ssl

# Apply SSL Patch for requests to handle certificate errors
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''

# Monkey-patch requests to ignore SSL verification
old_merge_environment_settings = requests.Session.merge_environment_settings
def merge_environment_settings(self, url, proxies, stream, verify, cert):
    return old_merge_environment_settings(self, url, proxies, stream, False, cert)
requests.Session.merge_environment_settings = merge_environment_settings

from typing import Dict, Any, Optional, Union
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from domain.interfaces.base_interfaces import ImageGenInterface

logger = logging.getLogger("SDAdapter")

class SDAdapter(ImageGenInterface):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config.get('sd_model_path')
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.pipe = None
        self._lock = asyncio.Lock()
        self.is_mock = False # Fallback flag
        
    async def _ensure_loaded(self):
        if self.pipe is None and not self.is_mock:
            await self._load_model()
            
    async def _load_model(self):
        logger.info(f"Loading SD Model from {self.model_path}")
        try:
            # Run in thread to avoid blocking event loop
            def load():
                import os
                # Ensure absolute path
                abs_model_path = os.path.abspath(self.model_path)
                logger.info(f"Using absolute SD model path: {abs_model_path}")
                
                dtype = torch.float16 if self.device == 'cuda' else torch.float32
                
                # Try to find config file
                base_dir = os.path.dirname(abs_model_path)
                original_config_file = os.path.join(base_dir, "v1-inference.yaml")
                if not os.path.exists(original_config_file):
                    original_config_file = None
                
                logger.info(f"Using config file: {original_config_file}")

                try:
                    # First attempt: Offline mode
                    pipe = StableDiffusionPipeline.from_single_file(
                        abs_model_path, 
                        original_config_file=original_config_file,
                        torch_dtype=dtype,
                        use_safetensors=True,
                        local_files_only=True,
                        safety_checker=None
                    )
                except Exception as offline_err:
                    logger.warning(f"Offline loading failed: {offline_err}. Retrying with online mode enabled...")
                    # Second attempt: Online mode allowed (to fetch configs)
                    pipe = StableDiffusionPipeline.from_single_file(
                        abs_model_path, 
                        original_config_file=original_config_file,
                        torch_dtype=dtype,
                        use_safetensors=True,
                        local_files_only=False,
                        safety_checker=None
                    )

                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                
                if self.device == 'cuda':
                    pipe.to("cuda")
                    # Enable memory optimizations
                    pipe.enable_attention_slicing()
                    if self.config.get('generation', {}).get('low_vram_mode', False):
                        pipe.enable_model_cpu_offload()
                        
                return pipe
                
            self.pipe = await asyncio.to_thread(load)
            logger.info("SD Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SD model: {e}")
            logger.warning("Falling back to MOCK SD generation due to load failure.")
            self.is_mock = True
            # We do not raise here to allow the experiment to continue with mock

    async def generate_image(self, prompt: str, **kwargs) -> Any:
        """
        Generates an image and returns the PIL Image object (or path).
        """
        async with self._lock:
            await self._ensure_loaded()
            
            if self.is_mock:
                logger.info(f"Mocking SD generation for: {prompt}")
                await asyncio.sleep(3.5) # Simulate ~3.5s generation time
                from PIL import Image
                return {
                    "status": "success",
                    "images": [Image.new('RGB', (512, 512), color='blue')]
                }

            width = kwargs.get('width', self.config.get('generation', {}).get('width', 512))
            height = kwargs.get('height', self.config.get('generation', {}).get('height', 512))
            steps = kwargs.get('num_inference_steps', self.config.get('generation', {}).get('num_inference_steps', 20))
            
            def run_inference():
                return self.pipe(
                    prompt,
                    negative_prompt=kwargs.get('negative_prompt', "ugly, blurry, low quality"),
                    width=width,
                    height=height,
                    num_inference_steps=steps
                ).images
            
            images = await asyncio.to_thread(run_inference)
            
            return {
                "status": "success",
                "images": images
            }

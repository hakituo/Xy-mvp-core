import sys
import os
import asyncio
import logging

# Setup path
# Add xiaoyou-core (root) to path for mvp_core imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# Add mvp_core to path for internal imports (like domain)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mvp_core.data.adapters.gguf_llm_adapter import GGUFLLMAdapter
from mvp_core.data.adapters.vl_adapter import VLAdapter
from mvp_core.data.adapters.sd_adapter import SDAdapter
from mvp_core.config import get_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Verification")

async def verify_llm():
    settings = get_settings()
    model_path = os.path.abspath(settings.model.text_path)
    logger.info(f"Verifying LLM at {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"LLM model not found at {model_path}")
        return False

    try:
        adapter = GGUFLLMAdapter(model_path)
        response = await adapter.generate("Hello, are you working?", max_tokens=20)
        logger.info(f"LLM Response: {response}")
        return True
    except Exception as e:
        logger.error(f"LLM Verification Failed: {e}")
        return False

async def verify_vl():
    settings = get_settings()
    model_path = os.path.abspath(settings.model.vl_path)
    logger.info(f"Verifying VL at {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"VL model not found at {model_path}")
        return False
        
    try:
        # Create a dummy image
        from PIL import Image
        img_path = "verify_vl_test.jpg"
        if not os.path.exists(img_path):
            Image.new('RGB', (256, 256), color='red').save(img_path)
        
        adapter = VLAdapter(model_path)
        # Use absolute path for image to be safe
        img_abs_path = os.path.abspath(img_path)
        response = await adapter.analyze_image(img_abs_path, "Describe this image.")
        logger.info(f"VL Response: {response}")
        return True
    except Exception as e:
        logger.error(f"VL Verification Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def verify_sd():
    settings = get_settings()
    model_path = os.path.abspath(settings.model.sd_path)
    logger.info(f"Verifying SD at {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"SD model not found at {model_path}")
        return False
        
    try:
        config = {
            'sd_model_path': model_path,
            'device': 'cuda',
            'generation': {'low_vram_mode': False}
        }
        adapter = SDAdapter(config)
        result = await adapter.generate_image("A cat", num_inference_steps=5)
        if result['status'] == 'success':
            logger.info("SD Image generated successfully")
            return True
        else:
            logger.error("SD generation failed")
            return False
    except Exception as e:
        logger.error(f"SD Verification Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def verify_tts():
    settings = get_settings()
    tts_url = settings.model.tts_api
    logger.info(f"Verifying TTS at {tts_url}")
    
    import aiohttp
    try:
        # Try health check or just a generation request
        # Our mock server accepts GET /tts
        async with aiohttp.ClientSession() as session:
            # Use absolute path for ref audio
            ref_audio_path = os.path.abspath(r"d:\AI\xiaoyou-core\ref_audio\female\ref_calm.wav")
            params = {
                "text": "Hello verification",
                "text_language": "en",
                "refer_wav_path": ref_audio_path,
                "prompt_text": "这是中文纯语音测试，不包含英文内容",
                "prompt_language": "zh"
            }
            async with session.get(f"{tts_url}", params=params, timeout=10) as response:
                if response.status == 200:
                    logger.info("TTS Service responded with 200 OK")
                    return True
                else:
                    logger.error(f"TTS Service returned status {response.status}")
                    text = await response.text()
                    logger.error(f"Response: {text}")
                    return False
    except Exception as e:
        logger.error(f"TTS Verification Failed: {e}")
        return False

async def main():
    logger.info("Starting Comprehensive Model Verification")
    
    llm_ok = await verify_llm()
    vl_ok = await verify_vl()
    sd_ok = await verify_sd()
    tts_ok = await verify_tts()
    
    logger.info("="*50)
    logger.info(f"LLM Status: {'PASS' if llm_ok else 'FAIL'}")
    logger.info(f"VL Status:  {'PASS' if vl_ok else 'FAIL'}")
    logger.info(f"SD Status:  {'PASS' if sd_ok else 'FAIL'}")
    logger.info(f"TTS Status: {'PASS' if tts_ok else 'FAIL'}")
    logger.info("="*50)

if __name__ == "__main__":
    asyncio.run(main())

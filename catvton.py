import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image
import numpy as np
import os
import gc
from datetime import datetime

class EfficientCatVTON:
    def __init__(self, device='cuda'):
        self.device = device
        self.width = 768
        self.height = 1024
        self.pipeline = None
        self.automasker = None
        self.mask_processor = None
        self.initialize_models()
    def _clear_memory(self):
        """Clear GPU memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        if self.automasker is not None:
            del self.automasker
            self.automasker = None
        torch.cuda.empty_cache()
        gc.collect()

    def initialize_models(self):
        """Initialize models with memory optimization"""
        if self.pipeline is not None:
            return

        self._clear_memory()

        # Download model weights
        repo_path = snapshot_download(repo_id="zhengchong/CatVTON")

        # Initialize pipeline with memory optimizations
        from model.pipeline import CatVTONPipeline
        self.pipeline = CatVTONPipeline(
            base_ckpt="booksforcharlie/stable-diffusion-inpainting",
            attn_ckpt=repo_path,
            attn_ckpt_version="mix",
            weight_dtype=torch.float16,
            use_tf32=True,
            device=self.device,
            skip_safety_check=True
        )

        # Initialize mask processor
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True
        )

        # Initialize automasker
        from model.cloth_masker import AutoMasker
        self.automasker = AutoMasker(
            densepose_ckpt=os.path.join(repo_path, "DensePose"),
            schp_ckpt=os.path.join(repo_path, "SCHP"),
            device=self.device
        )

    def _resize_and_crop(self, img, size):
        """Resize and crop image to target size"""
        aspect = img.size[0] / img.size[1]
        target_aspect = size[0] / size[1]

        if aspect > target_aspect:
            # Width is limiting factor
            new_height = size[1]
            new_width = int(new_height * aspect)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            left = (new_width - size[0]) // 2
            img = img.crop((left, 0, left + size[0], size[1]))
        else:
            # Height is limiting factor
            new_width = size[0]
            new_height = int(new_width / aspect)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            top = (new_height - size[1]) // 2
            img = img.crop((0, top, size[0], top + size[1]))
        return img

    def _resize_and_pad(self, img, size):
        """Resize and pad image to target size"""
        aspect = img.size[0] / img.size[1]
        target_aspect = size[0] / size[1]

        if aspect > target_aspect:
            # Width is limiting factor
            new_width = size[0]
            new_height = int(new_width / aspect)
        else:
            # Height is limiting factor
            new_height = size[1]
            new_width = int(new_height * aspect)

        img = img.resize((new_width, new_height), Image.LANCZOS)
        new_img = Image.new('RGB', size, (255, 255, 255))
        offset = ((size[0] - new_width) // 2, (size[1] - new_height) // 2)
        new_img.paste(img, offset)
        return new_img

    def try_on(self,
               person_image,
               cloth_image,
               mask_image=None,
               cloth_type="upper",
               num_inference_steps=30,
               guidance_scale=2.5,
               seed=42,
               output_dir="outputs"):
        """
        Perform virtual try-on

        Args:
            person_image (PIL.Image or str): Person image or path
            cloth_image (PIL.Image or str): Clothing image or path
            mask_image (PIL.Image or str, optional): Mask image or path
            cloth_type (str): Type of clothing ("upper", "lower", "overall")
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Guidance scale
            seed (int): Random seed (-1 for random)
            output_dir (str): Directory to save results

        Returns:
            PIL.Image: Generated try-on image and intermediate results
        """
        # Initialize models if needed
        # self.initialize_models()

        # Process input images
        if isinstance(person_image, str):
            person_image = Image.open(person_image).convert("RGB")
        if isinstance(cloth_image, str):
            cloth_image = Image.open(cloth_image).convert("RGB")
        if isinstance(mask_image, str):
            mask_image = Image.open(mask_image).convert("L")

        # Resize images
        person_image = self._resize_and_crop(person_image, (self.width, self.height))
        cloth_image = self._resize_and_pad(cloth_image, (self.width, self.height))

        # Process mask
        if mask_image is not None:
            mask = self._resize_and_crop(mask_image, (self.width, self.height))
            if len(np.unique(np.array(mask))) != 1:
                mask_arr = np.array(mask)
                mask_arr[mask_arr > 0] = 255
                mask = Image.fromarray(mask_arr)
        else:
            mask = self.automasker(person_image, cloth_type)['mask']

        mask = self.mask_processor.blur(mask, blur_factor=9)

        # Set generator for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed != -1 else None

        try:
            # Perform inference with memory optimization
            with torch.inference_mode():
                result_image = self.pipeline(
                    image=person_image,
                    condition_image=cloth_image,
                    mask=mask,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )[0]

            # Save results
            if output_dir:
                date_str = datetime.now().strftime("%Y%m%d%H%M%S")
                save_dir = os.path.join(output_dir, date_str[:8])
                os.makedirs(save_dir, exist_ok=True)

                # Create result grid
                from model.cloth_masker import vis_mask
                masked_person = vis_mask(person_image, mask)

                def image_grid(imgs, rows, cols):
                    w, h = imgs[0].size
                    grid = Image.new("RGB", size=(cols * w, rows * h))
                    for i, img in enumerate(imgs):
                        grid.paste(img, box=(i % cols * w, i // cols * h))
                    return grid

                grid = image_grid([person_image, masked_person, cloth_image, result_image], 1, 4)
                grid.save(os.path.join(save_dir, f"{date_str[8:]}.png"))

            return result_image

        except Exception as e:
            raise Exception(f"Error during inference: {str(e)}")
        finally:
            # Clear some memory after inference
            torch.cuda.empty_cache()
            gc.collect()

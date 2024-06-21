import torch
from diffusers import DiffusionPipeline

def generate_image_base(prompt):
    # Load the base model
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe.to("cuda")

    # Generate image
    images = pipe(prompt=prompt).images[0]
    images.save(f"{prompt.replace(' ', '_')}_base.png")

def generate_image_refined(prompt):
    # Load both base & refiner models
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    base.to("cuda")

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    refiner.to("cuda")

    # Define inference steps
    n_steps = 40
    high_noise_frac = 0.8

    # Generate latent image with base model
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent"
    ).images

    # Refine the latent image
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image
    ).images[0]

    image.save(f"{prompt.replace(' ', '_')}_refined.png")

if __name__ == "__main__":
    prompt = "A majestic lion jumping from a big stone at night"
    generate_image_base(prompt)
    generate_image_refined(prompt)

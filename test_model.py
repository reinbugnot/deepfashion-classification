import torch
from diffusers import StableDiffusionPipeline

model_path = "./fashion_out_trial_2"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

# image = pipe(prompt="A person wearing lattice sleeveless no dress v neckline cotton conventional clothes").images[0]
# image.save("./test_out/test_image.png")

image = pipe(prompt="A person wearing pleated sleeveless maxi length no neckline cotton conventional clothes").images[0]
image.save("./test_out/test_image_2.png")

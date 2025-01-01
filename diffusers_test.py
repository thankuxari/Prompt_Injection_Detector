from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
pipeline.to("cuda")

image  = pipeline("astronaut riding a horse in the desert").images[0]
image.save("image_of_astronaut_riding_a_horse_in_the_desert.png")
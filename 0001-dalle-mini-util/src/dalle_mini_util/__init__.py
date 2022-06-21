import math
import random

import numpy as np

from functools import partial

import jax
import jax.numpy as jnp

from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key

from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel

# from transformers import CLIPProcessor, FlaxCLIPModel

from PIL import Image

from tqdm.notebook import trange

def make_image_grid(images):
  rows = cols = math.floor(len(images)**0.5)
  width, height = images[0].size
  grid = Image.new('RGB', size=(cols*width, rows*height))
  for row in range(0, rows):
    y = row*height
    for col in range(0, cols):
      x = col*width
      image = images[y*cols+x]
      grid.paste(image, box=(x, y))
  return grid

class TextToImageModel:
  def __init__(
      self,
      dalle_model_path: str,
      vqgan_model_path: str,
  ):
    print("dalle_mini_util: trying to load dalle_model")
    self.dalle_model, self.dalle_params = DalleBart.from_pretrained(
      dalle_model_path, dtype=jnp.float16, _do_init=False)
    print("dalle_mini_util: loaded dalle_model")

    print("dalle_mini_util: trying to load vqgan_model")
    self.vqgan_model, self.vqgan_params = VQModel.from_pretrained(
      vqgan_model_path, _do_init=False)
    print("dalle_mini_util: loaded vqgan_model")

    print("dalle_mini_util: trying to replicate dalle_params")
    self.dalle_params = replicate(self.dalle_params)
    print("dalle_mini_util: replicated dalle_params")

    print("dalle_mini_util: trying to replicate vqgan_params")
    self.vqgan_params = replicate(self.vqgan_params)
    print("dalle_mini_util: replicated vqgan_params")

    print("dalle_mini_util: trying to load processor")
    self.processor = DalleBartProcessor.from_pretrained(
      dalle_model_path)
    print("dalle_mini_util: loaded processor")

    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
    def generate_latent_images(
        tokens,
        prng_key,
        params,
        top_k,
        top_p,
        temperature,
        condition_scale,
    ):
      return self.dalle_model.generate(
        **tokens,
        prng_key=prng_key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
      )

    @partial(jax.pmap, axis_name="batch")
    def decode_latent_images(indices, params):
      return self.vqgan_model.decode_code(indices, params=params)

    self._generate_latent_images = generate_latent_images
    self._decode_latent_images = decode_latent_images
    
  def generate(
      self,
      prompt: str,
      count: int = 4,
      prng_seed: int = 0,
  ):
    tokens = self.processor([prompt])
    tokens = replicate(tokens)

    # Jax has explicit effects for random numbers.
    prng_key = jax.random.PRNGKey(prng_seed)

    iterations = count**2
    iterations = max(iterations//jax.device_count(), 1)
    top_k = None
    top_p = None
    temperature = None
    condition_scale = 10.0

    for i in trange(iterations):
      # Jax has explicit effects for random numbers.
      prng_key, prng_subkey = jax.random.split(prng_key)

      # Sample latent image codes from the DALL-E frontend.
      latent_images = self._generate_latent_images(
        tokens,
        shard_prng_key(prng_subkey),
        self.dalle_params,
        top_k,
        top_p,
        temperature,
        condition_scale,
      )
      latent_images = latent_images.sequences[..., 1:]

      # Transform latent image codes in to pixel tensors.
      pixel_batch = self._decode_latent_images(
        latent_images, self.vqgan_params)
      pixel_batch = pixel_batch.clip(0, 1)
      pixel_batch = pixel_batch.reshape((-1, 256, 256, 3))

      # Convert pixel tensors to PIL images.
      for pixels in pixel_batch:
        buf = np.asarray(pixels*255, dtype=np.uint8)
        image = Image.fromarray(buf)
        yield image

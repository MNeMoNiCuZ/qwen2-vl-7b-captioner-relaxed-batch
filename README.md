# Qwen2 VL 7b Captioner Relaxed - Batch
This tool utilizes [Ertugrul/Qwen2-VL-7B-Captioner-Relaxed](https://huggingface.co/Ertugrul/Qwen2-VL-7B-Captioner-Relaxed) VLM, to caption image files in a batch.

Place all images you wish to caption in the /input directory and run `py batch.py`.

It's a very fast and fairly robust captioning model that can produce good outputs in 3 different levels of detail.

## Requirements
* Python 3.10 or 3.11.
  * It's been tested with 3.10 and 3.11

* Cuda 12.1.
  * It may work with other versions. Untested.
 
* GPU with ~20gb VRAM
  * It gets out of memory on a 3090 unless images are resized to ~1024x1024 (included in the script)

## Setup
1. Create a virtual environment. Use the included `venv_create.bat` to automatically create it. Use python 3.10 or above.
2. Install the libraries in requirements.txt. `pip install -r requirements.txt`. This is done by step 1 when asked if you use `venv_create`.
3. Install [Pytorch for your version of CUDA](https://pytorch.org/). It's only been tested with version 12.1 but may work with others.
4. Open `batch.py` in a text editor and edit any settings you want.


## How to use
1. Activate the virtual environment. If you installed with `venv_create.bat`, you can run `venv_activate.bat`.
2. Run `python batch.py` from the virtual environment.

This runs captioning on all images in the /input/-folder.

## Example prompts
You can edit `batch.py` in a text editor and edit the DEFAULT_PROMPT value to change the captioning prompt.
![example_image](https://github.com/user-attachments/assets/50f76cac-bb76-4f66-a99a-9b48fbe64633)

Here are some example prompts and their captions:
### Quick caption
In one short sentence, caption the key aspects of this image
> A whimsical green frog with a playful expression sits on a lily pad amidst a serene pond adorned with large, glossy leaves and delicate yellow flowers.
---
### Medium caption
In one medium sentence, caption the key aspects of this image
> A whimsical digital illustration depicts an anthropomorphic frog with human-like features sitting on a lily pad amidst a serene pond adorned with large, glossy green leaves and delicate yellow flowers. The scene is bathed in soft sunlight, creating a tranquil, almost dreamlike atmosphere.
---
### Long caption
Describe all details of this image
> This is a highly detailed, digital artwork depicting an anthropomorphic frog sitting on a lily pad in a serene pond environment. The frog has a light green skin with darker green patches and a prominent white belly. Its eyes are large and expressive, giving it a wise or contemplative look. The frog's mouth is slightly open, revealing a hint of its tongue and teeth, adding a touch of whimsy to the character. It wears a blue scarf around its neck, which adds a casual yet stylish element.  The background features lush green foliage, including large, round lily pads that float on calm water reflecting the sky above. There are also delicate yellow flowers emerging from some of the lilies, contributing to the tranquil ambiance. The lighting suggests a sunny day, with sunlight filtering through the leaves, creating a dappled effect on the ground and water surface. The overall color palette is vibrant, with shades of green dominating the scene, complemented by soft blues and yellows.   The texture of the lily pads appears smooth but slightly roughened at the edges, while the water looks still and reflective. The composition emphasizes the tranquility and natural beauty of the setting, making the viewer feel immersed in this peaceful landscape.
---
### Long list of tags
Caption this image using comma-separated tags
> frog, lily pads, pond, water, leaves, nature, green, smiling, cartoonish, illustration, summer, tranquility, outdoor, garden, whimsical, playful, calm, serene,
---
### Primary color
In one word: What is the primary color of this image.
> Green

# Image Generation Project

## Overview
This project explores three architectures to generate car images conditioned on textual descriptions:

**Diffusion Model**: A probabilistic approach for iterative noise removal to synthesize high-quality images.
**Multi-Vehicle Conditional GAN**: A Generative Adversarial Network conditioned on text descriptions to generate various types of vehicles.
**SUV and Sedan Conditional GAN**: A specialized GAN to generate images specifically for SUVs and sedans based on text prompts.

## Objectives
The primary goal is to explore and compare various generative models for car image synthesis, conditioned on textual descriptions. The project aims to:

Evaluate the performance of diffusion models and conditional GANs.
Generate high-quality, realistic car images based on textual inputs.

## Repository Structure
Image_generator/
│
├── Diffusion_model/
│   ├── diffusion-model.ipynb     # Training and generation script
│
├── Multi_vehicle_conditional_GAN/
│   ├── sample_images             # Images dataset
│   ├── multi_vehicle_GAN.ipynb   # Training and generation script
│
├── Suv_sedan_conditional_GAN/
│   ├── cars_dataset.zip          # Dataset used for training and text prompts
│   ├── suv_sedan_GAN.ipynb       # Training and generation script
│
└── README.md                     # Project documentation


## Models and Methodology
### 1. **Diffusion Model**
**Description:** Implements a diffusion process to iteratively denoise random noise into images guided by text embeddings.
**Inputs:** Text descriptions like "a red sedan car" or "a white SUV".
**Outputs:** Generated car images.
**Training:**
Encodes text using spaCy embeddings.
Conditions the denoising process with positional and textual embeddings.
**Results:** While training loss converges (see diffusion-model.ipynb), generated images still require fine-tuning for realistic outputs.
### 2. **Multi-Vehicle Conditional GAN**
**Description:** A GAN framework for generating images conditioned on car types (e.g., sedans, SUVs).
Components:
**Generator:** Produces images based on text embeddings.
**Discriminator:** Classifies real vs. fake images while considering the text condition.
**Training:**
Uses separate datasets for sedans and SUVs, stored in image_cars.zip and names_cars.zip.
**Outputs:** Conditional GAN output improves image fidelity compared to the diffusion model.
### 3. **SUV and Sedan Conditional GAN**
**Description:** A simplified GAN focused only on SUVs and sedans.
**Dataset:** Same dataset used for SUVs and sedans.
**Training:**
Text embeddings are used as conditional inputs.
**Outputs:** A focused approach to generating SUVs and sedans.

## Requirements:
**Python Libraries:**
torch
torchvision
matplotlib
numpy
PIL
tqdm
spacy
**Hardware:**
A GPU-enabled system for faster training and inference.
**Dataset:**
Ensure datasets are unzipped and paths are set correctly in each notebook.

## Training and Usage
**Setup:**
Clone the repository and install dependencies.
Ensure datasets (cars_dataset.zip, image_cars.zip, names_cars.zip) are unzipped in the respective folders.
**Run Models:**
Open the corresponding .ipynb files for each model.
Follow the instructions in the notebooks to train and evaluate models.
**Generate Images:**
Use the generate_images function in each notebook.
Example input: "a red sedan car" or "a silver SUV".
**Save Models:**
Models are saved in .pth files after training.

## Notes
**Diffusion Model**: 
Loss converges, but generated outputs appear pixelated or noisy.
Requires additional fine-tuning or hyperparameter adjustments.
**Multi-Vehicle Conditional GAN**: 
More robust and diverse outputs compared to the diffusion model.
Conditional embeddings improve text-to-image alignment.
**SUV and Sedan Conditional GAN**: 
Simplified architecture provides targeted results for SUVs and sedans.

## Future Work
Optimize training for higher resolution and realistic image generation.
Experiment with pretrained embeddings (e.g., CLIP) for improved text alignment.
Explore newer generative architectures like Stable Diffusion or StyleGAN.

## Contributors
**Archana Adhi**
**Havanitha Macha**
import argparse
import torch
from torch import nn
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoModelForCausalLM,
    AutoTokenizer
)
from PIL import Image


# ------------------------- #
# 1️⃣ Visual Encoder
# ------------------------- #
class VisualEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.visual = self.model.vision_model
        self.projection = self.model.visual_projection
        # Get output dimension dynamically
        self.output_dim = self.projection.out_features

    def forward(self, image_tensor):
        outputs = self.visual(image_tensor)
        pooled = outputs.pooler_output
        embeds = self.projection(pooled)
        return embeds

# ------------------------- #
# 2️⃣ Projection Head
# ------------------------- #
class ProjectionHead(nn.Module):
    def __init__(self, vision_dim, text_dim=4096, hidden_dim=2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_dim)
        )

    def forward(self, vision_emb):
        return self.proj(vision_emb)


# ------------------------- #
# 3️⃣ Vision-Language Model
# ------------------------- #
class VisionLanguageModel(nn.Module):
    def __init__(self, vision_model, projection_head, llm_name="microsoft/Phi-3-mini-4k-instruct"):
        super().__init__()
        self.vision_model = vision_model
        self.projection_head = projection_head
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)

    @torch.no_grad()
    def generate(self, image_tensor, prompt):
        img_emb = self.vision_model(image_tensor)
        img_emb_proj = self.projection_head(img_emb)

        visual_token = torch.mean(img_emb_proj, dim=1, keepdim=True)
        text_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)

        inputs_embeds = self.llm.get_input_embeddings()(text_inputs["input_ids"])
        fused_embeds = torch.cat([visual_token, inputs_embeds], dim=1)

        outputs = self.llm.generate(
            inputs_embeds=fused_embeds,
            max_new_tokens=120,
            temperature=0.2
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# ------------------------- #
# 4️⃣ CLI Interface
# ------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Mini Vision-Language Model CLI for Botanical Analysis")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--custom_prompt", help="Optional custom instruction (overrides default expert prompt)")
    args = parser.parse_args()

    # 🧠 Default Expert Prompt
    default_prompt = """
You are an expert botanist analyzing a banana shoot image.
Carefully examine the plant in the image and provide a structured JSON output with the following keys:

{
  "leaves_count": <integer>,
  "height_cm": <float>,
  "plant_to_root_ratio": <float>,
}

Guidelines:
- Measure height using the visible grid in the background (assume 1 cm per square).
- Count only fully formed leaves (ignore very small emerging ones).
- Estimate the plant-to-root ratio based on visible stem vs root length.
Return only the JSON output, nothing else.
"""

    # Load image
    image = Image.open(args.image)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    inputs = processor(images=image, return_tensors="pt")

    # Initialize model
    vision_model = VisualEncoder(model_name="openai/clip-vit-large-patch14")
    projection_head = ProjectionHead(vision_dim=vision_model.output_dim)
    vlm = VisionLanguageModel(vision_model, projection_head)

    # Decide which prompt to use
    final_prompt = args.custom_prompt if args.custom_prompt else default_prompt

    # Generate prediction
    result = vlm.generate(inputs["pixel_values"], final_prompt)
    print("\n🧠 Model Output:\n", result)


if __name__ == "__main__":
    main()

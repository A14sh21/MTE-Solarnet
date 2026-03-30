import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import gradio as gr
from huggingface_hub import InferenceClient

# --- MODEL DEFINITIONS
class Hswish(nn.Module):
    def forward(self, x): return x * F.relu6(x + 3, inplace=True) / 6

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = Hswish()
        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h

class SolarEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(SolarEfficientNet, self).__init__()
        self.base = models.efficientnet_b4(weights=None) # Don't need ImageNet weights here
        self.features = self.base.features
        self.ca = CoordinateAttention(1792, 1792)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1792, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.ca(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# --- MISSING GRAD-CAM CLASS
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output): self.activations = output
    def save_gradients(self, module, grad_input, grad_output): self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        _, pred_idx = torch.max(output, 1)
        output[:, pred_idx].backward()
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, pred_idx.item()

# --- INITIALIZATION ---
class_names = ['Bird_drop', 'Clean', 'Dusty', 'Electrical_damage', 'Physical_damage', 'Snow_covered']
device = torch.device("cpu") # Free tier uses CPU
model = SolarEfficientNet(len(class_names))
model.load_state_dict(torch.load('best_solar_effnet.pth', map_location=device))
model.eval()

# Secure Token Handling
hf_token = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=hf_token)
latest_diagnostic_data = {}

# --- VISION LOGIC ---
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def calculate_power_loss(cls_name, heatmap, confidence):
    """
    Statistically inferred power loss based on Spatial Extent and Literature Benchmarks.
    α (Alpha) values represent irradiance attenuation per defect type.
    """
    SEVERITY_COEFFICIENTS = {
        'Bird_drop': 0.85, 'Clean': 0.01, 'Dusty': 0.15, 
        'Electrical_damage': 0.95, 'Physical_damage': 0.60, 'Snow_covered': 0.90
    }

    # Extract spatial information from Grad-CAM
    threshold = 0.5
    area_ratio = np.sum(heatmap > threshold) / heatmap.size

    # Base Calculation: Spatial Ratio * Physical Severity * Model Certainty
    alpha = SEVERITY_COEFFICIENTS.get(cls_name, 0.10)
    inferred_loss = (area_ratio * alpha) * confidence * 100

    # Logarithmic scaling for hotspot detection (small area, high impact)
    if area_ratio > 0:
        inferred_loss = max(inferred_loss, (np.log1p(area_ratio) * alpha * 100))

    return round(float(np.clip(inferred_loss, 0.5, 98.0)), 2)

def full_research_diagnostic_pipeline(image):
    """
    Complete Research Pipeline:
    1. CNN Inference & Class Activation Mapping (Grad-CAM)
    2. Confidence Extraction (Inference)
    3. Empirical Power Loss Calculation (Spatial + Probabilistic)
    4. Metadata logging for RAG-lite Chatbot
    """
    global latest_diagnostic_data
    try:
        if image is None:
            return None, "Error: No image provided."

        # 1. Vision Preprocessing
        # Ensure image is in RGB for the model
        input_tensor = data_transforms(image).unsqueeze(0).to(device)
        
        # 2. Generate Heatmap & Class Prediction
        # Uses the features layer for spatial localization
        cam_extractor = GradCAM(model, model.features[-1])
        heatmap, pred_idx = cam_extractor.generate_heatmap(input_tensor)
        cls_name = class_names[pred_idx]
        
        # 3. CALCULATE STATISTICAL CONFIDENCE
        # We need this 'conf' value for the loss function and the paper metrics
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            conf = torch.max(probabilities).item() 
        
        # 4. CALCULATE POWER LOSS
        # This uses the literature-based coefficients and spatial area
        loss_val = calculate_power_loss(cls_name, heatmap, conf)
        
        # 5. UPDATE CHATBOT CONTEXT
        # Stores data for the RAG-lite hf_chat_fn to access
        latest_diagnostic_data = {
            "defect": cls_name, 
            "loss": loss_val, 
            "conf": conf,
            "timestamp": "2026-02-17"
        }

        # 6. IMAGE POST-PROCESSING (For Gradio Display)
        # Resize original image and heatmap to match (256x256)
        img_cv = cv2.resize(image, (256, 256))
        heatmap_rescaled = cv2.resize(heatmap, (256, 256))
        
        # Colorize the heatmap (JET represents 'heat' or 'attention')
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_rescaled), cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image (60% image, 40% heatmap)
        display_img = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)
        final_output = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        # 7. GENERATE SCIENTIFIC REPORT
        report = (
            f"DIAGNOSIS: {cls_name}\n"
            f"ESTIMATED POWER LOSS: {loss_val}%\n"
            f"CONFIDENCE: {conf*100:.2f}%\n"
            f"ANALYSIS: Localized via Grad-CAM (Spatial Inference)"
        )
        
        return final_output, report

    except Exception as e: 
        return None, f"Pipeline Error: {str(e)}"

# --- CHATBOT ---
# --- UPGRADED CHATBOT (RAG-lite + Prompt Engineering) ---
def hf_chat_fn(message, history):
    # Retrieve the latest data from the Vision Pipeline
    defect = latest_diagnostic_data.get('defect', 'None')
    loss = latest_diagnostic_data.get('loss', 0)
    conf = latest_diagnostic_data.get('conf', 0)

    # 1. Internal Knowledge Base (Simulating RAG retrieval)
    # This provides the LLM with specific "retrieved" facts from engineering manuals
    kb_protocols = {
        "Electrical_damage": "Critical: High risk of DC arcing. Check inverter logs and string voltage immediately.",
        "Bird_drop": "Maintenance: Hotspots may develop. Clean with deionized water when panels are cool.",
        "Physical_damage": "Safety: Potential moisture ingress. Inspect for micro-cracks using EL imaging.",
        "Dusty": "Operational: Gradual yield loss. Standard dry-brushing or robotic cleaning recommended.",
        "Snow_covered": "Environmental: Weight stress risk. Ensure mounting structures are secure; avoid manual scraping.",
        "Clean": "Nominal: No action required. Performance is within expected parameters."
    }
    
    protocol = kb_protocols.get(defect, "General inspection required.")

    # 2. Advanced Prompt Engineering (Persona + Chain-of-Thought)
    system_message = f"""
    You are a Senior Solar O&M (Operations & Maintenance) Engineer.
    
    CURRENT DIAGNOSTIC CONTEXT:
    - Target: Solar Photovoltaic Panel
    - Identified State: {defect}
    - Estimated Power Loss: {loss}%
    - AI Confidence Score: {conf*100:.2f}%
    - Technical Protocol: {protocol}

    INSTRUCTIONS:
    - Use a 'Chain-of-Thought' approach: First, analyze the impact of the {loss}% loss, then justify the safety protocol.
    - If confidence is below 85%, include a 'Responsible AI' disclaimer advising manual site verification.
    - Be technical, concise, and prioritize electrical safety.
    """

    messages = [{"role": "system", "content": system_message}]
    for msg in history:
        # Handling Gradio's history format (list of dicts)
        messages.append({"role": msg['role'], "content": msg['content']})
    
    messages.append({"role": "user", "content": message})
    
    response = ""
    # streaming response from Llama-3.2
    for chunk in client.chat_completion(
        model="meta-llama/Llama-3.2-3B-Instruct", 
        messages=messages, 
        max_tokens=600, 
        stream=True
    ):
        if chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
    return response

# --- GRADIO INTERFACE ---
with gr.Blocks() as demo:  # Removed 'theme' from here
    gr.Markdown("# MTE-SolarNet Inspector")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", type="numpy")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            output_heatmap = gr.Image(label="Localization")
            report_box = gr.Textbox(label="Report")
    
    # Removed 'type="messages"' as it's now standard
    gr.ChatInterface(fn=hf_chat_fn, title="Solar Panel Assistant")
    
    btn.click(fn=full_research_diagnostic_pipeline, inputs=input_image, outputs=[output_heatmap, report_box])

# Pass the theme to the launch method
demo.launch(theme=gr.themes.Soft())

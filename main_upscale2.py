import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from main3_simple import FreeFormFlow

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

def save_image(tensor, filename):
    tensor = tensor.squeeze(0).cpu().clamp(-1, 1).add(1).div(2)
    tensor = transforms.ToPILImage()(tensor)
    tensor.save(filename)

def upscale(model, image, scale_factor):
    model.eval()
    with torch.no_grad():
        # Ensure the input is a 4D tensor (batch, channels, height, width)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Move image to the same device as the model
        image = image.to(next(model.parameters()).device)
        
        # Encode and decode
        z = model.encode(image)
        output = model.decode(z)
        
        # Resize to the desired output size
        target_size = (int(image.shape[2] * scale_factor), int(image.shape[3] * scale_factor))
        output = F.interpolate(output, size=target_size, mode='bilinear', align_corners=False)
        
        return output.squeeze(0)  # Remove batch dimension if input was 3D

# Hauptfunktion zum Hochskalieren eines Bildes
def upscale_image(model_path, image_path, output_path, scale_factor):
    # Laden des trainierten Modells
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FreeFormFlow().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Laden und Vorbereiten des Bildes
    image = load_image(image_path).to(device)
    
    # Hochskalieren des Bildes
    upscaled_image = upscale(model, image, scale_factor)
    
    # Speichern des hochskalierten Bildes
    save_image(upscaled_image, output_path)
    
    print(f"Upscaled image saved to {output_path}")

# Beispielaufruf
if __name__ == "__main__":
    model_path = 'final_fff_upscaler.pth'
    image_path = 'img.jpg'
    output_path = 'img_output.png'
    scale_factor = 4  # für 4x Upscaling

    upscale_image(model_path, image_path, output_path, scale_factor)
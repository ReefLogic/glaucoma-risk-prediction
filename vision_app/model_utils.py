import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Sınıf isimleri (Eğitimdeki klasör isimlerine göre - Alfabetik)
# Eğer klasörlerin 'glaucoma' ve 'normal' ise:
CLASS_NAMES = ['glaucoma', 'normal']

def load_trained_model(model_path='saved_models/cnn_glaucoma.pth'):
    """Eğitilmiş .pth dosyasını yükler."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Mimariyi tekrar kur (ResNet18)
    model = models.resnet18(pretrained=False) # Ağırlıkları birazdan yükleyeceğiz
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Kayıtlı ağırlıkları yükle
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval() # Eğitim değil tahmin moduna al
        return model
    else:
        return None

def predict_single_image(model, image):
    """Tek bir PIL görüntüsünü alır ve tahmin sonucunu döndürür."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Görüntüyü hazırla
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0) # (1, 3, 224, 224) haline getir
    batch_t = batch_t.to(device)
    
    # Tahmin
    with torch.no_grad():
        output = model(batch_t)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    prob_glaucoma = probabilities[0].item() # 0. indeksin Glokom olduğunu varsayıyoruz (alfabetik)
    prob_normal = probabilities[1].item()
    
    # Emin olmak için sınıf isimlerini kontrol edelim
    # Klasör adları alfabetik sıralanır: ['glaucoma', 'normal'] -> 0: Glokom, 1: Normal
    
    return {
        "glaucoma_prob": prob_glaucoma,
        "normal_prob": prob_normal,
        "predicted_class": "Glokom" if prob_glaucoma > prob_normal else "Normal"
    }
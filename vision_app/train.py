import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time

def train_model():
    print("🚀 Eğitim Başlatılıyor... (Bu işlem bilgisayar hızına göre zaman alabilir)")
    
    # 1. AYARLAR
    data_dir = 'data/vision' # Resimlerin olduğu ana klasör
    save_path = 'saved_models/cnn_glaucoma.pth'
    
    # --- GÜNCELLEME: EPOCH SAYISI ARTIRILDI ---
    num_epochs = 15  # Model veriyi 15 kez baştan sona görecek
    batch_size = 32
    
    # Klasör Kontrolü
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    if not os.path.exists(data_dir):
        print(f"❌ HATA: '{data_dir}' klasörü bulunamadı!")
        return

    # 2. GÖRÜNTÜ İŞLEME (TRANSFORMS)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(), # Veri çoğaltma (Aynalama)
            transforms.RandomRotation(10),     # Hafif döndürme
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 3. VERİYİ YÜKLE
    try:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                          for x in ['train', 'val']}
    except FileNotFoundError:
        print("❌ HATA: 'train' veya 'val' klasörleri eksik. Lütfen data/vision klasörünü kontrol et.")
        return

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) 
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    print(f"📂 Sınıflar: {class_names}")
    # --- KRİTİK KONTROL: Hangi klasör hangi sayı? ---
    print(f"🔢 Sınıf İndeksleri (Harita): {image_datasets['train'].class_to_idx}")
    print(f"🖼️  Eğitim Görüntü Sayısı: {dataset_sizes['train']}")
    
    # 4. MODELİ KUR (Transfer Learning - ResNet18)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Kullanılan Cihaz: {device}")
    
    model = models.resnet18(pretrained=True)
    
    # Son katmanı değiştir (2 Sınıf: Glaucoma / Normal)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    # 5. EĞİTİM DÖNGÜSÜ
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'\nDevir (Epoch) {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    time_elapsed = time.time() - start_time
    print(f'\n⏱️ Eğitim tamamlandı: {time_elapsed // 60:.0f}dk {time_elapsed % 60:.0f}sn')

    # 6. MODELİ KAYDET
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model başarıyla kaydedildi: {save_path}")

if __name__ == '__main__':
    train_model()
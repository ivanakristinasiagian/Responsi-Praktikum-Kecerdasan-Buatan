# Animal Face Classifier — Optimized CNN

Proyek klasifikasi wajah hewan (Kucing, Anjing, Hewan Liar) menggunakan **Custom CNN** yang dibangun dari scratch dengan PyTorch.

## Deskripsi

Model ini dilatih menggunakan dataset **AFHQ (Animal Faces HQ)** dari Kaggle untuk mengklasifikasikan 3 kelas:
- **Cat** (Kucing)
- **Dog** (Anjing)
- **Wild** (Hewan Liar)

---

## Perbaikan yang Dilakukan

Dibandingkan model baseline, model ini dioptimasi dengan 5 perbaikan utama:

| # | Perbaikan | Detail |
|---|-----------|--------|
| 1 | **Split Data Seimbang** | 70% train / 20% val / 10% test (sebelumnya tidak seimbang) |
| 2 | **Data Augmentation** | Flip, rotasi ±15°, color jitter, random affine pada data training |
| 3 | **Batch Size Lebih Besar** | Batch size 32 untuk training lebih stabil |
| 4 | **BatchNorm + Dropout** | BatchNorm di setiap conv layer, Dropout 0.5 & 0.3 untuk cegah overfitting |
| 5 | **Scheduler + Early Stopping** | ReduceLROnPlateau (factor=0.5, patience=3) + Early stopping (patience=7) |

---

## ---

## 🚀 Cara Menjalankan

### Training ulang (Google Colab)
1. Buka file `.ipynb` di Google Colab
2. Jalankan semua cell secara berurutan
3. Download model `.pth` dari cell terakhir

### Inferensi lokal
```python
from PIL import Image
import torch
from torchvision import transforms

# Load model
model = Net()
model.load_state_dict(torch.load("animal_classifier_optimized.pth"))
model.eval()

# Prediksi
image = Image.open("gambar.jpg").convert("RGB")
# ... (lihat app.py untuk kode lengkap)
```

### Demo Online
👉 [Coba di Hugging Face Spaces](https://huggingface.co/spaces/vana21/animal_classfier)

---

## 📦 Dependencies
torch>=2.0.0
torchvision>=0.15.0
gradio>=5.0.0
Pillow>=9.0.0
numpy>=1.24.0
scikit-learn

---

## 📊 Dataset

[AFHQ — Animal Faces HQ](https://www.kaggle.com/datasets/andrewmvd/animal-faces) oleh Andrew MVD di Kaggle.

---

## 👩‍💻 Author

**Ivana Kristina Siagian**  
NIM: 241712021  
Tugas Responsi AI — A1

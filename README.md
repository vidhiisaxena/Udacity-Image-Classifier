# 🌿 Image Classifier

A command-line tool for predicting image classes using a pre-trained deep learning model in PyTorch. This Predict flower name from an image with predict.py along with the probability of that name, i.e. it returns the flower name and class probability

## 🚀 Features
- Loads a pre-trained model from a checkpoint.
- Processes an image and predicts its class.
- Supports **Top-K predictions**.
- Works on **CPU and GPU**.

---

## 📦 Installation
1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/image-classifier-cli.git
   cd image-classifier-cli
   ```
2. **Create a virtual environment (optional but recommended)**  
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```
3. **Install dependencies**  
   ```bash
   pip install torch torchvision numpy pillow argparse
   ```

---

## 🔧 Usage  

### **1. Save Your Trained Model**  
Make sure you have a trained model saved as `checkpoint.pth`. If you haven't saved it yet, use:
```python
import torch

model.class_to_idx = image_datasets['train'].class_to_idx

torch.save({
    'structure': 'alexnet',
    'hidden_layer1': 120,
    'dropout': 0.5,
    'epochs': 12,
    'state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx,
    'optimizer_dict': optimizer.state_dict()
}, 'checkpoint.pth')
```

### **2. Run the Prediction Script**
Run the command-line tool with:
```bash
python predict.py path/to/image.jpg checkpoint.pth --top_k 3
```
Example:
```bash
python predict.py flowers/test/37/image_03734.jpg checkpoint.pth --top_k 5
```

### **3. Example Output**
```
Predictions:
daisy: 82.45%
sunflower: 12.67%
tulip: 4.88%
```

---

## 🛠️ How It Works
### **predict.py**
- Loads the model checkpoint.
- Processes the input image (resizing, normalization).
- Runs inference and returns the **top K most probable classes**.
---

## 📝 Notes
- Ensure that your **image path is correct**.
- The model must be saved using `torch.save()` before running predictions.
- If using a **GPU**, modify `map_location='cpu'` to `map_location='cuda'` in `load_model()`.

---

## 🎯 Future Improvements
✅ Support GPU inference  
✅ Add JSON output  
✅ Improve formatting  

---

## 📜 License
This project is licensed under the MIT License.

---



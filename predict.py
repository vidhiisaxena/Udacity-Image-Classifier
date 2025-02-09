import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Function to load the model
def load_model(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = torch.hub.load('pytorch/vision:v0.10.0', checkpoint['structure'], pretrained=True)
    
    # Update classifier
    model.classifier[1] = torch.nn.Linear(9216, checkpoint['hidden_layer1'])
    model.classifier[4] = torch.nn.Linear(checkpoint['hidden_layer1'], len(checkpoint['class_to_idx']))
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    model.eval()
    return model

# Function to process an image
def process_image(image_path):
    image = Image.open(image_path)
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = preprocess(image)
    return image.numpy()

# Prediction function
def predict(image_path, model, topk=5):
    image = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0))
    
    with torch.no_grad():
        log_probs = model.forward(image)
    
    probs = torch.exp(log_probs)
    top_probs, top_labels = probs.topk(topk)
    
    top_probs = top_probs.numpy().flatten()
    top_labels = top_labels.numpy().flatten()
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[label] for label in top_labels]
    
    return top_probs, top_classes

# Command-line argument parsing
def main():
    parser = argparse.ArgumentParser(description="Image Classifier Command Line Tool")
    parser.add_argument("image_path", type=str, help="Path to the image")
    parser.add_argument("checkpoint", type=str, help="Path to the saved model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    
    args = parser.parse_args()
    
    model = load_model(args.checkpoint)
    probs, classes = predict(args.image_path, model, args.top_k)
    
    print("Predictions:")
    for i in range(len(classes)):
        print(f"{classes[i]}: {probs[i]*100:.2f}%")

if __name__ == "__main__":
    main()

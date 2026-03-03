from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("cats_dogs_resnet18.pth", map_location=device))
model = model.to(device)
model.eval()

classes = ["Cat", "Dog"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)

            prediction = classes[pred.item()]
            confidence = round(conf.item() * 100, 2)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
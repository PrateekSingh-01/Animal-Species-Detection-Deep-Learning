import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class AnimalClassifier:

    def __init__(self, model_path, device=None):

        # ====================================================
        # Device
        # ====================================================

        self.device = torch.device(
            device
            if device
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        # ====================================================
        # Load Checkpoint
        # ====================================================

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self.classes = checkpoint["classes"]
        self.class_to_idx = checkpoint["class_to_idx"]

        # ====================================================
        # Build ResNet18
        # ====================================================

        self.model = models.resnet18(
            weights=None
        )

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            len(self.classes)
        )

        # ====================================================
        # Load Trained Weights
        # ====================================================

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        # ====================================================
        # Image Transform
        # ====================================================

        self.transform = transforms.Compose([

            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

    # ========================================================
    # Prediction
    # ========================================================

    def predict(
        self,
        image,
        threshold=0.80
    ):

        # ----------------------------------------------------
        # Convert Image to RGB
        # ----------------------------------------------------

        image = image.convert("RGB")

        # ----------------------------------------------------
        # Transform Image
        # ----------------------------------------------------

        image_tensor = (
            self.transform(image)
            .unsqueeze(0)
            .to(self.device)
        )

        # ----------------------------------------------------
        # Model Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = self.model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_index = torch.max(
                probabilities,
                dim=1
            )

        # ----------------------------------------------------
        # Convert Tensor Values
        # ----------------------------------------------------

        confidence = confidence.item()

        predicted_index = predicted_index.item()

        # ----------------------------------------------------
        # Confidence Check
        # ----------------------------------------------------

        if confidence < threshold:

            return {
                "class": None,
                "confidence": confidence,
                "accepted": False
            }

        # ----------------------------------------------------
        # Accepted Prediction
        # ----------------------------------------------------

        return {
            "class": self.classes[predicted_index],
            "confidence": confidence,
            "accepted": True
        }
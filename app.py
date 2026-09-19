"""
DFU Detection & Wagner Grade Classification — Streamlit App

Pipeline:
Image
  -> YOLOv8 detection
  -> CIELAB normalization
  -> ResNet50 Wagner classification
  -> Grad-CAM++

The inference behavior is aligned with the Colab reference pipeline.
"""

import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from PIL import Image

from torchvision import models, transforms

from ultralytics import YOLO

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DFU Detection & Classification",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

YOLO_WEIGHTS_PATH = "models/dfu_yolo-3/weights/best.pt"

RESNET_AUG_PATH = "models/resnet50_v3_aug_best.pth"

RESNET_NOAUG_PATH = "models/resnet50_v3_noaug_best.pth"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = [
    "Grade 1",
    "Grade 2",
    "Grade 3",
    "Grade 4"
]

# Same as Colab
YOLO_CONFIDENCE = 0.40

# Same as Colab
BOX_PADDING = 5


# ============================================================
# EXACT COLAB VALIDATION TRANSFORM
# ============================================================

val_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ============================================================
# EXACT CIELAB FUNCTION FROM COLAB
# ============================================================

def cielab_normalize(image_bgr):

    lab = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    l_channel = np.uint8(
        lab[:, :, 0]
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    lab[:, :, 0] = clahe.apply(
        l_channel
    ).astype(np.float32)

    bgr_normalized = cv2.cvtColor(
        np.clip(
            lab,
            0,
            255
        ).astype(np.uint8),
        cv2.COLOR_LAB2BGR
    )

    return cv2.cvtColor(
        bgr_normalized,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# LOAD YOLO
# ============================================================

@st.cache_resource
def get_yolo_model():

    return YOLO(
        YOLO_WEIGHTS_PATH
    )


# ============================================================
# RESNET ARCHITECTURE
# ============================================================

def create_resnet50():

    model = models.resnet50(
        weights=None
    )

    # --------------------------------------------------------
    # THIS IS THE CLASSIFIER HEAD FROM YOUR TRAINED CHECKPOINT
    #
    # fc.1 = Linear(2048, 256)
    # fc.4 = Linear(256, 4)
    # --------------------------------------------------------

    model.fc = torch.nn.Sequential(

        torch.nn.Dropout(
            p=0.5
        ),

        torch.nn.Linear(
            2048,
            256
        ),

        torch.nn.ReLU(),

        torch.nn.Dropout(
            p=0.5
        ),

        torch.nn.Linear(
            256,
            4
        )
    )

    return model


# ============================================================
# LOAD TRAINED RESNET
# ============================================================

def load_resnet50_checkpoint(
    checkpoint_path,
    device
):

    model = create_resnet50()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # --------------------------------------------------------
    # Get state_dict
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint.state_dict()


    # --------------------------------------------------------
    # Remove DataParallel prefix if present
    # --------------------------------------------------------

    state_dict = {
        key.replace(
            "module.",
            "",
            1
        ): value

        for key, value in state_dict.items()
    }


    # --------------------------------------------------------
    # Load EXACT trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(
        device
    )

    model.eval()

    return model


# ============================================================
# CACHE RESNET MODELS
# ============================================================

@st.cache_resource
def get_aug_model():

    return load_resnet50_checkpoint(
        RESNET_AUG_PATH,
        DEVICE
    )


@st.cache_resource
def get_noaug_model():

    return load_resnet50_checkpoint(
        RESNET_NOAUG_PATH,
        DEVICE
    )


# ============================================================
# YOLO DETECTION
# ============================================================

def run_yolo_detection(
    yolo_model,
    image_rgb,
    confidence_threshold
):

    # --------------------------------------------------------
    # Convert RGB -> BGR
    # Colab uses BGR for YOLO
    # --------------------------------------------------------

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # Same YOLO call as Colab
    # --------------------------------------------------------

    result = yolo_model.predict(
        source=image_bgr,
        conf=confidence_threshold,
        save=False,
        verbose=False
    )[0]

    boxes = []

    if len(result.boxes) == 0:

        return boxes, image_rgb.copy()


    # --------------------------------------------------------
    # Get ALL detections for the UI
    #
    # Unlike the old app, we preserve all boxes.
    # --------------------------------------------------------

    for box in result.boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].cpu().numpy()
        )

        conf = box.conf.item()

        boxes.append(
            [
                x1,
                y1,
                x2,
                y2,
                conf
            ]
        )


    # --------------------------------------------------------
    # Draw all detections
    # --------------------------------------------------------

    annotated = image_rgb.copy()

    for i, box in enumerate(
        boxes,
        start=1
    ):

        x1, y1, x2, y2, conf = box

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            annotated,
            f"Wound {i}: {conf:.1%}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return (
        boxes,
        annotated
    )


# ============================================================
# CROP BOX
# ============================================================

def crop_box(
    image_rgb,
    box
):

    h, w = image_rgb.shape[:2]

    x1, y1, x2, y2 = map(
        int,
        box[:4]
    )

    # --------------------------------------------------------
    # SAME 5 PIXEL PADDING AS COLAB
    # --------------------------------------------------------

    x1 = max(
        0,
        x1 - BOX_PADDING
    )

    y1 = max(
        0,
        y1 - BOX_PADDING
    )

    x2 = min(
        w,
        x2 + BOX_PADDING
    )

    y2 = min(
        h,
        y2 + BOX_PADDING
    )

    return image_rgb[
        y1:y2,
        x1:x2
    ]


# ============================================================
# PREPARE CLASSIFIER INPUT
# ============================================================

def preprocess_for_classifier(
    normalized_rgb
):

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This follows the Colab:
    #
    # wound_resized = cv2.resize(...)
    #
    # tensor = val_transform(
    #     Image.fromarray(wound_resized)
    # )
    # --------------------------------------------------------

    wound_resized = cv2.resize(
        normalized_rgb,
        (224, 224)
    )

    tensor = val_transform(
        Image.fromarray(
            wound_resized
        )
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    return (
        tensor,
        wound_resized
    )


# ============================================================
# CLASSIFY
# ============================================================

def classify(
    model,
    input_tensor
):

    model.eval()

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probs = torch.softmax(
            output,
            dim=1
        )[0]

        pred_idx = torch.argmax(
            probs
        ).item()

        confidence = probs[
            pred_idx
        ].item()

    return (
        pred_idx,
        CLASS_NAMES[pred_idx],
        probs.cpu().numpy()
    )


# ============================================================
# GRAD-CAM++
# ============================================================

def generate_gradcam_overlay(
    model,
    input_tensor,
    base_float,
    class_idx
):

    # EXACT SAME TARGET LAYER AS COLAB

    target_layers = [
        model.layer4[-1]
    ]

    cam = GradCAMPlusPlus(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[
            ClassifierOutputTarget(
                class_idx
            )
        ]
    )[0]

    overlay = show_cam_on_image(
        base_float,
        grayscale_cam,
        use_rgb=True
    )

    return overlay


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Pipeline Settings"
)


model_choice = st.sidebar.radio(
    "ResNet50 model",
    options=[
        "With augmentation (CIELAB + skin darkening)",
        "Without augmentation"
    ],
    help="Compare the two models from your ablation study."
)


resnet_path = (
    RESNET_AUG_PATH
    if model_choice.startswith("With")
    else RESNET_NOAUG_PATH
)


# ------------------------------------------------------------
# Keep confidence control, but make the Colab value default
# ------------------------------------------------------------

conf_threshold = st.sidebar.slider(
    "YOLO confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.40,
    step=0.05
)


show_gradcam = st.sidebar.checkbox(
    "Show Grad-CAM++ heatmap",
    value=True
)


st.sidebar.markdown("---")


st.sidebar.caption(
    "Pipeline: YOLOv8 detection → crop → "
    "CIELAB (CLAHE on L*) normalization → "
    "ResNet50 Wagner grade classification → "
    "Grad-CAM++."
)


# ============================================================
# MAIN
# ============================================================

st.title(
    "Diabetic Foot Ulcer Detection & Wagner Grade Classification"
)


st.caption(
    "Two-stage pipeline with CIELAB-based skin tone "
    "bias mitigation, developed for a West African "
    "clinical context."
)


# ============================================================
# MODEL CHECK
# ============================================================

required_files = [
    YOLO_WEIGHTS_PATH,
    RESNET_AUG_PATH,
    RESNET_NOAUG_PATH
]


missing_files = [
    path
    for path in required_files
    if not os.path.exists(path)
]


if missing_files:

    st.error(
        "Model files not found. Place your trained "
        "weights at the following locations:"
    )

    for path in missing_files:

        st.code(path)

    st.stop()


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a foot image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    pil_image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_rgb = np.array(
        pil_image
    )


    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    yolo_model = get_yolo_model()

    if model_choice.startswith(
        "With"
    ):

        resnet_model = get_aug_model()

    else:

        resnet_model = get_noaug_model()


    # ========================================================
    # DETECTION
    # ========================================================

    with st.spinner(
        "Running detection..."
    ):

        boxes, annotated_image = (
            run_yolo_detection(
                yolo_model,
                image_rgb,
                conf_threshold
            )
        )


    # ========================================================
    # DISPLAY ORIGINAL + DETECTION
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Original image"
        )

        st.image(
            pil_image,
            use_container_width=True
        )


    with col2:

        st.subheader(
            f"Detection ({len(boxes)} found)"
        )

        st.image(
            annotated_image,
            use_container_width=True
        )


    # ========================================================
    # NO DETECTION
    # ========================================================

    if not boxes:

        st.warning(
            "No wound detected above the current "
            "confidence threshold. Try lowering "
            "the YOLO confidence threshold."
        )

        st.stop()


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    st.markdown("---")

    st.subheader(
        f"Classification results "
        f"({len(boxes)} wound"
        f"{'s' if len(boxes) != 1 else ''})"
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We keep your original UI's all-wounds behavior.
    #
    # But EVERY wound goes through the same preprocessing
    # as the Colab pipeline.
    # --------------------------------------------------------

    for wound_num, box in enumerate(
        boxes,
        start=1
    ):

        # ----------------------------------------------------
        # Crop
        # ----------------------------------------------------

        cropped_rgb = crop_box(
            image_rgb,
            box
        )


        # ----------------------------------------------------
        # CIELAB
        # Same BGR -> CIELAB -> RGB operation as Colab
        # ----------------------------------------------------

        cropped_bgr = cv2.cvtColor(
            cropped_rgb,
            cv2.COLOR_RGB2BGR
        )

        normalized_rgb = (
            cielab_normalize(
                cropped_bgr
            )
        )


        # ----------------------------------------------------
        # Classifier input
        # ----------------------------------------------------

        (
            input_tensor,
            wound_resized
        ) = preprocess_for_classifier(
            normalized_rgb
        )


        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        (
            pred_idx,
            pred_label,
            probs
        ) = classify(
            resnet_model,
            input_tensor
        )


        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        with st.expander(
            f"Wound {wound_num} — "
            f"{pred_label} "
            f"({probs[pred_idx] * 100:.1f}% confidence)",
            expanded=(
                wound_num == 1
            )
        ):

            col3, col4, col5 = st.columns(
                [1, 1, 1.3]
            )


            # ------------------------------------------------
            # Crop
            # ------------------------------------------------

            with col3:

                st.image(
                    cropped_rgb,
                    caption=(
                        "Cropped wound "
                        "(+5px padding)"
                    ),
                    use_container_width=True
                )


            # ------------------------------------------------
            # CIELAB
            # ------------------------------------------------

            with col4:

                st.image(
                    normalized_rgb,
                    caption="CIELAB normalized",
                    use_container_width=True
                )


            # ------------------------------------------------
            # Classification information
            # ------------------------------------------------

            with col5:

                st.metric(
                    "Predicted Wagner grade",
                    pred_label,
                    f"{probs[pred_idx] * 100:.1f}% confidence"
                )


                st.metric(
                    "Detection confidence",
                    f"{box[4] * 100:.1f}%"
                )


                prob_df = pd.DataFrame({

                    "Grade": [
                        "Grade 1",
                        "Grade 2",
                        "Grade 3",
                        "Grade 4"
                    ],

                    "Probability": probs

                }).set_index(
                    "Grade"
                )


                st.bar_chart(
                    prob_df
                )


            # =================================================
            # GRAD-CAM
            # =================================================

            if show_gradcam:

                st.markdown(
                    "**Grad-CAM++ explainability**"
                )

                st.caption(
                    "Highlights the regions the classifier "
                    "weighted most heavily for this prediction."
                )


                # ---------------------------------------------
                # EXACT COLAB BASE IMAGE
                # ---------------------------------------------

                base_for_overlay = np.array(
                    Image.fromarray(
                        wound_resized
                    ).resize(
                        (224, 224)
                    )
                )


                base_float = (
                    base_for_overlay.astype(
                        np.float32
                    ) / 255.0
                )


                # ---------------------------------------------
                # Grad-CAM
                # ---------------------------------------------

                heatmap_overlay = (
                    generate_gradcam_overlay(
                        resnet_model,
                        input_tensor,
                        base_float,
                        pred_idx
                    )
                )


                col6, col7 = st.columns(2)


                with col6:

                    st.image(
                        base_for_overlay,
                        caption=(
                            "Model input (224×224)"
                        ),
                        use_container_width=True
                    )


                with col7:

                    st.image(
                        heatmap_overlay,
                        caption=(
                            "Grad-CAM++ overlay"
                        ),
                        use_container_width=True
                    )


    # ========================================================
    # ABOUT
    # ========================================================

    st.markdown("---")


    with st.expander(
        "About this pipeline"
    ):

        st.markdown(
            """
- **Detection:** YOLOv8n trained on the Roboflow Foot Ulcer Detection dataset.
- **Detection threshold:** 0.40 by default, matching the Colab comparison.
- **Detection boxes:** all detected wounds are shown in the application.
- **Crop:** each detected wound receives the same 5-pixel padding used in Colab.
- **Preprocessing:** CIELAB color space conversion with CLAHE on the L* channel.
- **Classification:** ResNet50 predicting Wagner grades 1–4.
- **Classification preprocessing:** exact validation transform used in Colab.
- **Explainability:** Grad-CAM++ on `layer4[-1]`.
"""
        )
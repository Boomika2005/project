# report_generator.py
# Backend version with patient details in PDF

from __future__ import annotations
import os, uuid, datetime, tempfile
from pathlib import Path
from typing import Tuple, Optional
import io
import numpy as np
import cv2
from skimage import io as skio, measure, morphology
import tensorflow as tf
import keras


import numpy as np
import mysql.connector
from flask import Flask, request, jsonify, send_file
from skimage import io as skio, morphology
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader


import google.generativeai as genai

API_KEY = "AIzaSyCgd_bBl9vHKnU3BUXtYvhhT0pNyf6J6X8"
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ---------------------- CONFIG ----------------------
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models")).resolve()

UNET_PATH  = MODEL_DIR / "unet_final.h5"
VGG_PATH   = MODEL_DIR / "Brain_Tumer.h5"
PLANE_PATH = MODEL_DIR / "plane_classifier.h5"

SEG_OUTPUT_CHANNEL_TUMOR: Optional[int] = None

# 🔹 Define tumor classes (update if your training differs)
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
CLS_INPUT_SIZE = (224, 224)

PLANE_NAMES = ["axial", "sagittal", "coronal"]
PLANE_INPUT_SIZE = (224, 224)

# ------------------ MODEL LOADING -------------------
_unet = _cls = _plane = None

def _load_h5(path: Path):
    if not path.exists():
        return None
    try:
        return keras.models.load_model(str(path), compile=False, safe_mode=False)
    except TypeError:
        return keras.models.load_model(str(path), compile=False)

def load_models() -> Tuple[object, object, object]:
    global _unet, _cls, _plane
    if _unet is None:
        _unet = _load_h5(UNET_PATH)
        if _unet is None: raise FileNotFoundError(f"Missing UNet model: {UNET_PATH}")
    if _cls is None:
        _cls = _load_h5(VGG_PATH)
        if _cls is None: raise FileNotFoundError(f"Missing VGG model: {VGG_PATH}")
    if _plane is None:
        _plane = _load_h5(PLANE_PATH)
    return _unet, _cls, _plane

# --------------------- HELPERS ----------------------
def load_image_gray_from_path(path: Path) -> np.ndarray:
    img = skio.imread(str(path), as_gray=True).astype(np.float32)
    if img.max() > 1.0: img /= 255.0
    return img

def overlay_mask(img_gray: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    img_u8 = (img_gray * 255).clip(0,255).astype(np.uint8)
    rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    overlay = rgb.copy()
    red = np.zeros_like(rgb); red[...,2] = 255
    overlay[mask.astype(bool)] = (
        alpha*red[mask.astype(bool)] + (1-alpha)*rgb[mask.astype(bool)]
    ).astype(np.uint8)
    return overlay

def resize_keep_aspect(img: np.ndarray, target_hw: Tuple[int,int]):
    th, tw = target_hw
    h, w = img.shape[:2]
    s = min(th/h, tw/w)
    nh, nw = int(h*s), int(w*s)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw), dtype=resized.dtype)
    top, left = (th-nh)//2, (tw-nw)//2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, (s, top, left, (h, w))

def undo_resize_keep_aspect(mask_padded: np.ndarray, meta):
    s, top, left, (h, w) = meta
    nh, nw = int(h*s), int(w*s)
    crop = mask_padded[top:top+nh, left:left+nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)

def infer_mask_from_unet(model, img_gray: np.ndarray, seg_output_channel=None, thresh=0.5) -> np.ndarray:
    _, H, W, C = model.input_shape
    xpad, meta = resize_keep_aspect(img_gray, (H, W))
    x = xpad[None,...,None] if C==1 else np.repeat(xpad[...,None],3,axis=-1)[None,...]
    prob = model.predict(x, verbose=0)[0]
    if prob.ndim==3:
        prob = prob[...,0] if prob.shape[-1]==1 else prob[..., seg_output_channel or 0]
    prob_orig = undo_resize_keep_aspect(np.clip(prob,0,1).astype(np.float32), meta)
    mask = (prob_orig>=thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    return mask

def classify_with_vgg(model, img_path: Path, class_names=None, input_size=(224,224)):
    from tensorflow.keras.applications.vgg16 import preprocess_input
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Cannot read: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size, interpolation=cv2.INTER_LINEAR)
    x = preprocess_input(np.expand_dims(img_resized.astype(np.float32), 0))

    probs = model.predict(x, verbose=0)[0].astype(float)
    pred_idx = int(np.argmax(probs))

    # Auto-detect number of classes
    num_classes = len(probs)
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    # Ensure valid mapping
    predicted_label = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"

    return {
        "predicted_label": predicted_label,
        "probabilities": {
            class_names[i] if i < len(class_names) else f"class_{i}": float(probs[i]) 
            for i in range(num_classes)
        }
    }

def classify_plane(model, img_path: Path, class_names, input_size=(224,224)):
    if model is None: return {"predicted_label":"Not assessed","probabilities":{}}
    img = cv2.imread(str(img_path))
    if img is None: raise RuntimeError(f"Cannot read: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0
    probs = model.predict(np.expand_dims(img,0), verbose=0)[0].astype(float)
    pred_idx = int(np.argmax(probs))
    return {"predicted_label": class_names[pred_idx] if pred_idx<len(class_names) else f"class_{pred_idx}",
            "probabilities": {class_names[i] if i<len(class_names) else f"class_{i}": float(probs[i]) for i in range(len(probs))}}

def compute_mask_metrics(img_gray: np.ndarray, mask: np.ndarray):
    h,w = img_gray.shape[:2]
    area_px = int(mask.sum()); tumor_present = area_px>0
    centroid, laterality = None, "Not assessed"
    if tumor_present:
        props = measure.regionprops(mask.astype(np.uint8))[0]
        cy,cx = props.centroid
        centroid = [float(cx), float(cy)]
        laterality = "Right" if cx>(w/2.0) else "Left"
    return {"image_shape":{"height":h,"width":w},
            "tumor_present":tumor_present,
            "area_px":area_px,
            "centroid_xy":centroid,
            "laterality":laterality}

# ------------------- PDF RENDER ---------------------
# def render_pdf_bytes(facts: dict, overlay_img_path: Path) -> bytes:

#     buf = tempfile.SpooledTemporaryFile(max_size=5_000_000)
#     c = canvas.Canvas(buf, pagesize=A4); W,H=A4; m=15*mm; y=H-m

#     c.setFont("Helvetica-Bold",16)
#     c.drawString(m,y,"MRI Brain – AI-Assisted Pre-Report"); y-=10*mm
#     c.setFont("Helvetica",10)
#     c.drawString(m,y,f"Patient ID: {facts['patient']['patient_id']}    Name: {facts['patient']['name']}    Age: {facts['patient']['age']}    Sex: {facts['patient']['sex']}"); y-=6*mm
#     c.drawString(m,y,f"Modality: {facts['study']['modality']}    Date: {facts['study']['study_date']}"); y-=10*mm

#     f = facts["findings_extracted"]
#     c.setFont("Helvetica-Bold",12); c.drawString(m,y,"Findings"); y-=6*mm
#     c.setFont("Helvetica",10)
#     for L in [
#         f"Tumor present: {f['tumor_present']}",
#         f"Tumor type (AI): {facts['tumor_type']}",
#         f"Tumor size (px): {facts['tumor_size']}",
#         f"Plane axis: {facts['plane']['predicted_label']}",
#         f"Laterality: {f['laterality']}",
#         f"Centroid: {f['centroid_xy'] or 'Not assessed'}"
#     ]:
#         c.drawString(m,y,L); y-=5*mm

#     c.setFont("Helvetica-Bold",12); c.drawString(m,y,"Impression"); y-=6*mm
#     c.setFont("Helvetica",10)
#     if f["tumor_present"]:
#         c.drawString(m,y,f"- Imaging consistent with {facts['tumor_type']} on {facts['plane']['predicted_label']} plane."); y-=5*mm
#     else:
#         c.drawString(m,y,"- No tumor signal detected."); y-=5*mm

#     y-=6*mm
#     c.setFont("Helvetica-Bold",12); c.drawString(m,y,"AI Prediction Summary"); y-=6*mm
#     c.setFont("Helvetica",10); c.drawString(m,y,facts.get("result_summary","Not available")); y-=10*mm

#     img_reader = ImageReader(str(overlay_img_path)); img = skio.imread(str(overlay_img_path)); ih,iw = img.shape[:2]
#     draw_w=120*mm; draw_h=draw_w*(ih/iw)
#     if y-draw_h<m: c.showPage(); y=H-m
#     c.drawImage(img_reader,m,y-draw_h,width=draw_w,height=draw_h,mask='auto'); y=y-draw_h-6*mm

#     c.setFont("Helvetica",9); c.drawString(m,m,"Generated automatically. Review required by radiologist.")
#     c.showPage(); c.save(); buf.seek(0)
#     return buf.read()

# ----------------- MAIN ENTRYPOINT ------------------
# def build_report_pdf(image_bytes: bytes,
#                      result_summary: str,
#                      patient_id: str="Unknown",
#                      patient_name: str="NA",
#                      patient_age: str="NA",
#                      patient_sex: str="NA") -> tuple[bytes, dict]:
#     with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
#         tmp.write(image_bytes)
#         tmp_path = Path(tmp.name)
#     try:
#         seg_model, cls_model, plane_model = load_models()

#         # Segmentation
#         img_gray = load_image_gray_from_path(tmp_path)
#         mask = infer_mask_from_unet(seg_model, img_gray,
#                                     seg_output_channel=SEG_OUTPUT_CHANNEL_TUMOR,
#                                     thresh=0.5)
#         mask = morphology.remove_small_objects(mask.astype(bool), min_size=25).astype(np.uint8)

#         # Classification (tumor type)
#         classification = classify_with_vgg(cls_model, tmp_path, CLASS_NAMES, CLS_INPUT_SIZE)
#         tumor_type = classification["predicted_label"]

#         # ✅ Ensure tumor_type is mapped correctly
#         if tumor_type.startswith("class_"):
#             idx = int(tumor_type.split("_")[1])
#             if idx < len(CLASS_NAMES):
#                 tumor_type = CLASS_NAMES[idx]

#         # Plane
#         plane = classify_plane(plane_model, tmp_path, PLANE_NAMES, PLANE_INPUT_SIZE)

#         # Tumor size
#         findings = compute_mask_metrics(img_gray, mask)
#         tumor_size = findings["area_px"]

#         # Collect facts
#         facts = {
#             "case_id": str(uuid.uuid4())[:8],
#             "patient": {
#                 "patient_id": patient_id,
#                 "name": patient_name,
#                 "age": patient_age,
#                 "sex": patient_sex
#             },
#             "study": {"modality": "MRI",
#                       "study_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
#             "classification": classification,
#             "plane": plane,
#             "findings_extracted": findings,
#             "quality_flags": {
#                 "mask_empty": mask.sum() == 0,
#                 "image_nan": bool(np.isnan(img_gray).any())
#             },
#             "result_summary": result_summary,
#             "tumor_type": tumor_type,
#             "tumor_size": tumor_size
#         }

#         if facts["quality_flags"]["image_nan"]:
#             raise ValueError("Image contains NaNs.")

#         # Overlay
#         overlay = overlay_mask(img_gray, mask, alpha=0.35)
#         with tempfile.NamedTemporaryFile(suffix="_overlay.png", delete=False) as ov:
#             ov_path = Path(ov.name)
#             skio.imsave(str(ov_path), overlay)

#         pdf_bytes = render_pdf_bytes(facts, ov_path)
#         ov_path.unlink(missing_ok=True)

#         # return both PDF and metadata
#         return pdf_bytes, {
#             "tumor_type": tumor_type,
#             "tumor_size": tumor_size
#         }

#     finally:
#         tmp_path.unlink(missing_ok=True)

# def build_report_pdf(image_bytes: bytes,
#                      result_summary: str,
#                      patient_id: str="Unknown",
#                      patient_name: str="NA",
#                      patient_age: str="NA",
#                      patient_sex: str="NA") -> tuple[bytes, dict]:
#     with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
#         tmp.write(image_bytes)
#         tmp_path = Path(tmp.name)
#     try:
#         seg_model, cls_model, plane_model = load_models()

#         # Segmentation
#         img_gray = load_image_gray_from_path(tmp_path)
#         mask = infer_mask_from_unet(seg_model, img_gray,
#                                     seg_output_channel=SEG_OUTPUT_CHANNEL_TUMOR,
#                                     thresh=0.5)
#         mask = morphology.remove_small_objects(mask.astype(bool), min_size=25).astype(np.uint8)

#         # Classification (tumor type)
#         classification = classify_with_vgg(cls_model, tmp_path, CLASS_NAMES, CLS_INPUT_SIZE)
#         tumor_type = classification["predicted_label"]

#         # ✅ Map back from class index if needed
#         if tumor_type.startswith("class_"):
#             idx = int(tumor_type.split("_")[1])
#             if idx < len(CLASS_NAMES):
#                 tumor_type = CLASS_NAMES[idx]

#         # Plane classification
#         plane = classify_plane(plane_model, tmp_path, PLANE_NAMES, PLANE_INPUT_SIZE)

#         # Tumor size
#         findings = compute_mask_metrics(img_gray, mask)
#         tumor_size = findings["area_px"]

#         # Collect structured facts
#         facts = {
#             "case_id": str(uuid.uuid4())[:8],
#             "patient": {
#                 "patient_id": patient_id,
#                 "name": patient_name,
#                 "age": patient_age,
#                 "sex": patient_sex
#             },
#             "study": {
#                 "modality": "MRI",
#                 "study_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             },
#             "classification": classification,
#             "plane": plane,
#             "findings_extracted": findings,
#             "quality_flags": {
#                 "mask_empty": mask.sum() == 0,
#                 "image_nan": bool(np.isnan(img_gray).any())
#             },
#             "result_summary": result_summary,
#             "tumor_type": tumor_type,
#             "tumor_size": tumor_size
#         }

#         if facts["quality_flags"]["image_nan"]:
#             raise ValueError("Image contains NaNs.")

#         # ✅ Generate narrative summary using Gemini
#         prompt = f"""
#         Generate a clear, medically-oriented but patient-friendly report summary 
#         for the following brain MRI findings:

#         Patient ID: {patient_id}
#         Name: {patient_name}
#         Age: {patient_age}, Sex: {patient_sex}
#         Tumor Type: {tumor_type}
#         Tumor Size (px area): {tumor_size}
#         MRI Plane: {plane}
#         Extracted Findings: {findings}
#         AI Result Summary: {result_summary}

#         Please summarize in a professional radiology style (2-3 paragraphs).
#         """
#         gemini_response = gemini_model.generate_content(prompt)
#         narrative_summary = gemini_response.text.strip()

#         # Overlay with tumor mask
#         overlay = overlay_mask(img_gray, mask, alpha=0.35)
#         with tempfile.NamedTemporaryFile(suffix="_overlay.png", delete=False) as ov:
#             ov_path = Path(ov.name)
#             skio.imsave(str(ov_path), overlay)

#         # ✅ Merge Gemini summary into PDF facts
#         facts["gemini_summary"] = narrative_summary

#         pdf_bytes = render_pdf_bytes(facts, ov_path)
#         ov_path.unlink(missing_ok=True)

#         return pdf_bytes, {
#             "tumor_type": tumor_type,
#             "tumor_size": tumor_size,
#             "gemini_summary": narrative_summary
#         }

#     finally:
#         tmp_path.unlink(missing_ok=True)




def render_pdf_bytes(facts: dict, overlay_path: Path | None) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    m = 15 * mm
    y = H - m

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(m, y, "MRI Brain – AI-Assisted Pre-Report")
    y -= 10 * mm

    # Patient info
    c.setFont("Helvetica", 10)
    c.drawString(m, y,
                 f"Patient ID: {facts['patient']['patient_id']}    "
                 f"Name: {facts['patient']['name']}    "
                 f"Age: {facts['patient']['age']}    "
                 f"Sex: {facts['patient']['sex']}")
    y -= 6 * mm

    # Study info
    c.drawString(m, y,
                 f"Modality: {facts['study']['modality']}    "
                 f"Date: {facts['study']['study_date']}")
    y -= 10 * mm

    # Findings
    f = facts.get("findings_extracted", {})
    c.setFont("Helvetica-Bold", 12)
    c.drawString(m, y, "Findings")
    y -= 6 * mm

    c.setFont("Helvetica", 10)
    plane_info = facts.get("plane", {})
    predicted_plane = plane_info.get("predicted_label", "NA")

    findings_lines = [
        f"Tumor present: {f.get('tumor_present', True)}",
        f"Tumor type (AI): {facts.get('tumor_type','NA')}",
        f"Tumor size (px): {facts.get('tumor_size','NA')}",
        f"Plane axis: {predicted_plane}",
        f"Laterality: {f.get('laterality','NA')}",
        f"Centroid: {f.get('centroid_xy','Not assessed')}"
]

    for line in findings_lines:
        c.drawString(m, y, line)
        y -= 5 * mm

    # Impression
    c.setFont("Helvetica-Bold", 12)
    c.drawString(m, y, "Impression")
    y -= 6 * mm

    c.setFont("Helvetica", 10)
    if f.get("tumor_present", True):
        c.drawString(m, y,
                    f"- Imaging consistent with {facts.get('tumor_type','NA')} "
                    f"on {predicted_plane} plane.")

    else:
        c.drawString(m, y, "- No tumor signal detected.")
    y -= 10 * mm

    # AI Prediction Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(m, y, "AI Prediction Summary")
    y -= 6 * mm

    c.setFont("Helvetica", 10)
    c.drawString(m, y, facts.get("result_summary", "Not available"))
    y -= 10 * mm

    # # (Optional) Gemini Narrative
    # if facts.get("gemini_summary"):
    #     c.setFont("Helvetica-Bold", 12)
    #     c.drawString(m, y, "AI Narrative Summary")
    #     y -= 6 * mm
    #     c.setFont("Helvetica", 10)
    #     text = c.beginText(m, y)
    #     text.setFont("Helvetica", 10)
    #     for line in facts["gemini_summary"].split("\n"):
    #         text.textLine(line)
    #     c.drawText(text)
    #     y = text.getY() - 6 * mm

    # Overlay image (if provided)
    if overlay_path and Path(overlay_path).exists():
        img = skio.imread(str(overlay_path))
        ih, iw = img.shape[:2]
        draw_w = 120 * mm
        draw_h = draw_w * (ih / iw)
        if y - draw_h < m:  # new page if not enough space
            c.showPage()
            y = H - m
        img_reader = ImageReader(str(overlay_path))
        c.drawImage(img_reader, m, y - draw_h, width=draw_w, height=draw_h, mask='auto')
        y = y - draw_h - 6 * mm

    # Footer
    c.setFont("Helvetica", 9)
    c.drawString(m, m, "Generated automatically. Review required by radiologist.")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

def build_summary_pdf(metadata: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Summary Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Tumor Type: {metadata['tumor_type']}", styles["Normal"]))
    story.append(Paragraph(f"Tumor Size: {metadata['tumor_size']} px", styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Narrative:</b>", styles["Heading2"]))
    story.append(Paragraph(metadata["gemini_summary"], styles["Normal"]))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ---- Main Builder ----
def build_report_pdf(image_bytes: bytes,
                     result_summary: str,
                     patient_id: str = "Unknown",
                     patient_name: str = "NA",
                     patient_age: str = "NA",
                     patient_sex: str = "NA") -> tuple[bytes, dict]:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = Path(tmp.name)

    try:
        seg_model, cls_model, plane_model = load_models()

        # Segmentation
        img_gray = load_image_gray_from_path(tmp_path)
        mask = infer_mask_from_unet(seg_model, img_gray,
                                    seg_output_channel=1,
                                    thresh=0.5)
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=25).astype(np.uint8)

        # Classification
        classification = classify_with_vgg(
            cls_model, tmp_path,
            ["Glioma", "Meningioma", "Pituitary"], (224, 224)
        )
        tumor_type = classification["predicted_label"]

        # Plane classification
        plane = classify_plane(
            plane_model, tmp_path,
            ["Axial", "Coronal", "Sagittal"], (224, 224)
        )

        # Tumor size
        findings = compute_mask_metrics(img_gray, mask)
        tumor_size = findings["area_px"]

        # Narrative summary (Gemini)
        prompt = f"""
        Patient: {patient_name} ({patient_id}), Age {patient_age}, Sex {patient_sex}
        Tumor: {tumor_type}, Size {tumor_size}, Plane {plane}
        Result: {result_summary}
        """
        gemini_response = gemini_model.generate_content(prompt)
        narrative_summary = gemini_response.text.strip()

        # Overlay image
        overlay = overlay_mask(img_gray, mask, alpha=0.35)
        with tempfile.NamedTemporaryFile(suffix="_overlay.png", delete=False) as ov:
            ov_path = Path(ov.name)
            skio.imsave(str(ov_path), overlay)

        # ✅ Facts dictionary (now includes study + findings_extracted)
        facts = {
            "patient": {
                "patient_id": patient_id,
                "name": patient_name,
                "age": patient_age,
                "sex": patient_sex
            },
            "study": {
                "modality": "MRI",
                "study_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "classification": classification,
            "plane": plane,
            "findings_extracted": findings,
            "result_summary": result_summary,
            "tumor_type": tumor_type,
            "tumor_size": tumor_size,
            "gemini_summary": narrative_summary
        }

        # Build PDF
        pdf_bytes = render_pdf_bytes(facts, ov_path)
        ov_path.unlink(missing_ok=True)

        return pdf_bytes, {
            "tumor_type": tumor_type,
            "tumor_size": tumor_size,
            "gemini_summary": narrative_summary
        }

    finally:
        tmp_path.unlink(missing_ok=True)

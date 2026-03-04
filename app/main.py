from fastapi import FastAPI, UploadFile, File, HTTPException
from app.models import load_model
from app.utils import read_image, image_to_base64


app = FastAPI(title="Car Dent Detection API")

model = load_model()


@app.post("/detect")
async def detect_dent(
    file: UploadFile = File(...),
    conf: float = 0.25
):

    if not file.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="Invalid image")

    contents = await file.read()
    image = read_image(contents)

    # Inference
    results = model.predict(
        source=image,
        conf=conf,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    # Annotated image
    annotated = result.plot()

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            detections.append({
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist(),
                "class_id": int(box.cls[0])
            })

    return {
        "status": "success",
        "detections": detections,
        "image_base64": image_to_base64(annotated)
    }
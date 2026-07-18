from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import face_recognition
import cv2
import os
import tempfile

app = FastAPI()

ENCODING_FOLDER = r"E:\Anaconda\attCodeKiwi\encodings"

@app.post("/recognize")

async def recognize(
        class_name: str = Form(...),
        image: UploadFile = File(...)
):

    enc_file = os.path.join(
        ENCODING_FOLDER,
        f"encoded_faces_{class_name}.npy"
    )

    name_file = os.path.join(
        ENCODING_FOLDER,
        f"class_names_{class_name}.npy"
    )

    known_encodings = np.load(
        enc_file,
        allow_pickle=True
    )

    known_names = np.load(
        name_file,
        allow_pickle=True
    )

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    temp.write(await image.read())
    temp.close()

    img = face_recognition.load_image_file(
        temp.name
    )

    locations = face_recognition.face_locations(img)

    encodings = face_recognition.face_encodings(
        img,
        locations
    )

    present = []

    for encoding in encodings:

        matches = face_recognition.compare_faces(
            known_encodings,
            encoding,
            tolerance=0.55
        )

        distance = face_recognition.face_distance(
            known_encodings,
            encoding
        )

        best = np.argmin(distance)

        if matches[best]:
            name = known_names[best]

            if name not in present:
                present.append(name)

    absent = [
        x for x in known_names
        if x not in present
    ]

    return {
        "present": sorted(present),
        "absent": sorted(absent)
    }
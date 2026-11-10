import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model("mnist_cnn.keras")

def preprocess(img):
    if isinstance(img, dict):
        img = img.get("composite")
    if img is None:
        return None
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] == 4 and arr[..., 3].min() < 255:
        gray = arr[..., 3].astype("float32")          # transparent background
    elif arr.ndim == 3:
        gray = 255 - arr[..., :3].mean(axis=2)        # white background
    else:
        gray = 255 - arr.astype("float32")
    if gray.max() < 20:
        return None                                    # nothing drawn
    # crop to the digit, resize to 20x20, center in 28x28 (like MNIST)
    ys, xs = np.where(gray > 20)
    gray = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    im = Image.fromarray(gray.astype("uint8"))
    w, h = im.size
    scale = 20.0 / max(w, h)
    im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(im, ((28 - im.size[0]) // 2, (28 - im.size[1]) // 2))
    return np.array(canvas).astype("float32")[None, ..., None] / 255.0

def predict(img):
    x = preprocess(img)
    if x is None:
        return {}
    probs = model.predict(x, verbose=0)[0]
    return {str(i): float(probs[i]) for i in range(10)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(label="Draw a digit (0-9)"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Handwritten Digit Recognition (MNIST)",
    description="CNN trained on MNIST. Draw a big, clear digit and see the prediction.",
)
demo.launch()
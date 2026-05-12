import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATA_PATH = r"grid"
WEIGHTS_PATH = r"model/lipnet_epoch_50.h5"

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    oov_token="",
    invert=True
)

def create_model():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Input(shape=(75, 46, 140, 1)))

    model.add(tf.keras.layers.Conv3D(128, 3, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPool3D((1, 2, 2)))

    model.add(tf.keras.layers.Conv3D(256, 3, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPool3D((1, 2, 2)))

    model.add(tf.keras.layers.Conv3D(75, 3, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPool3D((1, 2, 2)))

    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))

    model.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, kernel_initializer="Orthogonal")
        )
    )
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, kernel_initializer="Orthogonal")
        )
    )
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(41, activation="softmax", kernel_initializer="he_normal"))

    return model

def load_trained_model():
    model = create_model()
    model.load_weights(WEIGHTS_PATH)
    return model

model = None
model_load_error = None

def try_load_model():
    global model, model_load_error
    try:
        model = load_trained_model()
        model_load_error = None
        print("Model loaded successfully")
    except Exception as e:
        model = None
        model_load_error = str(e)
        print("Model load failed:", e)

try_load_model()

def load_video_for_prediction(path):
    frames = []
    cap = cv2.VideoCapture(path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (140, 46))
        frames.append(frame)

    cap.release()

    frames = np.array(frames, dtype=np.float32)

    if len(frames) == 0:
        raise ValueError("No frames were read from the uploaded video.")

    frames = np.expand_dims(frames, axis=-1)

    max_frames = 75
    current_frames = frames.shape[0]

    if current_frames > max_frames:
        frames = frames[:max_frames]
    elif current_frames < max_frames:
        padding = np.zeros((max_frames - current_frames, 46, 140, 1), dtype=np.float32)
        frames = np.concatenate([frames, padding], axis=0)

    return frames

def load_alignment_text(path):
    with open(path, "r") as f:
        lines = f.readlines()

    words = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[2] != "sil":
            words.append(parts[2])

    return " ".join(words)

def find_original_sentence(uploaded_filename):
    if not os.path.exists(DATA_PATH):
        return None

    file_name = os.path.splitext(uploaded_filename)[0]

    for speaker in os.listdir(DATA_PATH):
        speaker_path = os.path.join(DATA_PATH, speaker)
        if not os.path.isdir(speaker_path):
            continue

        align_path = os.path.join(DATA_PATH, speaker, "align", f"{file_name}.align")
        if os.path.exists(align_path):
            return load_alignment_text(align_path)

    return None

def edit_distance(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[m][n]

def compute_metrics(original, predicted):
    original_words = original.split()
    predicted_words = predicted.split()
    original_chars = list(original)
    predicted_chars = list(predicted)

    word_ed = edit_distance(original_words, predicted_words)
    char_ed = edit_distance(original_chars, predicted_chars)

    wer = word_ed / max(len(original_words), 1)
    cer = char_ed / max(len(original_chars), 1)

    word_accuracy = max(0, 1 - wer) * 100
    char_accuracy = max(0, 1 - cer) * 100

    return {
        "word_edit_distance": word_ed,
        "char_edit_distance": char_ed,
        "wer": round(wer, 2),
        "cer": round(cer, 2),
        "word_accuracy": round(word_accuracy, 2),
        "char_accuracy": round(char_accuracy, 2)
    }

def get_status(word_accuracy):
    if word_accuracy >= 80:
        return "Excellent Match"
    elif word_accuracy >= 50:
        return "Good Match"
    return "Needs Improvement"

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    original_sentence = None
    metrics = None
    status = None
    video_filename = None
    error = None
    input_shape = None

    if request.method == "POST":
        try:
            if "video" not in request.files:
                error = "No video file uploaded."
            else:
                file = request.files["video"]

                if file.filename == "":
                    error = "Please choose a video file."
                else:
                    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
                    file.save(save_path)
                    video_filename = file.filename

                    if model is None:
                        raise ValueError(f"Model could not be loaded: {model_load_error}")

                    frames = load_video_for_prediction(save_path)
                    x = tf.expand_dims(frames, axis=0)
                    input_shape = tuple(x.shape)

                    yhat = model.predict(x, verbose=0)
                    input_len = np.ones(yhat.shape[0]) * yhat.shape[1]

                    decoded = tf.keras.backend.ctc_decode(
                        yhat,
                        input_length=input_len,
                        greedy=False,
                        beam_width=10
                    )[0][0].numpy()

                    prediction = tf.strings.reduce_join(
                        [num_to_char(word) for word in decoded[0]]
                    ).numpy().decode("utf-8")

                    original_sentence = find_original_sentence(file.filename)

                    if original_sentence is not None:
                        metrics = compute_metrics(original_sentence, prediction)
                        status = get_status(metrics["word_accuracy"])

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        original_sentence=original_sentence,
        metrics=metrics,
        status=status,
        video_filename=video_filename,
        error=error,
        model_loaded=(model is not None),
        model_load_error=model_load_error,
        input_shape=input_shape
    )

if __name__ == "__main__":
    app.run(debug=True)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
import os

# ============================================================
# ROAD VISION STUDIO | Image-based Computer Vision Simulator
# Modules: 7 (Pothole Detection removed)
# ============================================================

APP_BG = "#0b1220"
PANEL = "#111c2e"
PANEL_LIGHT = "#17263d"
ACCENT = "#38bdf8"
ACCENT_DARK = "#0284c7"
GREEN = "#22c55e"
TEXT = "#e5edf7"
MUTED = "#91a4bd"
BORDER = "#263750"

MODULES = [
    "Lane Detection",
    "Vehicle Detection and Counting",
    "Traffic Sign Detection",
    "Pedestrian Detection",
    "Edge Detection",
    "Road Segmentation",
    "Bird's-Eye View Transformation"
]

original_image = None
processed_image = None
image_path = None
yolo_model = None
original_photo = None
processed_photo = None


# ---------------------------- CV MODULES ----------------------------

def lane_detection(image):
    output = image.copy()
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.42), int(height * 0.55)),
        (int(width * 0.58), int(height * 0.55)),
        (int(width * 0.95), height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        masked_edges, 1, np.pi / 180,
        threshold=40, minLineLength=40, maxLineGap=100
    )

    left_lines, right_lines = [], []
    if lines is not None:
        # Flatten first, so OpenCV's output shape cannot break tuple unpacking.
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            if x2 == x1:
                continue
            slope = (y2 - y1) / float(x2 - x1)
            if abs(slope) < 0.4:
                continue
            (left_lines if slope < 0 else right_lines).append(
                (x1, y1, x2, y2)
            )

    def draw_average_line(group, color):
        if not group:
            return
        points = []
        for x1, y1, x2, y2 in group:
            points.extend([[x1, y1], [x2, y2]])
        points = np.asarray(points, dtype=np.float32)
        vx, vy, x, y = cv2.fitLine(
            points, cv2.DIST_L2, 0, 0.01, 0.01
        ).flatten()
        if abs(vy) < 0.0001:
            return
        y_bottom, y_top = height, int(height * 0.55)
        x_bottom = int(x + (y_bottom - y) * vx / vy)
        x_top = int(x + (y_top - y) * vx / vy)
        cv2.line(output, (x_bottom, y_bottom), (x_top, y_top), color, 7)

    draw_average_line(left_lines, (50, 220, 120))
    draw_average_line(right_lines, (50, 220, 120))
    cv2.putText(output, "LANE DETECTION", (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 220, 120), 2)
    return output, f"Lane line segments: {len(left_lines) + len(right_lines)}"


def load_yolo():
    global yolo_model
    if yolo_model is None:
        from ultralytics import YOLO
        print("Loading YOLOv8 Small model. First run may download weights...")
        yolo_model = YOLO("yolov8s.pt")
        print("YOLO loaded. Classes:", yolo_model.names)
    return yolo_model


def vehicle_detection(image):
    model = load_yolo()
    output = image.copy()
    results = model.predict(
        source=image, imgsz=1280, conf=0.10, iou=0.45,
        max_det=500, device="cpu", verbose=False
    )

    vehicle_names = {"car", "motorcycle", "bus", "truck"}
    counts = {name: 0 for name in vehicle_names}

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = model.names[class_id]
            print("Detected:", class_name, "Confidence:", round(confidence, 3))

            if class_name not in vehicle_names:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            counts[class_name] += 1
            cv2.rectangle(output, (x1, y1), (x2, y2), (50, 220, 120), 3)
            cv2.putText(
                output, f"{class_name}: {confidence:.2f}",
                (x1, max(24, y1 - 9)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (50, 220, 120), 2
            )

    total = sum(counts.values())
    summary = (
        f"Total: {total} | Cars: {counts['car']} | "
        f"Motorcycles: {counts['motorcycle']} | "
        f"Buses: {counts['bus']} | Trucks: {counts['truck']}"
    )
    return output, summary


def traffic_sign_detection(image):
    output = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    blue = cv2.inRange(hsv, np.array([90, 70, 50]), np.array([130, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([15, 70, 80]), np.array([40, 255, 255]))

    mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), blue)
    mask = cv2.bitwise_or(mask, yellow)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 250:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        x, y, w, h = cv2.boundingRect(contour)
        ratio = w / float(h)
        if len(approx) >= 3 and 0.55 <= ratio <= 1.45:
            count += 1
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 220, 255), 3)
            cv2.putText(output, "Sign Candidate", (x, max(22, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    return output, f"Traffic sign candidates: {count}"


def pedestrian_detection(image):
    model = load_yolo()
    output = image.copy()
    results = model.predict(
        source=image, imgsz=1280, conf=0.15, iou=0.45,
        max_det=500, device="cpu", verbose=False
    )
    count = 0
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            if model.names[class_id] != "person":
                continue
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            count += 1
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 150, 40), 3)
            cv2.putText(output, f"Person: {confidence:.2f}",
                        (x1, max(24, y1 - 9)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 150, 40), 2)
    return output, f"Pedestrians detected: {count}"


def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return output, f"Edge pixels: {cv2.countNonZero(edges):,}"


def road_segmentation(image):
    output = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = image.shape[:2]

    roi = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array([[
        (0, int(height * 0.55)), (width, int(height * 0.55)),
        (width, height), (0, height)
    ]], dtype=np.int32)
    cv2.fillPoly(roi, polygon, 255)

    mask = cv2.inRange(
        hsv, np.array([0, 0, 35]), np.array([180, 100, 210])
    )
    mask = cv2.bitwise_and(mask, roi)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    overlay = np.zeros_like(image)
    overlay[:] = (50, 220, 120)
    colored_mask = cv2.bitwise_and(overlay, overlay, mask=mask)
    output = cv2.addWeighted(output, 0.72, colored_mask, 0.28, 0)

    percentage = cv2.countNonZero(mask) / (height * width) * 100
    cv2.putText(output, "APPROXIMATE ROAD REGION", (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 120), 2)
    return output, f"Approximate road area: {percentage:.2f}%"


def bird_eye_view(image):
    height, width = image.shape[:2]
    src = np.float32([
        [width * 0.20, height * 0.60],
        [width * 0.80, height * 0.60],
        [width * 0.05, height * 0.95],
        [width * 0.95, height * 0.95]
    ])
    dst = np.float32([
        [width * 0.20, 0], [width * 0.80, 0],
        [width * 0.20, height], [width * 0.80, height]
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    output = cv2.warpPerspective(image, matrix, (width, height))
    cv2.putText(output, "BIRD'S-EYE VIEW", (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 220, 120), 2)
    return output, "Perspective transformation completed"


def run_module(image, module):
    functions = {
        "Lane Detection": lane_detection,
        "Vehicle Detection and Counting": vehicle_detection,
        "Traffic Sign Detection": traffic_sign_detection,
        "Pedestrian Detection": pedestrian_detection,
        "Edge Detection": edge_detection,
        "Road Segmentation": road_segmentation,
        "Bird's-Eye View Transformation": bird_eye_view
    }
    return functions[module](image)


# ---------------------------- UI HELPERS ----------------------------

def rounded_panel(parent, **kwargs):
    frame = tk.Frame(parent, bg=PANEL, highlightthickness=1,
                     highlightbackground=BORDER, **kwargs)
    return frame


def set_status(message, color=MUTED):
    status_label.config(text=message, fg=color)


def show_image(image, target_label, which):
    global original_photo, processed_photo
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    # Fit the image to the preview area without distorting its aspect ratio.
    max_w, max_h = 620, 410
    pil.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(pil)

    target_label.config(image=photo, text="")
    target_label.image = photo

    if which == "original":
        original_photo = photo
    else:
        processed_photo = photo


def upload_image():
    global original_image, processed_image, image_path
    path = filedialog.askopenfilename(
        title="Choose a road image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                   ("All files", "*.*")]
    )
    if not path:
        return

    image = cv2.imread(path)
    if image is None:
        messagebox.showerror("Image error", "The selected image could not be opened.")
        return

    original_image = image
    processed_image = None
    image_path = path

    show_image(original_image, original_preview, "original")
    processed_preview.config(image="", text="Your processed result will appear here")
    processed_preview.image = None
    processed_photo = None

    file_name = os.path.basename(path)
    file_label.config(text=file_name)
    result_value.config(text="Ready to run", fg=TEXT)
    count_value.config(text="No analysis yet")
    time_value.config(text="--")
    set_status("Image loaded successfully. Choose a module and run the simulation.", GREEN)


def run_simulation():
    global processed_image

    if original_image is None:
        messagebox.showwarning("No image selected", "Upload a road image before running a module.")
        return

    module = module_combo.get()
    if module not in MODULES:
        messagebox.showwarning("Select a module", "Choose one of the available simulations.")
        return

    run_button.config(state="disabled", text="PROCESSING...")
    root.update_idletasks()
    set_status(f"Running {module}...", ACCENT)

    try:
        start = time.perf_counter()
        processed_image, summary = run_module(original_image.copy(), module)
        elapsed = time.perf_counter() - start

        show_image(processed_image, processed_preview, "processed")
        result_value.config(text=module, fg=TEXT)
        count_value.config(text=summary)
        time_value.config(text=f"{elapsed:.3f} seconds")
        set_status("Simulation completed successfully.", GREEN)

    except Exception as error:
        processed_image = None
        print("Simulation error:", repr(error))
        messagebox.showerror("Simulation error", str(error))
        set_status("Simulation failed. See the terminal for details.", "#fb7185")

    finally:
        run_button.config(state="normal", text="RUN SIMULATION")


def save_output():
    if processed_image is None:
        messagebox.showwarning("No output", "Run a simulation before saving the result.")
        return

    path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        initialfile="road_vision_output.jpg",
        filetypes=[("JPEG image", "*.jpg"), ("PNG image", "*.png")]
    )
    if not path:
        return

    if cv2.imwrite(path, processed_image):
        set_status(f"Output saved: {os.path.basename(path)}", GREEN)
        messagebox.showinfo("Saved", "Processed image saved successfully.")
    else:
        messagebox.showerror("Save error", "Could not save the output image.")


def reset_app():
    global original_image, processed_image, image_path
    global original_photo, processed_photo

    original_image = None
    processed_image = None
    image_path = None
    original_photo = None
    processed_photo = None

    original_preview.config(image="", text="Upload an image to preview it here")
    processed_preview.config(image="", text="Your processed result will appear here")
    original_preview.image = None
    processed_preview.image = None

    module_combo.set("Select a simulation")
    file_label.config(text="No image selected")
    result_value.config(text="Waiting for input", fg=TEXT)
    count_value.config(text="No analysis yet")
    time_value.config(text="--")
    set_status("Workspace reset. Upload an image to begin.", MUTED)


# ---------------------------- BUILD UI ----------------------------

root = tk.Tk()
root.title("Road Vision Studio | AI Road Analysis")
root.geometry("1440x900")
root.minsize(1120, 760)
root.configure(bg=APP_BG)

style = ttk.Style()
style.theme_use("clam")
style.configure(
    "Road.TCombobox",
    fieldbackground="#1b2b43",
    background="#1b2b43",
    foreground=TEXT,
    arrowcolor=TEXT,
    bordercolor=BORDER,
    lightcolor=BORDER,
    darkcolor=BORDER,
    padding=10
)
style.map(
    "Road.TCombobox",
    fieldbackground=[("readonly", "#1b2b43")],
    foreground=[("readonly", TEXT)]
)

# Header
header = tk.Frame(root, bg=APP_BG)
header.pack(fill="x", padx=30, pady=(24, 16))

brand_row = tk.Frame(header, bg=APP_BG)
brand_row.pack(fill="x")

logo = tk.Label(
    brand_row, text="RV", font=("Segoe UI", 16, "bold"),
    bg=ACCENT_DARK, fg="white", width=4, height=2
)
logo.pack(side="left", padx=(0, 13))

heading_box = tk.Frame(brand_row, bg=APP_BG)
heading_box.pack(side="left", fill="y")

tk.Label(
    heading_box, text="ROAD VISION STUDIO",
    font=("Segoe UI", 22, "bold"), bg=APP_BG, fg=TEXT
).pack(anchor="w")

tk.Label(
    heading_box, text="COMPUTER VISION  /  IMAGE-BASED ROAD ANALYSIS",
    font=("Segoe UI", 9, "bold"), bg=APP_BG, fg=MUTED
).pack(anchor="w", pady=(3, 0))

tk.Label(
    brand_row, text="7 ANALYSIS MODULES",
    font=("Segoe UI", 9, "bold"), bg=PANEL_LIGHT, fg=ACCENT,
    padx=14, pady=9
).pack(side="right", pady=8)

# Thin accent line
tk.Frame(root, bg=ACCENT_DARK, height=2).pack(fill="x", padx=30)

# Main controls
control = rounded_panel(root)
control.pack(fill="x", padx=30, pady=(18, 14))

control_inner = tk.Frame(control, bg=PANEL, padx=18, pady=16)
control_inner.pack(fill="x")

left_controls = tk.Frame(control_inner, bg=PANEL)
left_controls.pack(side="left", fill="x", expand=True)

tk.Label(
    left_controls, text="SIMULATION CONTROL",
    font=("Segoe UI", 9, "bold"), bg=PANEL, fg=MUTED
).pack(anchor="w", pady=(0, 9))

control_row = tk.Frame(left_controls, bg=PANEL)
control_row.pack(anchor="w")

def make_button(parent, text, command, bg, fg="white", width=16):
    return tk.Button(
        parent, text=text, command=command,
        font=("Segoe UI", 9, "bold"),
        bg=bg, fg=fg, activebackground=bg,
        activeforeground=fg, relief="flat", bd=0,
        padx=14, pady=12, width=width, cursor="hand2"
    )

upload_button = make_button(
    control_row, "＋  UPLOAD IMAGE", upload_image, ACCENT_DARK, width=17
)
upload_button.pack(side="left", padx=(0, 12))

module_box = tk.Frame(control_row, bg=PANEL)
module_box.pack(side="left", padx=(0, 12))

module_combo = ttk.Combobox(
    module_box, values=MODULES, state="readonly",
    style="Road.TCombobox", width=31, font=("Segoe UI", 10)
)
module_combo.set("Select a simulation")
module_combo.pack(ipady=2)

run_button = make_button(
    control_row, "▶  RUN SIMULATION", run_simulation, GREEN, fg="#062313", width=18
)
run_button.pack(side="left", padx=(0, 10))

save_button = make_button(
    control_row, "↓  SAVE OUTPUT", save_output, "#6d4bd1", width=15
)
save_button.pack(side="left", padx=(0, 10))

reset_button = make_button(
    control_row, "RESET", reset_app, "#293a53", width=9
)
reset_button.pack(side="left")

# Selected file info
file_info = tk.Frame(control_inner, bg=PANEL_LIGHT, padx=13, pady=10)
file_info.pack(side="right", padx=(15, 0), fill="y")

tk.Label(
    file_info, text="CURRENT IMAGE",
    font=("Segoe UI", 8, "bold"), bg=PANEL_LIGHT, fg=MUTED
).pack(anchor="w")

file_label = tk.Label(
    file_info, text="No image selected",
    font=("Segoe UI", 9), bg=PANEL_LIGHT, fg=TEXT,
    width=24, anchor="w"
)
file_label.pack(anchor="w", pady=(5, 0))

# Image previews
preview_area = tk.Frame(root, bg=APP_BG)
preview_area.pack(fill="both", expand=True, padx=30, pady=(0, 14))

def create_preview(parent, title, subtitle):
    card = rounded_panel(parent)
    top = tk.Frame(card, bg=PANEL, padx=16, pady=12)
    top.pack(fill="x")

    tk.Label(
        top, text=title, font=("Segoe UI", 11, "bold"),
        bg=PANEL, fg=TEXT
    ).pack(anchor="w")

    tk.Label(
        top, text=subtitle, font=("Segoe UI", 8),
        bg=PANEL, fg=MUTED
    ).pack(anchor="w", pady=(3, 0))

    image_holder = tk.Frame(card, bg="#0a1424", padx=8, pady=8)
    image_holder.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    image_label = tk.Label(
        image_holder,
        text="Upload an image to preview it here" if title == "ORIGINAL IMAGE"
             else "Your processed result will appear here",
        font=("Segoe UI", 10), bg="#0a1424", fg=MUTED,
        compound="center"
    )
    image_label.pack(fill="both", expand=True)
    return card, image_label

original_card, original_preview = create_preview(
    preview_area, "01  /  ORIGINAL IMAGE", "Input image"
)
original_card.pack(side="left", fill="both", expand=True, padx=(0, 8))

processed_card, processed_preview = create_preview(
    preview_area, "02  /  PROCESSED OUTPUT", "Selected module result"
)
processed_card.pack(side="left", fill="both", expand=True, padx=(8, 0))

# Results dashboard
dashboard = rounded_panel(root)
dashboard.pack(fill="x", padx=30, pady=(0, 12))

dash_header = tk.Frame(dashboard, bg=PANEL, padx=17, pady=11)
dash_header.pack(fill="x")

tk.Label(
    dash_header, text="ANALYSIS SUMMARY",
    font=("Segoe UI", 10, "bold"), bg=PANEL, fg=TEXT
).pack(side="left")

tk.Label(
    dash_header, text="LIVE RESULT PANEL",
    font=("Segoe UI", 8, "bold"), bg=PANEL, fg=ACCENT
).pack(side="right")

dash_body = tk.Frame(dashboard, bg=PANEL, padx=10, pady=12)
dash_body.pack(fill="x")

def metric_card(parent, title, initial, width=34):
    card = tk.Frame(parent, bg=PANEL_LIGHT, padx=14, pady=10,
                    highlightthickness=1, highlightbackground=BORDER)
    card.pack(side="left", fill="both", expand=True, padx=6)
    tk.Label(
        card, text=title, font=("Segoe UI", 8, "bold"),
        bg=PANEL_LIGHT, fg=MUTED
    ).pack(anchor="w")
    value = tk.Label(
        card, text=initial, font=("Segoe UI", 10, "bold"),
        bg=PANEL_LIGHT, fg=TEXT, anchor="w",
        justify="left", wraplength=width * 10
    )
    value.pack(anchor="w", pady=(7, 0))
    return value

result_value = metric_card(dash_body, "SELECTED MODULE", "Waiting for input")
count_value = metric_card(dash_body, "DETECTION / RESULT", "No analysis yet", 55)
time_value = metric_card(dash_body, "PROCESSING TIME", "--")

# Status bar
status_bar = tk.Frame(root, bg="#080e19", padx=30, pady=10)
status_bar.pack(fill="x", side="bottom")

status_label = tk.Label(
    status_bar, text="Ready. Upload a road image to begin.",
    font=("Segoe UI", 9), bg="#080e19", fg=MUTED, anchor="w"
)
status_label.pack(side="left")

tk.Label(
    status_bar, text="ROAD VISION STUDIO  •  OPENCV + YOLO",
    font=("Segoe UI", 8, "bold"), bg="#080e19", fg="#60748f"
).pack(side="right")

root.mainloop()

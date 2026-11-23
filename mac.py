import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
import re
import platform
from collections import Counter
from datetime import datetime

# --- Deteksi OS dan pilih backend kamera ---
SYSTEM = platform.system()
if SYSTEM == "Darwin":
    CAMERA_BACKEND = cv2.CAP_AVFOUNDATION
elif SYSTEM == "Windows":
    CAMERA_BACKEND = cv2.CAP_DSHOW
else:
    CAMERA_BACKEND = cv2.CAP_ANY

# --- Inisialisasi MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Indeks Landmark untuk Mata ---
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_EDGES = [33, 133, 159, 145]
RIGHT_EYE_EDGES = [362, 263, 386, 374]

LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# --- Model 3D Head Pose ---
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1)
], dtype=np.float64)

LANDMARK_IDS = [4, 152, 263, 33, 287, 57]

# --- Threshold Gaze ---
GAZE_H_THRESHOLD = 0.08
GAZE_V_THRESHOLD_DOWN = 0.04
GAZE_V_THRESHOLD_UP = 0.04

# --- Threshold Head Pose ---
HEAD_YAW_THRESHOLD = 28.0
HEAD_PITCH_DOWN_THRESHOLD = 22.0
HEAD_PITCH_UP_THRESHOLD = 20.0

# --- Kedipan & validitas mata ---
EAR_BLINK_THRESHOLD = 0.17
EAR_MIN_OPEN_THRESHOLD = 0.25
MIN_EYE_HEIGHT_RATIO = 0.010

# --- Toleransi waktu ---
BLINK_GRACE_SECONDS = 1.0
HEAD_POSE_STABLE_SECONDS = 0.4
GAZE_STABLE_SECONDS = 0.3
SMOOTHING_ALPHA = 0.3

WINDOW_NAME = 'Proctoring System (Press Q to Quit)'

# --- Folder hasil ---
RESULT_DIR = "result"


def get_head_pose(landmarks, frame_shape):
    h, w = frame_shape[:2]
    image_points = np.array([
        (landmarks[i].x * w, landmarks[i].y * h) for i in LANDMARK_IDS
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0

    pitch = np.degrees(x)
    yaw = np.degrees(y)
    roll = np.degrees(z)
    return -pitch, -yaw, roll


def eye_aspect_ratio(landmarks, eye_indices):
    pts = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
    vert1 = np.linalg.norm(pts[1] - pts[5])
    vert2 = np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    if horiz == 0:
        return 0.0
    return (vert1 + vert2) / (2.0 * horiz)


def deteksi_kedipan(landmarks):
    ear_l = eye_aspect_ratio(landmarks, LEFT_EYE_EAR)
    ear_r = eye_aspect_ratio(landmarks, RIGHT_EYE_EAR)
    ear = (ear_l + ear_r) / 2.0
    return ear < EAR_BLINK_THRESHOLD, ear


def hitung_rasio_iris(landmarks, frame_shape):
    h_frame = frame_shape[0]
    min_height = MIN_EYE_HEIGHT_RATIO * h_frame

    def calc(iris_ids, edge_ids):
        iris_x = np.mean([landmarks[i].x for i in iris_ids])
        iris_y = np.mean([landmarks[i].y for i in iris_ids])
        left_x = landmarks[edge_ids[0]].x
        right_x = landmarks[edge_ids[1]].x
        top_y = landmarks[edge_ids[2]].y
        bottom_y = landmarks[edge_ids[3]].y

        eye_width = right_x - left_x
        eye_height = bottom_y - top_y

        if eye_width <= 0 or eye_height <= 0:
            return None
        if eye_height * h_frame < min_height:
            return None

        h_ratio = (iris_x - left_x) / eye_width
        v_ratio = (iris_y - top_y) / eye_height
        return np.clip(h_ratio, 0.1, 0.9), np.clip(v_ratio, 0.1, 0.9)

    l = calc(LEFT_IRIS, LEFT_EYE_EDGES)
    r = calc(RIGHT_IRIS, RIGHT_EYE_EDGES)
    if l is None or r is None:
        return None
    return (l[0] + r[0]) / 2, (l[1] + r[1]) / 2


def klasifikasi_gaze(h_ratio, v_ratio, baseline_h, baseline_v):
    dh = h_ratio - baseline_h
    dv = v_ratio - baseline_v
    dv = -dv
    if dh < -GAZE_H_THRESHOLD:
        return "Left"
    if dh > GAZE_H_THRESHOLD:
        return "Right"
    if dv < -GAZE_V_THRESHOLD_UP:
        return "Up"
    if dv > GAZE_V_THRESHOLD_DOWN:
        return "Down"
    return "Center"


def klasifikasi_head(yaw, pitch):
    if yaw > HEAD_YAW_THRESHOLD:
        return "Turn Right"
    if yaw < -HEAD_YAW_THRESHOLD:
        return "Turn Left"
    if pitch < -HEAD_PITCH_DOWN_THRESHOLD:
        return "Down"
    if pitch > HEAD_PITCH_UP_THRESHOLD:
        return "Up"
    return "Normal"


def kalibrasi_gaze(cap, durasi=3.0):
    print(f"Kalibrasi gaze: silakan hadap layar normal selama {durasi:.0f} detik...")
    h_values, v_values = [], []
    start = time.time()
    while time.time() - start < durasi:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            blinking, ear = deteksi_kedipan(lm)
            if not blinking and ear >= EAR_MIN_OPEN_THRESHOLD:
                rasio = hitung_rasio_iris(lm, frame.shape)
                if rasio is not None:
                    h_values.append(rasio[0])
                    v_values.append(rasio[1])
        sisa = durasi - (time.time() - start)
        cv2.putText(frame, f"Kalibrasi... {sisa:.1f}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, "Hadap layar normal, jangan berkedip lama", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if h_values and v_values:
        baseline_h = float(np.mean(h_values))
        baseline_v = float(np.mean(v_values))
        print(f"Baseline gaze: H={baseline_h:.3f}, V={baseline_v:.3f} "
              f"(dari {len(h_values)} frame valid)")
        return baseline_h, baseline_v
    print("Kalibrasi gagal: wajah tidak terdeteksi. Menggunakan baseline default 0.5.")
    return 0.5, 0.5


def open_camera():
    print(f"Sistem operasi: {SYSTEM}")
    print(f"Backend kamera: {CAMERA_BACKEND}")

    cap = cv2.VideoCapture(0, CAMERA_BACKEND)
    if not cap.isOpened():
        print("KAMERA TIDAK BISA DIBUKA dengan backend utama, mencoba fallback...")
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("KAMERA TIDAK BISA DIBUKA.")
        if SYSTEM == "Darwin":
            print("Di macOS: buka System Settings > Privacy & Security > Camera,")
            print("pastikan Terminal/iTerm/VS Code diizinkan mengakses kamera,")
            print("lalu tutup dan buka kembali aplikasi terminalnya.")
        else:
            print("Tutup aplikasi Kamera/browser/Zoom/OBS, lalu jalankan ulang.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 20)
    return cap


def tentukan_nama_file(skenario_input):
    """Nama log: {input}.log, atau bebas.log jika kosong."""
    s = skenario_input.strip()
    if not s:
        return "bebas.log"
    safe = re.sub(r'[^A-Za-z0-9_\-]', '_', s)
    return f"{safe}.log"


def tulis_ringkasan_txt(path_txt, info):
    """Tulis ringkasan statistik ke .txt dan cetak ke terminal."""
    lines = []
    lines.append("=" * 60)
    lines.append(" RINGKASAN HASIL PROCTORING")
    lines.append("=" * 60)
    lines.append(f" Nama sesi        : {info['nama']}")
    lines.append(f" File log (.log)  : {info['log_path']}")
    lines.append(f" Waktu mulai      : {info['waktu_mulai']}")
    lines.append(f" Waktu selesai    : {info['waktu_selesai']}")
    lines.append(f" Durasi           : {info['durasi_detik']:.2f} detik")
    lines.append(f" Total frame      : {info['total_frame']}")
    lines.append(f" Rata-rata FPS    : {info['fps_efektif']:.2f}")
    lines.append("-" * 60)
    lines.append(" DISTRIBUSI STATUS")
    lines.append("-" * 60)
    if info['total_frame'] > 0:
        for status, cnt in info['counter_status'].most_common():
            pct = cnt / info['total_frame'] * 100
            lines.append(f"  {status:<28s} : {cnt:>5d} ({pct:5.1f}%)")
    else:
        lines.append("  (tidak ada frame tercatat)")
    lines.append("-" * 60)
    lines.append(" DISTRIBUSI GAZE CLASS")
    lines.append("-" * 60)
    if info['total_frame'] > 0:
        for gz, cnt in info['counter_gaze'].most_common():
            pct = cnt / info['total_frame'] * 100
            lines.append(f"  {gz:<28s} : {cnt:>5d} ({pct:5.1f}%)")
    lines.append("-" * 60)
    lines.append(" DISTRIBUSI HEAD POSE CLASS")
    lines.append("-" * 60)
    if info['total_frame'] > 0:
        for hp, cnt in info['counter_head'].most_common():
            pct = cnt / info['total_frame'] * 100
            lines.append(f"  {hp:<28s} : {cnt:>5d} ({pct:5.1f}%)")
    lines.append("-" * 60)
    lines.append(" AKUMULASI PERILAKU MENCURIGAKAN")
    lines.append("-" * 60)
    if info['total_frame'] > 0:
        pct_cur = info['frame_curiga'] / info['total_frame'] * 100
        lines.append(f"  Frame mencurigakan : {info['frame_curiga']} "
                     f"({pct_cur:.1f}%)")
        lines.append(f"  Frame normal       : {info['frame_normal']} "
                     f"({100 - pct_cur:.1f}%)")
    lines.append("=" * 60)

    text = "\n".join(lines)
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\n[INFO] Ringkasan disimpan di: {os.path.abspath(path_txt)}")


def hitung_ringkasan(nama, log_path, waktu_mulai, waktu_selesai,
                     total_frame, counter_status, counter_gaze,
                     counter_head, frame_curiga, frame_normal):
    durasi = (waktu_selesai - waktu_mulai).total_seconds()
    fps = total_frame / durasi if durasi > 0 else 0
    return {
        "nama": nama,
        "log_path": log_path,
        "waktu_mulai": waktu_mulai.strftime("%Y-%m-%d %H:%M:%S"),
        "waktu_selesai": waktu_selesai.strftime("%Y-%m-%d %H:%M:%S"),
        "durasi_detik": durasi,
        "total_frame": total_frame,
        "fps_efektif": fps,
        "counter_status": counter_status,
        "counter_gaze": counter_gaze,
        "counter_head": counter_head,
        "frame_curiga": frame_curiga,
        "frame_normal": frame_normal,
    }


def main():
    cap = open_camera()
    if cap is None:
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW_NAME, 100, 100)

    skenario = input("Masukan nama: ").strip()
    nama_sesi = skenario if skenario else "bebas"
    csv_path = tentukan_nama_file(skenario)

    # Pastikan folder result ada
    os.makedirs(RESULT_DIR, exist_ok=True)
    result_txt = os.path.join(RESULT_DIR, f"result-{nama_sesi}.txt")

    baseline_h, baseline_v = kalibrasi_gaze(cap, durasi=3.0)

    csv_file = open(csv_path, mode='w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp", "frame", "yaw", "pitch", "roll", "gaze",
                     "h_ratio", "v_ratio", "status", "face_detected",
                     "blink", "ear", "gaze_valid",
                     "head_class", "gaze_class", "eyes_closed_sec"])

    print(f"Menulis log ke: {csv_path}")
    print(f"Hasil evaluasi akan disimpan ke: {result_txt}")
    print("Tekan Q pada jendela video untuk berhenti.")
    print("Program berjalan... (jendela muncul di layar)")

    frame_count = 0
    fail_count = 0
    waktu_mulai = datetime.now()
    start_time = time.time()

    # Akumulator untuk ringkasan
    counter_status = Counter()
    counter_gaze = Counter()
    counter_head = Counter()
    frame_curiga = 0
    frame_normal = 0

    last_gaze = "Center"
    last_h = last_v = None
    eyes_closed_start = None
    head_deviation_start = None
    smooth_yaw = smooth_pitch = None

    gaze_candidate = "Center"
    gaze_candidate_start = None

    held_status = "Normal"
    held_color = (0, 255, 0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            fail_count += 1
            print(f"read() gagal ({fail_count}x), mencoba lagi...")
            if fail_count > 50:
                print("Terlalu banyak kegagalan, keluar.")
                break
            time.sleep(0.05)
            continue
        fail_count = 0

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        now = time.time() - start_time
        status_text = "Normal"
        color = (0, 255, 0)
        yaw = pitch = roll = None
        gaze_dir = last_gaze
        h_ratio = v_ratio = None
        face_detected = 0
        blink_flag = 0
        ear_value = 0.0
        gaze_valid = 0
        head_class = "Normal"
        gaze_class = "Center"
        eyes_closed_sec = 0.0
        gaze_ok = False

        if results.multi_face_landmarks:
            face_detected = 1
            landmarks = results.multi_face_landmarks[0].landmark

            pose = get_head_pose(landmarks, frame.shape)
            if pose:
                raw_pitch, raw_yaw, roll = pose
                if smooth_yaw is None:
                    smooth_yaw, smooth_pitch = raw_yaw, raw_pitch
                else:
                    smooth_yaw = SMOOTHING_ALPHA * raw_yaw + (1 - SMOOTHING_ALPHA) * smooth_yaw
                    smooth_pitch = SMOOTHING_ALPHA * raw_pitch + (1 - SMOOTHING_ALPHA) * smooth_pitch
                yaw, pitch = smooth_yaw, smooth_pitch

                cv2.putText(frame, f"Yaw: {yaw:.1f} Pitch: {pitch:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                head_class = klasifikasi_head(yaw, pitch)
                if head_class != "Normal":
                    if head_deviation_start is None:
                        head_deviation_start = now
                    elif now - head_deviation_start >= HEAD_POSE_STABLE_SECONDS:
                        status_text = f"Head {head_class}!"
                        color = (0, 0, 255)
                else:
                    head_deviation_start = None

            blinking, ear_value = deteksi_kedipan(landmarks)
            if blinking:
                blink_flag = 1
                if eyes_closed_start is None:
                    eyes_closed_start = now
                eyes_closed_sec = now - eyes_closed_start
            else:
                eyes_closed_start = None

            gaze_ok = (not blinking) and (ear_value >= EAR_MIN_OPEN_THRESHOLD)
            if gaze_ok:
                rasio = hitung_rasio_iris(landmarks, frame.shape)
                if rasio is not None:
                    h_ratio, v_ratio = rasio
                    gaze_valid = 1
                    gaze_class = klasifikasi_gaze(h_ratio, v_ratio, baseline_h, baseline_v)
                    gaze_dir = gaze_class
                    last_gaze = gaze_dir
                    last_h, last_v = h_ratio, v_ratio
                else:
                    gaze_class = "Center"
            else:
                gaze_class = "Center"
                gaze_candidate = "Center"
                gaze_candidate_start = None

            if gaze_ok and gaze_valid:
                if gaze_class == gaze_candidate:
                    if gaze_candidate_start is None:
                        gaze_candidate_start = now
                else:
                    gaze_candidate = gaze_class
                    gaze_candidate_start = now

                stable = (gaze_candidate_start is not None and
                          now - gaze_candidate_start >= GAZE_STABLE_SECONDS)
                if stable and gaze_candidate != "Center" and status_text == "Normal":
                    status_text = f"Looking {gaze_candidate}!"
                    color = (0, 0, 255)

            disp_h = h_ratio if h_ratio is not None else last_h
            disp_v = v_ratio if v_ratio is not None else last_v
            if disp_h is not None:
                if gaze_valid:
                    label = gaze_dir
                elif blink_flag:
                    label = f"{last_gaze} (blink)"
                else:
                    label = f"{last_gaze} (transisi)"
                cv2.putText(frame, f"Gaze: {label} (H:{disp_h:.2f} V:{disp_v:.2f})", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if blink_flag:
                cv2.putText(frame, f"BLINK (EAR:{ear_value:.2f}) {eyes_closed_sec:.1f}s",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            if eyes_closed_sec > BLINK_GRACE_SECONDS and status_text == "Normal":
                status_text = "Eyes Closed >1s!"
                color = (0, 0, 255)

        else:
            status_text = "No Face Detected"
            color = (0, 0, 255)

        if face_detected and (blink_flag or not gaze_ok):
            if eyes_closed_sec <= BLINK_GRACE_SECONDS:
                status_text = held_status
                color = held_color
        else:
            if face_detected and gaze_ok:
                held_status = status_text
                held_color = color

        cv2.putText(frame, f"Status: {status_text}", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        writer.writerow([
            f"{now:.3f}",
            frame_count,
            f"{yaw:.2f}" if yaw is not None else "",
            f"{pitch:.2f}" if pitch is not None else "",
            f"{roll:.2f}" if roll is not None else "",
            gaze_dir,
            f"{h_ratio:.4f}" if h_ratio is not None else "",
            f"{v_ratio:.4f}" if v_ratio is not None else "",
            status_text,
            face_detected,
            blink_flag,
            f"{ear_value:.3f}",
            gaze_valid,
            head_class,
            gaze_class,
            f"{eyes_closed_sec:.2f}"
        ])

        # --- Akumulasi statistik ---
        counter_status[status_text] += 1
        counter_gaze[gaze_class] += 1
        counter_head[head_class] += 1
        if status_text == "Normal":
            frame_normal += 1
        else:
            frame_curiga += 1

        frame_count += 1

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Bersihkan resource ---
    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

    # --- Tulis & cetak ringkasan .txt saat keluar ---
    waktu_selesai = datetime.now()
    info = hitung_ringkasan(
        nama=nama_sesi,
        log_path=os.path.abspath(csv_path),
        waktu_mulai=waktu_mulai,
        waktu_selesai=waktu_selesai,
        total_frame=frame_count,
        counter_status=counter_status,
        counter_gaze=counter_gaze,
        counter_head=counter_head,
        frame_curiga=frame_curiga,
        frame_normal=frame_normal,
    )
    tulis_ringkasan_txt(result_txt, info)

    print(f"Selesai. Total frame: {frame_count}.")
    print(f"Log (.log)  : {os.path.abspath(csv_path)}")
    print(f"Ringkasan   : {os.path.abspath(result_txt)}")


if __name__ == "__main__":
    main()
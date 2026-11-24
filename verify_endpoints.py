"""
API endpoint verification suite for the upgraded CNN Image Classifier.
Tests: Health, Classes, File Upload w/ Grad-CAM, URL prediction, and error handling.
"""

import requests
import json

BASE = "http://127.0.0.1:5000"

def run_tests():
    print("=" * 60)
    print("  Endpoint Verification Suite")
    print("=" * 60)

    # Test 1: Health
    r = requests.get(f"{BASE}/health")
    h = r.json()
    print(f"[1] Health: {r.status_code} | model_loaded={h.get('model_loaded')} | features={h.get('features')}")

    # Test 2: Classes
    r = requests.get(f"{BASE}/classes")
    print(f"[2] Classes: {r.status_code} | classes={r.json().get('classes')}")

    # Test 3: File upload - Bird (Grad-CAM should be returned)
    with open("dataset/birds/bald_eagle.jpg", "rb") as f:
        r = requests.post(f"{BASE}/predict", files={"image": ("bald_eagle.jpg", f, "image/jpeg")})
    d = r.json()
    gcam_len = len(d.get("gradcam_image", ""))
    print(f"[3] File Upload | status={r.status_code} | class={d.get('predicted_class')} | conf={d.get('confidence_percentage')} | gradcam_b64_len={gcam_len}")

    # Test 4: URL prediction - Reptile (Gila monster)
    payload = {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Gila_monster2.JPG/330px-Gila_monster2.JPG"}
    r = requests.post(f"{BASE}/predict", headers={"Content-Type": "application/json"}, data=json.dumps(payload))
    d = r.json()
    print(f"[4] URL Predict | status={r.status_code} | class={d.get('predicted_class')} | conf={d.get('confidence_percentage')} | gradcam_ok={bool(d.get('gradcam_image'))}")

    # Test 5: Invalid file extension
    r = requests.post(f"{BASE}/predict", files={"image": ("doc.txt", b"hello", "text/plain")})
    print(f"[5] Bad Ext     | status={r.status_code} | error={r.json().get('error')}")

    # Test 6: No file provided
    r = requests.post(f"{BASE}/predict", data={})
    print(f"[6] No File     | status={r.status_code} | error={r.json().get('error')}")

    # Test 7: Invalid URL protocol
    payload = {"url": "ftp://not-allowed.com/image.jpg"}
    r = requests.post(f"{BASE}/predict", headers={"Content-Type": "application/json"}, data=json.dumps(payload))
    print(f"[7] Bad Proto   | status={r.status_code} | error={r.json().get('error')}")

    # Test 8: Webcam Base64 snapshot simulation
    from PIL import Image
    import io, base64
    img = Image.open("dataset/reptiles/chameleon.jpg").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    payload = {"image_base64": b64}
    r = requests.post(f"{BASE}/predict", headers={"Content-Type": "application/json"}, data=json.dumps(payload))
    d = r.json()
    print(f"[8] Webcam B64  | status={r.status_code} | class={d.get('predicted_class')} | conf={d.get('confidence_percentage')} | gradcam_ok={bool(d.get('gradcam_image'))}")

    print("=" * 60)
    print("All tests complete!")

if __name__ == "__main__":
    run_tests()

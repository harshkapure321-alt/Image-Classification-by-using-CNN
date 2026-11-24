/**
 * CNN Image Classifier & AI Vision Lab
 * Supports: File Drag & Drop, Live Webcam, Image URL, Grad-CAM XAI View Switcher,
 * and the "Guess the Object" Live Webcam Scavenger Game.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Navigation Tabs
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  // Tab 1: File Dropzone
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");

  // Tab 2: Webcam Elements
  const webcamVideo = document.getElementById("webcamVideo");
  const cameraPlaceholder = document.getElementById("cameraPlaceholder");
  const webcamControls = document.getElementById("webcamControls");
  const startCamBtn = document.getElementById("startCamBtn");
  const stopCamBtn = document.getElementById("stopCamBtn");
  const flipCamBtn = document.getElementById("flipCamBtn");
  const captureBtn = document.getElementById("captureBtn");

  // Tab 3: URL Elements
  const imageUrlInput = document.getElementById("imageUrlInput");
  const fetchUrlBtn = document.getElementById("fetchUrlBtn");
  const samplePills = document.querySelectorAll(".sample-pill");

  // Tab 4: Scavenger Game Elements
  const gameTargetClass = document.getElementById("gameTargetClass");
  const gameScore = document.getElementById("gameScore");
  const gameStreak = document.getElementById("gameStreak");
  const gameHint = document.getElementById("gameHint");
  const gameVideo = document.getElementById("gameVideo");
  const gameCamOff = document.getElementById("gameCamOff");
  const startGameCamBtn = document.getElementById("startGameCamBtn");
  const gameScanBtn = document.getElementById("gameScanBtn");
  const gameSkipBtn = document.getElementById("gameSkipBtn");
  const gameAutoScan = document.getElementById("gameAutoScan");
  const gameScanStatus = document.getElementById("gameScanStatus");

  // Shared Preview & Results Elements
  const previewContainer = document.getElementById("previewContainer");
  const imagePreview = document.getElementById("imagePreview");
  const fileName = document.getElementById("fileName");
  const fileSize = document.getElementById("fileSize");
  const removeBtn = document.getElementById("removeBtn");
  const predictBtn = document.getElementById("predictBtn");
  const btnSpinner = document.getElementById("btnSpinner");
  const alertBox = document.getElementById("alertBox");
  const alertMessage = document.getElementById("alertMessage");
  const snapshotCanvas = document.getElementById("snapshotCanvas");

  // Grad-CAM View Mode Elements
  const viewModeBar = document.getElementById("viewModeBar");
  const viewOriginalBtn = document.getElementById("viewOriginalBtn");
  const viewGradcamBtn = document.getElementById("viewGradcamBtn");

  // Results Card Elements
  const resultsPlaceholder = document.getElementById("resultsPlaceholder");
  const loadingState = document.getElementById("loadingState");
  const resultsContent = document.getElementById("resultsContent");
  const topClassName = document.getElementById("topClassName");
  const topConfidenceVal = document.getElementById("topConfidenceVal");
  const topList = document.getElementById("topList");
  const confettiContainer = document.getElementById("confettiContainer");

  // State Variables
  let currentFile = null;
  let currentBase64 = null;
  let currentUrl = null;
  let originalImageSrc = null;
  let gradcamImageSrc = null;

  // Webcam Streams
  let webcamStream = null;
  let gameStream = null;
  let currentFacingMode = "user";

  // Game State
  let availableClasses = ["mammals", "birds", "reptiles"];
  let currentQuestTarget = "mammals";
  let score = 0;
  let streak = 0;
  let autoScanInterval = null;
  let isGameScanning = false;

  const CLASS_HINTS = {
    mammals: "Point at a dog, cat, plush toy, whale picture, human, or elephant photo!",
    birds: "Point at a bird photo, parrot, eagle picture, feather, or duck toy!",
    reptiles: "Point at a lizard, chameleon, crocodile picture, turtle, or snake photo!"
  };

  // ==========================================
  // 1. Initial Setup & Fetch Classes
  // ==========================================
  async function fetchAvailableClasses() {
    try {
      const resp = await fetch("/classes");
      if (resp.ok) {
        const data = await resp.json();
        if (data.classes && data.classes.length > 0) {
          availableClasses = data.classes;
        }
      }
    } catch (e) {
      console.warn("Could not fetch dynamic classes, using defaults:", e);
    }
    setRandomGameQuest();
  }
  fetchAvailableClasses();

  // ==========================================
  // 2. Navigation Tabs
  // ==========================================
  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");

      tabButtons.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(`tab${capitalize(targetTab)}`);
      if (targetPane) targetPane.classList.add("active");

      // Pause or stop webcam when switching away from webcam tab
      if (targetTab !== "webcam" && webcamStream) {
        stopWebcamStream();
      }

      // Stop auto-scan if switching away from game tab
      if (targetTab !== "game") {
        if (gameAutoScan.checked) {
          gameAutoScan.checked = false;
          clearInterval(autoScanInterval);
        }
        if (gameStream) {
          stopGameCamera();
        }
      }
    });
  });

  function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
  }

  // ==========================================
  // 3. Helper Functions: Alerts & View Reset
  // ==========================================
  function showAlert(msg, type = "error") {
    alertBox.className = `alert-box ${type}`;
    alertMessage.textContent = msg;
    alertBox.style.display = "flex";
  }

  function hideAlert() {
    alertBox.style.display = "none";
    alertMessage.textContent = "";
  }

  function formatBytes(bytes, decimals = 1) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
  }

  function resetResultsView() {
    resultsPlaceholder.style.display = "flex";
    loadingState.style.display = "none";
    resultsContent.style.display = "none";
    viewModeBar.style.display = "none";
    originalImageSrc = null;
    gradcamImageSrc = null;
  }

  function displayPreview(src, nameText, sizeText) {
    originalImageSrc = src;
    gradcamImageSrc = null;
    imagePreview.src = src;
    fileName.textContent = nameText;
    fileSize.textContent = sizeText;
    previewContainer.style.display = "flex";
    viewModeBar.style.display = "none";
    hideAlert();
    resetResultsView();
  }

  removeBtn.addEventListener("click", () => {
    currentFile = null;
    currentBase64 = null;
    currentUrl = null;
    fileInput.value = "";
    previewContainer.style.display = "none";
    viewModeBar.style.display = "none";
    hideAlert();
    resetResultsView();
  });

  // ==========================================
  // 4. Tab 1: File Upload & Drag/Drop
  // ==========================================
  dropzone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) handleSelectedFile(file);
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) handleSelectedFile(file);
  });

  function handleSelectedFile(file) {
    const allowed = [".jpg", ".jpeg", ".png"];
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!allowed.includes(ext) && !file.type.startsWith("image/")) {
      showAlert("Invalid format. Please upload a JPG, JPEG, or PNG image.");
      return;
    }
    if (file.size > 16 * 1024 * 1024) {
      showAlert("File is too large. Max size is 16 MB.");
      return;
    }

    currentFile = file;
    currentBase64 = null;
    currentUrl = null;

    const reader = new FileReader();
    reader.onload = (e) => {
      displayPreview(e.target.result, file.name, formatBytes(file.size));
    };
    reader.readAsDataURL(file);
  }

  // ==========================================
  // 5. Tab 2: Live Webcam Controls
  // ==========================================
  startCamBtn.addEventListener("click", startWebcamStream);
  stopCamBtn.addEventListener("click", stopWebcamStream);
  flipCamBtn.addEventListener("click", () => {
    currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
    startWebcamStream();
  });

  async function startWebcamStream() {
    hideAlert();
    if (webcamStream) {
      webcamStream.getTracks().forEach((t) => t.stop());
    }

    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: currentFacingMode, width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      webcamVideo.srcObject = webcamStream;
      cameraPlaceholder.style.display = "none";
      webcamControls.style.display = "flex";
    } catch (err) {
      console.error("Camera access failed:", err);
      showAlert("Could not access webcam. Please ensure camera permissions are enabled in your browser.");
    }
  }

  function stopWebcamStream() {
    if (webcamStream) {
      webcamStream.getTracks().forEach((t) => t.stop());
      webcamStream = null;
    }
    webcamVideo.srcObject = null;
    cameraPlaceholder.style.display = "flex";
    webcamControls.style.display = "none";
  }

  captureBtn.addEventListener("click", () => {
    if (!webcamVideo.videoWidth) {
      showAlert("Camera is not ready yet.");
      return;
    }

    snapshotCanvas.width = webcamVideo.videoWidth;
    snapshotCanvas.height = webcamVideo.videoHeight;
    const ctx = snapshotCanvas.getContext("2d");
    ctx.drawImage(webcamVideo, 0, 0);

    const dataUrl = snapshotCanvas.toDataURL("image/jpeg", 0.92);
    currentBase64 = dataUrl;
    currentFile = null;
    currentUrl = null;

    displayPreview(dataUrl, "webcam_snapshot.jpg", "Snapshot (Live)");
  });

  // ==========================================
  // 6. Tab 3: Image URL Input
  // ==========================================
  fetchUrlBtn.addEventListener("click", () => {
    const url = imageUrlInput.value.trim();
    if (!url) {
      showAlert("Please enter an image URL.");
      return;
    }
    loadUrlImage(url);
  });

  samplePills.forEach((pill) => {
    pill.addEventListener("click", () => {
      const url = pill.getAttribute("data-url");
      imageUrlInput.value = url;
      loadUrlImage(url);
    });
  });

  function loadUrlImage(url) {
    currentUrl = url;
    currentFile = null;
    currentBase64 = null;
    const urlBasename = url.split("/").pop().split("?")[0] || "remote_image.jpg";
    displayPreview(url, urlBasename, "Remote URL");
  }

  // ==========================================
  // 7. Tab 4: Scavenger Game Mode
  // ==========================================
  function setRandomGameQuest() {
    if (availableClasses.length === 0) return;
    const remaining = availableClasses.filter((c) => c !== currentQuestTarget);
    const chosen = remaining.length > 0 ? remaining[Math.floor(Math.random() * remaining.length)] : availableClasses[0];
    currentQuestTarget = chosen;
    gameTargetClass.textContent = currentQuestTarget.toUpperCase();
    gameHint.innerHTML = `<strong>Hint:</strong> ${CLASS_HINTS[currentQuestTarget] || "Point the camera at an example of " + currentQuestTarget}`;
    gameScanStatus.textContent = `Looking for ${currentQuestTarget.toUpperCase()}...`;
    gameScanStatus.style.color = "#38bdf8";
  }

  gameSkipBtn.addEventListener("click", () => {
    setRandomGameQuest();
  });

  startGameCamBtn.addEventListener("click", startGameCamera);

  async function startGameCamera() {
    hideAlert();
    try {
      gameStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      gameVideo.srcObject = gameStream;
      gameCamOff.style.display = "none";
      gameScanBtn.disabled = false;
    } catch (err) {
      console.error("Game camera error:", err);
      showAlert("Could not start camera for game mode. Ensure camera permissions are granted.");
    }
  }

  function stopGameCamera() {
    if (gameStream) {
      gameStream.getTracks().forEach((t) => t.stop());
      gameStream = null;
    }
    gameVideo.srcObject = null;
    gameCamOff.style.display = "flex";
    gameScanBtn.disabled = true;
  }

  gameScanBtn.addEventListener("click", () => {
    scanGameFrame();
  });

  gameAutoScan.addEventListener("change", (e) => {
    if (e.target.checked) {
      if (!gameStream) startGameCamera();
      autoScanInterval = setInterval(() => {
        if (!isGameScanning && gameStream) {
          scanGameFrame();
        }
      }, 1600);
    } else {
      clearInterval(autoScanInterval);
    }
  });

  async function scanGameFrame() {
    if (!gameVideo.videoWidth || isGameScanning) return;
    isGameScanning = true;
    gameScanStatus.textContent = "Scanning frame...";
    gameScanStatus.style.color = "#facc15";

    snapshotCanvas.width = gameVideo.videoWidth;
    snapshotCanvas.height = gameVideo.videoHeight;
    const ctx = snapshotCanvas.getContext("2d");
    ctx.drawImage(gameVideo, 0, 0);

    const b64 = snapshotCanvas.toDataURL("image/jpeg", 0.85);

    try {
      const resp = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: b64 }),
      });
      const data = await resp.json();

      if (!resp.ok) throw new Error(data.error || "Inference failed");

      // Check match against quest target
      const pred = (data.predicted_class || "").toLowerCase();
      const target = currentQuestTarget.toLowerCase();
      const conf = data.confidence || 0;

      if (pred === target && conf >= 0.65) {
        // MATCH!
        score += 100 + streak * 20;
        streak += 1;
        gameScore.textContent = score;
        gameStreak.textContent = streak;

        gameScanStatus.textContent = `MATCH! ${pred.toUpperCase()} (${data.confidence_percentage}) +100 PTS!`;
        gameScanStatus.style.color = "#4ade80";

        playVictoryChime();
        triggerConfetti();

        // Pause briefly before next quest
        setTimeout(() => {
          setRandomGameQuest();
          isGameScanning = false;
        }, 2200);
      } else {
        gameScanStatus.textContent = `Saw: ${pred.toUpperCase()} (${data.confidence_percentage}) - Keep looking!`;
        gameScanStatus.style.color = "#fb923c";
        isGameScanning = false;
      }
    } catch (err) {
      console.warn("Scan failed:", err);
      gameScanStatus.textContent = "Scan error, trying again...";
      gameScanStatus.style.color = "#f87171";
      isGameScanning = false;
    }
  }

  // Web Audio Synthesizer (Zero-dependency sound effect)
  function playVictoryChime() {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const notes = [523.25, 659.25, 783.99, 1046.5]; // C5, E5, G5, C6 arpeggio

      notes.forEach((freq, idx) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(freq, ctx.currentTime + idx * 0.1);
        gain.gain.setValueAtTime(0.2, ctx.currentTime + idx * 0.1);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + idx * 0.1 + 0.35);

        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(ctx.currentTime + idx * 0.1);
        osc.stop(ctx.currentTime + idx * 0.1 + 0.4);
      });
    } catch (e) {
      // Audio autoplay policy fallback
    }
  }

  // CSS Confetti Generator
  function triggerConfetti() {
    confettiContainer.innerHTML = "";
    const colors = ["#4f46e5", "#06b6d4", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"];
    for (let i = 0; i < 45; i++) {
      const p = document.createElement("div");
      p.className = "confetti-piece";
      p.style.left = `${Math.random() * 100}vw`;
      p.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
      p.style.animationDuration = `${1.2 + Math.random() * 1.5}s`;
      p.style.animationDelay = `${Math.random() * 0.3}s`;
      p.style.width = `${8 + Math.random() * 8}px`;
      p.style.height = `${10 + Math.random() * 12}px`;
      confettiContainer.appendChild(p);
    }
    setTimeout(() => {
      confettiContainer.innerHTML = "";
    }, 3000);
  }

  // ==========================================
  // 8. Predict Action (Shared)
  // ==========================================
  predictBtn.addEventListener("click", async () => {
    if (!currentFile && !currentBase64 && !currentUrl) {
      showAlert("Please choose, capture, or paste an image first.");
      return;
    }

    hideAlert();
    predictBtn.disabled = true;
    btnSpinner.style.display = "inline-block";

    resultsPlaceholder.style.display = "none";
    resultsContent.style.display = "none";
    loadingState.style.display = "flex";

    try {
      let response;
      if (currentFile) {
        const formData = new FormData();
        formData.append("image", currentFile);
        response = await fetch("/predict", { method: "POST", body: formData });
      } else if (currentBase64) {
        response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_base64: currentBase64 }),
        });
      } else if (currentUrl) {
        response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: currentUrl }),
        });
      }

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || `Server returned error (${response.status})`);
      }

      renderResults(data);
    } catch (err) {
      console.error("Classification error:", err);
      showAlert(err.message || "Classification failed. Please try again.");
      resetResultsView();
    } finally {
      predictBtn.disabled = false;
      btnSpinner.style.display = "none";
      loadingState.style.display = "none";
    }
  });

  // ==========================================
  // 9. Render Prediction & Grad-CAM Switcher
  // ==========================================
  function renderResults(data) {
    resultsPlaceholder.style.display = "none";
    loadingState.style.display = "none";
    resultsContent.style.display = "block";

    // Set Primary Result
    topClassName.textContent = data.predicted_class;
    topConfidenceVal.textContent = data.confidence_percentage;

    // Set Top 3 List
    topList.innerHTML = "";
    if (data.top_3 && data.top_3.length > 0) {
      data.top_3.forEach((item) => {
        const itemEl = document.createElement("div");
        itemEl.className = "top-item";

        itemEl.innerHTML = `
          <div class="top-item-header">
            <div>
              <span class="rank-badge rank-${item.rank}">#${item.rank}</span>
              <span class="item-class">${item.class}</span>
            </div>
            <span class="item-percentage">${item.percentage}</span>
          </div>
          <div class="item-progress-bar">
            <div class="item-progress-fill" style="width: 0%;"></div>
          </div>
        `;
        topList.appendChild(itemEl);

        setTimeout(() => {
          const fill = itemEl.querySelector(".item-progress-fill");
          if (fill) fill.style.width = `${Math.max(1, item.percentage_value)}%`;
        }, 50);
      });
    }

    // Handle Grad-CAM Heatmap
    if (data.gradcam_image) {
      gradcamImageSrc = data.gradcam_image;
      viewModeBar.style.display = "flex";
      viewOriginalBtn.classList.remove("active");
      viewGradcamBtn.classList.add("active");
      imagePreview.src = gradcamImageSrc; // Show Grad-CAM attention initially
    } else {
      viewModeBar.style.display = "none";
    }
  }

  // Grad-CAM Switcher Handlers
  viewOriginalBtn.addEventListener("click", () => {
    if (originalImageSrc) {
      imagePreview.src = originalImageSrc;
      viewOriginalBtn.classList.add("active");
      viewGradcamBtn.classList.remove("active");
    }
  });

  viewGradcamBtn.addEventListener("click", () => {
    if (gradcamImageSrc) {
      imagePreview.src = gradcamImageSrc;
      viewGradcamBtn.classList.add("active");
      viewOriginalBtn.classList.remove("active");
    }
  });
});

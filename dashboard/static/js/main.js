(function () {
  "use strict";

  function spawnParticles() {
    const container = document.getElementById("particles");
    if (!container) return;
    const count = window.innerWidth < 640 ? 14 : 26;
    for (let i = 0; i < count; i++) {
      const p = document.createElement("div");
      p.className = "particle";
      p.style.left = Math.random() * 100 + "%";
      p.style.bottom = "-10px";
      p.style.animationDuration = 14 + Math.random() * 18 + "s";
      p.style.animationDelay = Math.random() * 20 + "s";
      p.style.opacity = 0.3 + Math.random() * 0.5;
      container.appendChild(p);
    }
  }

  function initUploadDrop() {
    const drop = document.getElementById("upload-drop");
    const input = document.getElementById("image-input");
    const filenameLabel = document.getElementById("upload-filename");
    if (!drop || !input) return;

    input.addEventListener("change", () => {
      if (input.files && input.files[0]) {
        filenameLabel.textContent = "Selected: " + input.files[0].name;
      }
    });

    ["dragenter", "dragover"].forEach((evt) => {
      drop.addEventListener(evt, (e) => {
        e.preventDefault();
        drop.style.borderColor = "var(--sage)";
      });
    });
    ["dragleave", "drop"].forEach((evt) => {
      drop.addEventListener(evt, (e) => {
        e.preventDefault();
        drop.style.borderColor = "";
      });
    });
    drop.addEventListener("drop", (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        input.files = e.dataTransfer.files;
        filenameLabel.textContent = "Selected: " + e.dataTransfer.files[0].name;
      }
    });
  }

  function animateRiskNumber() {
    const el = document.querySelector(".risk-ring__number");
    if (!el) return;
    const target = parseFloat(el.getAttribute("data-target")) || 0;
    const duration = 1200;
    const start = performance.now();
    function tick(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.round(target * eased);
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  function initAnalysePage() {
    const form = document.getElementById("analyse-form");
    const progress = document.getElementById("scan-progress");
    if (form && progress) {
      form.addEventListener("submit", () => {
        progress.classList.add("active");
        const btn = document.getElementById("scan-btn");
        if (btn) { btn.disabled = true; btn.textContent = "Scanning…"; }
      });
    }
    animateRiskNumber();
  }

  document.addEventListener("DOMContentLoaded", () => {
    spawnParticles();
    initUploadDrop();
    animateRiskNumber();
  });

  window.WildGuard = { initAnalysePage };
})();

// ===== Backend base URL (adjust if needed) =====
const API_BASE = "http://127.0.0.1:5000";

// ===== Elements =====
const fileInput   = document.getElementById('fileInput');
const preview     = document.getElementById('preview');
const emptyState  = document.getElementById('emptyState');
const progress    = document.getElementById('progress');
const progressBar = document.getElementById('progressBar');
const predictBtn  = document.getElementById('predictBtn');
const genState    = document.getElementById('genState');
const downloadBtn = document.getElementById('downloadBtn');
const dropZone    = document.getElementById('dropZone');
const browseLink  = document.getElementById('browseLink');


const downloadReportBtn = document.getElementById("downloadReportBtn");
const downloadSummaryPdfBtn = document.getElementById("downloadSummaryPdfBtn");
// const downloadSummaryTxtBtn = document.getElementById("downloadSummaryTxtBtn");

let loadedImageDataURL = null;
let lastPdfBlobUrl = null;

// ===== Helpers =====
function resetUI() {
  if (preview) { preview.style.display = 'none'; preview.src = ''; }
  if (emptyState) emptyState.style.display = 'block';
  if (predictBtn) predictBtn.disabled = true;
  if (genState) genState.style.display = 'none';
  if (downloadBtn) { downloadBtn.style.display = 'none'; downloadBtn.removeAttribute('data-url'); }
  if (progress) progress.style.display = 'none';
  if (progressBar) progressBar.style.width = '0%';
  if (lastPdfBlobUrl) { URL.revokeObjectURL(lastPdfBlobUrl); lastPdfBlobUrl = null; }
  loadedImageDataURL = null;
}
resetUI();

function isJPG(file) {
  if (!file) return false;
  const typeOK = file.type === 'image/jpeg';
  const nameOK = /\.jpe?g$/i.test(file.name || '');
  return typeOK || nameOK;
}

function handleFile(file) {
  if (!file) return;
  if (!isJPG(file)) {
    alert('Please select a JPG/JPEG file.');
    resetUI();
    return;
  }

  if (progress) { progress.style.display = 'block'; progressBar.style.width = '0%'; }

  const reader = new FileReader();
  reader.onprogress = (e) => {
    if (e.lengthComputable && progressBar) {
      const pct = Math.min(100, Math.round((e.loaded / e.total) * 100));
      progressBar.style.width = pct + '%';
    }
  };
  reader.onloadstart = () => { if (progressBar) progressBar.style.width = '5%'; };
  reader.onload = (e) => {
    loadedImageDataURL = e.target.result;
    if (preview) preview.src = loadedImageDataURL;
    if (preview) {
      preview.onload = () => {
        if (emptyState) emptyState.style.display = 'none';
        preview.style.display = 'block';
        if (progressBar) progressBar.style.width = '100%';
        setTimeout(() => { if (progress) progress.style.display = 'none'; }, 200);
        if (predictBtn) predictBtn.disabled = false;
        if (genState) genState.style.display = 'none';
        if (downloadBtn) downloadBtn.style.display = 'none';
      };
    }
  };
  reader.onerror = () => { alert('Failed to load the image. Please try another JPG.'); resetUI(); };
  reader.readAsDataURL(file);
}

// ===== File input change =====
if (fileInput) {
  fileInput.addEventListener('change', () => {
    const file = fileInput.files && fileInput.files[0];
    handleFile(file);
  });
}

// ===== "browse" clickable =====
if (browseLink) {
  browseLink.addEventListener('click', (e) => {
    e.preventDefault(); e.stopPropagation();
    fileInput && fileInput.click();
  });
}

// ===== Upload box click + drag-drop =====
if (dropZone) {
  dropZone.addEventListener('click', (e) => {
    if (!e.target.closest('.upload-btn')) fileInput && fileInput.click();
  });

  ['dragenter','dragover'].forEach(evt =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault(); e.stopPropagation();
      dropZone.classList.add('dragover');
    })
  );
  ['dragleave','drop'].forEach(evt =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault(); e.stopPropagation();
      dropZone.classList.remove('dragover');
    })
  );
  dropZone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) {
      if (fileInput) {
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
      }
      handleFile(file);
    }
  });
}

// // ===== Call backend and get PDF =====
// if (predictBtn) {
//   predictBtn.addEventListener('click', async () => {
//     const file = fileInput && fileInput.files && fileInput.files[0];
//     if (!file) { alert('Please select an image first.'); return; }

//     // 🔹 Ensure patient is logged in
//     const patientId = localStorage.getItem("patient_id");
//     if (!patientId) {
//       alert("You must login first before prediction.");
//       return;
//     }

//     predictBtn.disabled = true;
//     const oldText = predictBtn.textContent;
//     predictBtn.textContent = 'Predicting…';

//     try {
//       const form = new FormData();
//       form.append('image', file);
//       form.append('patient_id', patientId); // ✅ send patient_id

//       const resp = await fetch(`${API_BASE}/predict`, {
//         method: 'POST',
//         body: form
//       });

//       if (!resp.ok) {
//         const maybeJson = await resp.json().catch(()=>null);
//         const msg = (maybeJson && maybeJson.error) ? maybeJson.error : `Server error (${resp.status})`;
//         throw new Error(msg);
//       }

//       const blob = await resp.blob();
//       if (lastPdfBlobUrl) URL.revokeObjectURL(lastPdfBlobUrl);
//       lastPdfBlobUrl = URL.createObjectURL(blob);

//       if (genState) genState.style.display = 'flex';
//       if (downloadBtn) {
//         downloadBtn.style.display = 'inline-block';
//         downloadBtn.setAttribute('data-url', lastPdfBlobUrl);
//       }
//     } catch (err) {
//       alert(`Prediction failed: ${err.message}`);
//     } finally {
//       predictBtn.textContent = oldText;
//       predictBtn.disabled = false;
//     }
//   });
// }

// // ===== Download the returned PDF =====
// if (downloadBtn) {
//   downloadBtn.addEventListener('click', () => {
//     const url = downloadBtn.getAttribute('data-url');
//     if (!url) return;
//     const a = document.createElement('a');
//     a.href = url;
//     a.download = 'Brain_Tumor_Report.pdf';
//     document.body.appendChild(a);
//     a.click();
//     a.remove();
//   });
// }



if (predictBtn) {
  predictBtn.addEventListener('click', async () => {
    const file = fileInput && fileInput.files && fileInput.files[0];
    if (!file) { alert('Please select an image first.'); return; }

    const patientId = localStorage.getItem("patient_id");
    if (!patientId) {
      alert("You must login first before prediction.");
      return;
    }

    predictBtn.disabled = true;
    const oldText = predictBtn.textContent;
    predictBtn.textContent = 'Predicting…';

    try {
      const form = new FormData();
      form.append('image', file);
      form.append('patient_id', patientId);

      const resp = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: form
      });

      if (!resp.ok) {
        const maybeJson = await resp.json().catch(()=>null);
        const msg = (maybeJson && maybeJson.error) ? maybeJson.error : `Server error (${resp.status})`;
        throw new Error(msg);
      }

      // 🔹 Unpack ZIP from backend
      const blob = await resp.blob();
      const zip = await JSZip.loadAsync(blob);
    

      // Save file Blobs for download
      const reportBlob = await zip.file(/Report.*\.pdf$/i)[0].async("blob");
      const summaryPdfBlob = await zip.file(/Summary.*\.pdf$/i)[0].async("blob");
      const summaryTxtBlob = await zip.file(/Summary.*\.txt$/i)[0].async("blob");

      // Create object URLs
      const reportUrl = URL.createObjectURL(reportBlob);
      const summaryPdfUrl = URL.createObjectURL(summaryPdfBlob);
      const summaryTxtUrl = URL.createObjectURL(summaryTxtBlob);

      // Attach to buttons
      downloadReportBtn.style.display = 'inline-block';
      downloadReportBtn.setAttribute('data-url', reportUrl);

      downloadSummaryPdfBtn.style.display = 'inline-block';
      downloadSummaryPdfBtn.setAttribute('data-url', summaryPdfUrl);

      downloadSummaryTxtBtn.style.display = 'inline-block';
      downloadSummaryTxtBtn.setAttribute('data-url', summaryTxtUrl);

    } catch (err) {
      alert(`Prediction failed: ${err.message}`);
    } finally {
      predictBtn.textContent = oldText;
      predictBtn.disabled = false;
    }
  });
}

// ===== Download handlers =====
function setupDownload(btn, defaultName) {
  btn.addEventListener('click', () => {
    const url = btn.getAttribute('data-url');
    if (!url) return;
    const a = document.createElement('a');
    a.href = url;
    a.download = defaultName;
    document.body.appendChild(a);
    a.click();
    a.remove();
  });
}

setupDownload(downloadReportBtn, "Brain_Tumor_Report.pdf");
setupDownload(downloadSummaryPdfBtn, "Gemini_Summary.pdf");
// setupDownload(downloadSummaryTxtBtn, "Gemini_Summary.txt");
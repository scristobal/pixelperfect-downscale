import init, { process_image } from './pkg/downscaler.js';

const MAX_GRID = 50;

const dropZone = document.getElementById('drop-zone');
const divider = document.getElementById('divider');
const fileInput = document.getElementById('file-input');
const main = document.querySelector('main');
const panels = document.getElementById('panels');
const inputImg = document.getElementById('input-img');
const outputImg = document.getElementById('output-img');
const report = document.getElementById('report');
const downloads = document.getElementById('downloads');
const statusEl = document.getElementById('status');

const reportGroups = {
    input: report.querySelector('dl[data-group="input"]'),
    detection: report.querySelector('dl[data-group="detection"]'),
    output: report.querySelector('dl[data-group="output"]'),
};

let wasmReady = false;
let inputBytes = null;
let inputStem = 'image';
let outputBlob = null;
let outputWidth = 0;
let outputHeight = 0;

function setStatus(text, kind = '') {
    statusEl.textContent = text;
    statusEl.className = 'status' + (kind ? ' ' + kind : '');
}

function fmtKB(n) {
    return (n / 1024).toFixed(1) + ' KB';
}

function infoList(el, entries) {
    el.innerHTML = '';
    for (const [k, v] of entries) {
        const row = document.createElement('div');
        const dt = document.createElement('dt');
        dt.textContent = k;
        const dd = document.createElement('dd');
        dd.textContent = v;
        row.append(dt, dd);
        el.append(row);
    }
}

async function loadFile(file) {
    if (!file) return;
    inputStem = file.name.replace(/\.[^.]+$/, '');
    inputBytes = new Uint8Array(await file.arrayBuffer());

    inputImg.src = URL.createObjectURL(file);
    await new Promise((resolve, reject) => {
        inputImg.onload = resolve;
        inputImg.onerror = () => reject(new Error('Could not decode image'));
    });

    const w = inputImg.naturalWidth;
    const h = inputImg.naturalHeight;
    const aspect = `${w} / ${h}`;
    for (const wrap of document.querySelectorAll('.image-wrap')) {
        wrap.style.setProperty('--aspect', aspect);
    }

    panels.classList.remove('hidden');
    await runProcess();
}

async function runProcess() {
    if (!wasmReady || !inputBytes) return;
    setStatus('Processing…');
    try {
        const t0 = performance.now();
        const result = process_image(inputBytes, 1, MAX_GRID, true);
        const ms = (performance.now() - t0).toFixed(0);

        const png = result.png;
        outputBlob = new Blob([png], { type: 'image/png' });
        outputWidth = result.width;
        outputHeight = result.height;
        outputImg.src = URL.createObjectURL(outputBlob);

        const confPct = (result.confidence * 100).toFixed(0) + '%';
        const gridText = result.upscaled_detected
            ? `${result.grid_w} × ${result.grid_h} @ (${result.offset_x}, ${result.offset_y})`
            : 'none detected';

        infoList(reportGroups.input, [
            ['Dimensions', `${inputImg.naturalWidth} × ${inputImg.naturalHeight}`],
            ['File size', fmtKB(inputBytes.byteLength)],
            ['Colors', result.input_colors.toLocaleString()],
        ]);
        infoList(reportGroups.detection, [
            ['Grid', gridText],
            ['Confidence', confPct],
            ['Max grid', String(MAX_GRID)],
            ['Time', `${ms} ms`],
        ]);
        infoList(reportGroups.output, [
            ['Dimensions', `${outputWidth} × ${outputHeight}`],
            ['File size', fmtKB(png.byteLength)],
            ['Colors', result.output_colors.toLocaleString()],
        ]);
        report.classList.remove('hidden');
        downloads.classList.remove('hidden');
        if (dropZone.parentElement === main && main.lastElementChild !== dropZone) {
            divider.classList.remove('hidden');
            main.append(dropZone);
        }

        setStatus(
            result.upscaled_detected
                ? ''
                : 'No upscaled grid detected — output is the source.',
            result.upscaled_detected ? '' : 'success',
        );
        result.free();
    } catch (e) {
        console.error(e);
        setStatus('Error: ' + (e?.message || e), 'error');
    }
}

async function downloadAt(scale) {
    if (!outputBlob) return;
    if (scale === 1) {
        triggerDownload(outputBlob, `${inputStem}_1x.png`);
        return;
    }
    const bitmap = await createImageBitmap(outputBlob);
    const canvas = document.createElement('canvas');
    canvas.width = outputWidth * scale;
    canvas.height = outputHeight * scale;
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
    triggerDownload(blob, `${inputStem}_${scale}x.png`);
}

function triggerDownload(blob, name) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
}

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const f = e.dataTransfer.files[0];
    if (f) loadFile(f);
});
fileInput.addEventListener('change', (e) => loadFile(e.target.files[0]));

downloads.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-scale]');
    if (!btn) return;
    downloadAt(parseInt(btn.dataset.scale, 10));
});

const imageWraps = document.querySelectorAll('.image-wrap');
for (const wrap of imageWraps) {
    wrap.addEventListener('mousemove', (e) => {
        const rect = wrap.getBoundingClientRect();
        const x = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
        const y = Math.max(0, Math.min(100, ((e.clientY - rect.top) / rect.height) * 100));
        for (const w of imageWraps) {
            w.classList.add('zoom');
            w.querySelector('img').style.transformOrigin = `${x}% ${y}%`;
        }
    });
    wrap.addEventListener('mouseleave', () => {
        for (const w of imageWraps) {
            w.classList.remove('zoom');
            w.querySelector('img').style.transformOrigin = '';
        }
    });
}

setStatus('Loading WebAssembly…');
init().then(() => {
    wasmReady = true;
    setStatus('Ready. Drop an image to start.');
}).catch((e) => {
    setStatus('Failed to load WebAssembly: ' + e.message, 'error');
});

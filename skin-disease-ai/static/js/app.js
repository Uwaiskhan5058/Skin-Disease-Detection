/**
 * AI Dermatology Assistant — Frontend Logic v2.0
 * ================================================
 * Drag-and-drop uploads, API calls, animated results,
 * healthcare-themed micro-animations, and scroll reveals.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ─── DOM Elements ─────────────────────────────────────────────
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const uploadContent = document.getElementById('upload-content');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const removeImageBtn = document.getElementById('remove-image');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsSection = document.getElementById('results-section');
    const newAnalysisBtn = document.getElementById('new-analysis-btn');
    const navbar = document.getElementById('navbar');

    let selectedFile = null;

    // ─── Navbar Scroll Effect ─────────────────────────────────────
    let lastScroll = 0;
    window.addEventListener('scroll', () => {
        const current = window.scrollY;
        if (current > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
        lastScroll = current;
    });

    // ─── Smooth Scroll ────────────────────────────────────────────
    document.querySelectorAll('.nav-link, .hero-cta').forEach(link => {
        link.addEventListener('click', (e) => {
            const href = link.getAttribute('href');
            if (href && href.startsWith('#')) {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                if (link.classList.contains('nav-link')) link.classList.add('active');
            }
        });
    });

    // ─── Intersection Observer for Reveal Animations ──────────────
    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                revealObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

    document.querySelectorAll('.about-card, .feature-card, .section-header').forEach((el, i) => {
        el.classList.add('reveal');
        el.style.transitionDelay = `${i * 0.08}s`;
        revealObserver.observe(el);
    });

    // ─── Counter Animation for Stats ──────────────────────────────
    const counterObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const el = entry.target;
                const target = parseInt(el.dataset.target);
                if (!isNaN(target)) {
                    animateCounter(el, 0, target, 2000, '');
                }
                counterObserver.unobserve(el);
            }
        });
    }, { threshold: 0.5 });

    document.querySelectorAll('.counter').forEach(el => counterObserver.observe(el));

    // ─── Active Section Detection ─────────────────────────────────
    const sections = document.querySelectorAll('section[id]');
    const sectionObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.getAttribute('id');
                document.querySelectorAll('.nav-link').forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${id}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }, { threshold: 0.3 });

    sections.forEach(section => sectionObserver.observe(section));

    // ─── Drag & Drop Upload ───────────────────────────────────────
    uploadArea.addEventListener('click', () => {
        if (!selectedFile) fileInput.click();
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) handleFileSelect(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
    });

    // ─── File Selection ───────────────────────────────────────────
    function handleFileSelect(file) {
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/webp'];
        if (!allowedTypes.includes(file.type)) {
            showToast('Invalid file type. Please upload PNG, JPG, JPEG, BMP, or WebP.', 'error');
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            showToast('File is too large. Maximum size is 10 MB.', 'error');
            return;
        }

        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadContent.style.display = 'none';
            previewContainer.style.display = 'block';
            uploadArea.classList.add('has-image');
            analyzeBtn.disabled = false;

            // Add a small haptic-like animation
            uploadArea.style.transform = 'scale(0.98)';
            setTimeout(() => { uploadArea.style.transform = 'scale(1)'; }, 150);
        };
        reader.readAsDataURL(file);
    }

    // ─── Remove Image ─────────────────────────────────────────────
    removeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        previewImage.src = '';
        previewContainer.style.display = 'none';
        uploadContent.style.display = 'block';
        uploadArea.classList.remove('has-image');
        analyzeBtn.disabled = true;
        resultsSection.style.display = 'none';
    }

    // ─── Analyze ──────────────────────────────────────────────────
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        showLoading();

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            animateLoadingSteps();

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                hideLoading();
                showToast(data.message || 'Analysis failed. Please try again.', 'error');
                return;
            }

            await sleep(1000);
            hideLoading();
            displayResults(data);
            showToast('Analysis complete! Scroll down to view results.', 'success');

        } catch (error) {
            console.error('Prediction error:', error);
            hideLoading();
            showToast('Failed to connect to server. Please ensure the backend is running.', 'error');
        }
    });

    // ─── New Analysis ─────────────────────────────────────────────
    newAnalysisBtn.addEventListener('click', () => {
        resetUpload();
        document.getElementById('upload-section').scrollIntoView({ behavior: 'smooth' });
    });

    // ─── Loading Animation ────────────────────────────────────────
    function showLoading() {
        loadingOverlay.style.display = 'flex';
        document.body.style.overflow = 'hidden';

        const steps = document.querySelectorAll('.load-step');
        steps.forEach(s => s.classList.remove('active', 'done'));
        steps[0].classList.add('active');

        analyzeBtn.querySelector('.btn-content').style.display = 'none';
        analyzeBtn.querySelector('.btn-loading').style.display = 'flex';
        analyzeBtn.disabled = true;
    }

    function hideLoading() {
        loadingOverlay.style.display = 'none';
        document.body.style.overflow = '';

        analyzeBtn.querySelector('.btn-content').style.display = 'flex';
        analyzeBtn.querySelector('.btn-loading').style.display = 'none';
        analyzeBtn.disabled = false;
    }

    function animateLoadingSteps() {
        const steps = document.querySelectorAll('.load-step');
        const delays = [0, 500, 1000, 1500];

        delays.forEach((delay, index) => {
            setTimeout(() => {
                if (index > 0) {
                    steps[index - 1].classList.remove('active');
                    steps[index - 1].classList.add('done');
                }
                steps[index].classList.add('active');
            }, delay);
        });

        setTimeout(() => {
            steps[steps.length - 1].classList.remove('active');
            steps[steps.length - 1].classList.add('done');
        }, 2200);
    }

    // ─── Display Results ──────────────────────────────────────────
    function displayResults(data) {
        const { prediction, top_3, disease_info, gradcam, uploaded_image } = data;

        resultsSection.style.display = 'block';

        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);

        // Images
        document.getElementById('result-original').src = uploaded_image;
        document.getElementById('result-gradcam').src = gradcam;

        // Prediction name
        document.getElementById('prediction-name').textContent = disease_info.name;

        // Risk badge
        const riskBadge = document.getElementById('risk-badge');
        riskBadge.textContent = prediction.risk_level + ' Risk';
        riskBadge.className = 'risk-badge risk-' + prediction.risk_level.toLowerCase();

        // Confidence bar animation
        const confidenceValue = document.getElementById('confidence-value');
        const confidenceFill = document.getElementById('confidence-fill');
        confidenceFill.style.width = '0%';

        setTimeout(() => {
            confidenceFill.style.width = prediction.confidence_pct + '%';
            animateCounter(confidenceValue, 0, prediction.confidence_pct, 1200, '%');
        }, 400);

        // Color confidence bar based on risk
        if (prediction.risk_level === 'High') {
            confidenceFill.style.background = 'var(--gradient-danger)';
        } else if (prediction.risk_level === 'Medium') {
            confidenceFill.style.background = 'var(--gradient-warning)';
        } else {
            confidenceFill.style.background = 'var(--gradient-health)';
        }

        renderTop3(top_3);
        setRiskIndicator(prediction.risk_level);
        setDiseaseInfo(disease_info);
    }

    function renderTop3(top3) {
        const container = document.getElementById('top3-list');
        container.innerHTML = '';

        top3.forEach((item, index) => {
            const el = document.createElement('div');
            el.className = 'top3-item';
            el.style.animation = `fadeInUp 0.4s ease-out ${index * 0.15}s both`;
            el.innerHTML = `
                <div class="top3-rank">${item.rank}</div>
                <div class="top3-info">
                    <div class="top3-name">${item.label}</div>
                </div>
                <div class="top3-bar">
                    <div class="top3-bar-fill" style="width: 0%"></div>
                </div>
                <div class="top3-conf">${item.confidence_pct}%</div>
            `;
            container.appendChild(el);

            setTimeout(() => {
                el.querySelector('.top3-bar-fill').style.width = item.confidence_pct + '%';
            }, 500 + index * 200);
        });
    }

    function setRiskIndicator(riskLevel) {
        const riskFill = document.getElementById('risk-fill');
        const riskText = document.getElementById('risk-text');

        const positions = { 'Low': '12%', 'Medium': '50%', 'High': '88%' };
        const colors = { 'Low': '#22c55e', 'Medium': '#f59e0b', 'High': '#ef4444' };

        riskText.textContent = riskLevel + ' Risk';
        riskText.style.color = colors[riskLevel] || '#f59e0b';

        setTimeout(() => {
            riskFill.style.left = positions[riskLevel] || '50%';
            riskFill.style.borderColor = colors[riskLevel] || '#f59e0b';
        }, 500);
    }

    function setDiseaseInfo(info) {
        document.getElementById('info-description').textContent = info.description;

        const symptomsList = document.getElementById('info-symptoms');
        symptomsList.innerHTML = '';
        info.symptoms.forEach(s => {
            const li = document.createElement('li');
            li.textContent = s;
            symptomsList.appendChild(li);
        });

        const causesList = document.getElementById('info-causes');
        causesList.innerHTML = '';
        info.causes.forEach(c => {
            const li = document.createElement('li');
            li.textContent = c;
            causesList.appendChild(li);
        });

        const actionsList = document.getElementById('info-actions');
        actionsList.innerHTML = '';
        info.recommended_actions.forEach(a => {
            const li = document.createElement('li');
            li.textContent = a;
            actionsList.appendChild(li);
        });
    }

    // ─── Utility Functions ────────────────────────────────────────
    function animateCounter(element, start, end, duration, suffix = '') {
        const startTime = performance.now();
        const range = end - start;

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = start + (range * eased);

            if (Number.isInteger(end)) {
                element.textContent = Math.round(current) + suffix;
            } else {
                element.textContent = current.toFixed(1) + suffix;
            }

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                if (Number.isInteger(end)) {
                    element.textContent = end + suffix;
                } else {
                    element.textContent = end.toFixed(1) + suffix;
                }
            }
        }

        requestAnimationFrame(update);
    }

    function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

    function showToast(message, type = 'info') {
        const colors = {
            error: { bg: 'rgba(239, 68, 68, 0.12)', border: 'rgba(239, 68, 68, 0.25)', text: '#fca5a5' },
            success: { bg: 'rgba(34, 197, 94, 0.12)', border: 'rgba(34, 197, 94, 0.25)', text: '#86efac' },
            info: { bg: 'rgba(6, 182, 212, 0.12)', border: 'rgba(6, 182, 212, 0.25)', text: '#67e8f9' }
        };

        const c = colors[type] || colors.info;

        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed; top: 24px; right: 24px; z-index: 3000;
            padding: 16px 24px; max-width: 400px;
            background: ${c.bg}; border: 1px solid ${c.border};
            backdrop-filter: blur(20px); border-radius: 14px;
            color: ${c.text}; font-size: 0.88rem;
            font-family: 'Inter', sans-serif;
            animation: slideInRight 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
            cursor: pointer; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        `;
        toast.textContent = message;
        toast.addEventListener('click', () => toast.remove());
        document.body.appendChild(toast);

        if (!document.getElementById('toast-styles')) {
            const style = document.createElement('style');
            style.id = 'toast-styles';
            style.textContent = `
                @keyframes slideInRight {
                    from { opacity: 0; transform: translateX(40px) scale(0.95); }
                    to { opacity: 1; transform: translateX(0) scale(1); }
                }
            `;
            document.head.appendChild(style);
        }

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(40px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }
});

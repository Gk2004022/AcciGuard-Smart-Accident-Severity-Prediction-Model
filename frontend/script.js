// CHANGE THIS to your actual deployed Render backend URL (e.g., https://my-accident-predictor.onrender.com)
const DEPLOYED_BACKEND_URL = 'https://acciguard-smart-accident-severity.onrender.com';

document.addEventListener('DOMContentLoaded', () => {
    // Determine the API base URL dynamically based on environment
    const API_BASE_URL = window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost'
        ? '' // When running locally, use relative path (e.g., /predict)
        : DEPLOYED_BACKEND_URL; // When deployed on Vercel, use the Render URL

    // Sync input coordinates back to map marker on blur
    const latInput = document.getElementById('latitude');
    const lngInput = document.getElementById('longitude');

    // 1. Initialize Lucide Icons
    lucide.createIcons();

    // Track active gauge animation timers to prevent overlapping animations
    const gaugeTimers = {};

    // 2. Set Default Datetime to Local Current Time
    initDefaultDateTime();

    // 3. Initialize Leaflet Map (Centered on UK)
    const defaultLat = 52.5;
    const defaultLng = -1.5;
    const map = L.map('map').setView([defaultLat, defaultLng], 6);

    let activeTileLayer;
    function updateMapTiles(theme) {
        if (activeTileLayer) {
            map.removeLayer(activeTileLayer);
        }
        const isLight = theme === 'light';
        // CartoDB Dark Matter for dark theme, CartoDB Voyager for light theme
        const tileUrl = isLight
            ? 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png'
            : 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';

        activeTileLayer = L.tileLayer(tileUrl, {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 20
        }).addTo(map);
    }

    // Set initial map tiles based on current HTML theme attribute
    const initialTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    updateMapTiles(initialTheme);

    // Initial Marker
    let marker = L.marker([51.5074, -0.1278], { draggable: true }).addTo(map);
    updateCoordsInput(51.5074, -0.1278);

    // Map Click Listener
    map.on('click', (e) => {
        const { lat, lng } = e.latlng;
        marker.setLatLng([lat, lng]);
        updateCoordsInput(lat, lng);
    });

    // Marker Drag Listener
    marker.on('dragend', () => {
        const position = marker.getLatLng();
        updateCoordsInput(position.lat, position.lng);
    });

    function syncInputsToMap() {
        const lat = parseFloat(latInput.value);
        const lng = parseFloat(lngInput.value);
        if (!isNaN(lat) && !isNaN(lng) && lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180) {
            marker.setLatLng([lat, lng]);
            map.panTo([lat, lng]);
        }
    }
    latInput.addEventListener('blur', syncInputsToMap);
    lngInput.addEventListener('blur', syncInputsToMap);

    // 4. Form Submission and Prediction Handling
    const form = document.getElementById('predictor-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // High fidelity feedback loop
        submitBtn.disabled = true;
        btnText.textContent = "Analyzing Accident Features...";
        submitBtn.style.opacity = "0.7";

        // Extract form values
        const payload = {
            latitude: parseFloat(latInput.value),
            longitude: parseFloat(lngInput.value),
            speed_limit: parseFloat(document.getElementById('speed_limit').value),
            accident_datetime: document.getElementById('accident_datetime').value,
            number_of_vehicles: document.getElementById('number_of_vehicles').value,
            number_of_casualties: document.getElementById('number_of_casualties').value,
            urban_or_rural_area: document.getElementById('urban_or_rural_area').value,
            light_conditions: document.getElementById('light_conditions').value,
            weather_conditions: document.getElementById('weather_conditions').value,
            road_surface_conditions: document.getElementById('road_surface_conditions').value,
            road_type: document.getElementById('road_type').value,
            junction_detail: document.getElementById('junction_detail').value,
            first_road_class: document.getElementById('first_road_class').value
        };

        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Prediction request failed.");
            }

            const data = await response.json();
            renderResults(data);

        } catch (error) {
            console.error("Prediction Error:", error);
            alert(`Error running predictive model: ${error.message}`);
        } finally {
            submitBtn.disabled = false;
            btnText.textContent = "Run Predictive Analysis";
            submitBtn.style.opacity = "1";
        }
    });

    // Helper: Set Lat/Lng values in form
    function updateCoordsInput(lat, lng) {
        latInput.value = lat.toFixed(6);
        lngInput.value = lng.toFixed(6);
    }

    // Helper: Populate default local datetime string
    function initDefaultDateTime() {
        const now = new Date();
        const year = now.getFullYear();
        const month = String(now.getMonth() + 1).padStart(2, '0');
        const day = String(now.getDate()).padStart(2, '0');
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');

        const localDateTime = `${year}-${month}-${day}T${hours}:${minutes}`;
        document.getElementById('accident_datetime').value = localDateTime;
    }

    // Helper: Render predict results with styling and gauges
    function renderResults(data) {
        const placeholder = document.getElementById('results-placeholder');
        const content = document.getElementById('results-content');

        // Hide placeholder and reveal results card
        placeholder.classList.add('hidden');
        content.classList.remove('hidden');

        const prediction = data.prediction;
        const meta = data.meta;

        // 1. Set Class Name & Theme Color Badge
        const classEl = document.getElementById('res-severity-class');
        const badgeEl = document.getElementById('res-badge');

        classEl.textContent = prediction.severity_class;

        // Severity mapping
        if (prediction.severity_class === "Fatal") {
            classEl.style.color = "var(--danger)";
            badgeEl.textContent = "Fatal Severity predicted";
            badgeEl.className = "result-status-badge text-danger";
            badgeEl.style.color = "var(--danger)";
            badgeEl.style.backgroundColor = "rgba(239, 68, 68, 0.1)";
        } else if (prediction.severity_class === "Serious") {
            classEl.style.color = "var(--warning)";
            badgeEl.textContent = "Serious Severity predicted";
            badgeEl.className = "result-status-badge text-warning";
            badgeEl.style.color = "var(--warning)";
            badgeEl.style.backgroundColor = "rgba(245, 158, 11, 0.1)";
        } else {
            classEl.style.color = "var(--success)";
            badgeEl.textContent = "Slight Severity predicted";
            badgeEl.className = "result-status-badge";
            badgeEl.style.color = "var(--success)";
            badgeEl.style.backgroundColor = "rgba(16, 185, 129, 0.1)";
        }

        // 2. Animate Circular Gauges
        const slightProb = prediction.probabilities.Slight;
        const seriousProb = prediction.probabilities.Serious;
        const fatalProb = prediction.probabilities.Fatal;

        animateGauge('gauge-slight-path', 'gauge-slight-text', slightProb);
        animateGauge('gauge-serious-path', 'gauge-serious-text', seriousProb);
        animateGauge('gauge-fatal-path', 'gauge-fatal-text', fatalProb);

        // 3. Risk Multipliers
        const fatalMult = prediction.risk_multipliers.Fatal_vs_Average;
        const seriousMult = prediction.risk_multipliers.Serious_vs_Average;

        document.getElementById('res-fatal-multiplier').textContent = `${fatalMult}x`;
        document.getElementById('res-serious-multiplier').textContent = `${seriousMult}x`;

        // Highlight high multipliers
        const fatalCard = document.getElementById('card-fatal-mult');
        if (fatalMult > 3.0) {
            fatalCard.style.borderColor = "var(--danger)";
            fatalCard.style.backgroundColor = "rgba(239, 68, 68, 0.08)";
        } else {
            fatalCard.style.borderColor = "";
            fatalCard.style.backgroundColor = "";
        }

        const seriousCard = document.getElementById('card-serious-mult');
        if (seriousMult > 1.5) {
            seriousCard.style.borderColor = "var(--warning)";
            seriousCard.style.backgroundColor = "rgba(245, 158, 11, 0.08)";
        } else {
            seriousCard.style.borderColor = "";
            seriousCard.style.backgroundColor = "";
        }

        // 4. Extracted Metadata
        const days = {
            1: "Sunday",
            2: "Monday",
            3: "Tuesday",
            4: "Wednesday",
            5: "Thursday",
            6: "Friday",
            7: "Saturday"
        };

        document.getElementById('meta-hour').textContent = `${String(meta.extracted_hour).padStart(2, '0')}:00`;
        document.getElementById('meta-day').textContent = days[meta.extracted_day_of_week] || "Unknown";
        document.getElementById('meta-weekend').textContent = meta.extracted_is_weekend ? "Yes" : "No";
        document.getElementById('meta-tod').textContent = meta.extracted_time_of_day;

        // Smooth scroll results into view on mobile
        document.getElementById('results-section').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Gauge Animate Utility
    function animateGauge(pathId, textId, targetPct) {
        const path = document.getElementById(pathId);
        const text = document.getElementById(textId);

        // Cancel any in-progress animation for this gauge to prevent overlapping
        if (gaugeTimers[pathId]) {
            clearInterval(gaugeTimers[pathId]);
            gaugeTimers[pathId] = null;
        }

        // Reset to zero before animating to ensure clean start
        path.setAttribute('stroke-dasharray', '0, 100');
        text.textContent = '0%';

        // Guard against zero-value edge case
        if (targetPct <= 0) {
            return;
        }

        // Circular dash offset animation
        path.setAttribute('stroke-dasharray', `${targetPct}, 100`);

        // Numeric text counting animation (30 steps over ~450ms)
        let currentPct = 0;
        const increment = targetPct / 30;
        gaugeTimers[pathId] = setInterval(() => {
            currentPct += increment;
            if (currentPct >= targetPct) {
                currentPct = targetPct;
                clearInterval(gaugeTimers[pathId]);
                gaugeTimers[pathId] = null;
            }
            text.textContent = `${Math.round(currentPct)}%`;
        }, 15);
    }

    const themeToggleBtn = document.getElementById('theme-engine-toggle');
    const themeStatusBadge = document.getElementById('theme-engine-status');

    // Sync initial UI badge label on load
    if (themeStatusBadge) {
        themeStatusBadge.textContent = initialTheme === 'dark' ? 'Dark' : 'Light';
    }

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

            // Apply theme attributes to document element
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);

            // Smoothly update state badge label
            if (themeStatusBadge) {
                themeStatusBadge.textContent = newTheme === 'dark' ? 'Dark' : 'Light';
            }

            // Swap Leaflet map tile layers dynamically
            updateMapTiles(newTheme);
        });
    }
});

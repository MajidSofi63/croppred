// India Crop Recommendation System - Frontend JavaScript

let selectedDistrictId = null;
let currentRecommendations = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadSeasons();
    setupDistrictSearch();
    setupRecommendButton();
});

// Load available seasons
async function loadSeasons() {
    try {
        const response = await fetch('/api/seasons');
        const data = await response.json();
        const seasonSelect = document.getElementById('seasonSelect');
        
        data.seasons.forEach(season => {
            const option = document.createElement('option');
            option.value = season;
            option.textContent = season;
            seasonSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading seasons:', error);
    }
}

// Setup district search with autocomplete
function setupDistrictSearch() {
    const districtSearch = document.getElementById('districtSearch');
    const suggestionsDiv = document.getElementById('districtSuggestions');
    
    districtSearch.addEventListener('input', async function() {
        const query = this.value.trim();
        selectedDistrictId = null;
        document.getElementById('districtId').value = '';
        document.getElementById('recommendBtn').disabled = true;
        
        if (query.length < 2) {
            suggestionsDiv.style.display = 'none';
            return;
        }
        
        try {
            const response = await fetch(`/api/districts?q=${encodeURIComponent(query)}`);
            const data = await response.json();
            
            suggestionsDiv.innerHTML = '';
            
            if (data.districts.length > 0) {
                data.districts.forEach(district => {
                    const item = document.createElement('div');
                    item.className = 'suggestion-item';
                    item.textContent = `${district.district}, ${district.state}`;
                    item.onclick = function() {
                        districtSearch.value = `${district.district}, ${district.state}`;
                        selectedDistrictId = district.id;
                        document.getElementById('districtId').value = district.id;
                        suggestionsDiv.style.display = 'none';
                        checkFormValidity();
                    };
                    suggestionsDiv.appendChild(item);
                });
                suggestionsDiv.style.display = 'block';
            } else {
                suggestionsDiv.style.display = 'none';
            }
        } catch (error) {
            console.error('Error searching districts:', error);
            suggestionsDiv.style.display = 'none';
        }
    });
    
    // Hide suggestions when clicking outside
    document.addEventListener('click', function(e) {
        if (e.target !== districtSearch) {
            suggestionsDiv.style.display = 'none';
        }
    });
    
    // Check validity when season changes
    document.getElementById('seasonSelect').addEventListener('change', checkFormValidity);
}

// Check if form is valid
function checkFormValidity() {
    const season = document.getElementById('seasonSelect').value;
    const btn = document.getElementById('recommendBtn');
    
    if (selectedDistrictId && season) {
        btn.disabled = false;
    } else {
        btn.disabled = true;
    }
}

// Setup recommend button
function setupRecommendButton() {
    document.getElementById('recommendBtn').addEventListener('click', getRecommendations);
}

// Get recommendations
async function getRecommendations() {
    const districtId = selectedDistrictId;
    const season = document.getElementById('seasonSelect').value;
    
    // Hide previous results and errors
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('errorState').style.display = 'none';
    document.getElementById('weatherPanel').style.display = 'none';
    
    // Show loading
    document.getElementById('loadingState').style.display = 'block';
    
    try {
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                district_id: districtId,
                season: season
            })
        });
        
        document.getElementById('loadingState').style.display = 'none';
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to get recommendations');
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        document.getElementById('loadingState').style.display = 'none';
        document.getElementById('errorState').style.display = 'block';
        document.getElementById('errorMessage').textContent = error.message;
    }
}

// Display results
function displayResults(data) {
    currentRecommendations = data.recommendations;
    
    // Display weather panel
    displayWeather(data.weather_used);
    document.getElementById('weatherPanel').style.display = 'block';
    
    // Display location info
    document.getElementById('resultDistrict').textContent = `${data.district.name}, ${data.district.state}`;
    document.getElementById('resultSeason').textContent = data.season;
    
    // Display top recommendation
    if (data.recommendations.length > 0) {
        displayTopRecommendation(data.recommendations[0]);
        displayOtherRecommendations(data.recommendations.slice(1));
        displayReasons(data.recommendations);
        displayLimitations(data.limitations);
        displayModelOutputs(data.recommendations, data.model_info);
    }
    
    document.getElementById('resultsSection').style.display = 'block';
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Display weather information
function displayWeather(weatherUsed) {
    // Model input weather
    const modelInput = weatherUsed.model_input;
    document.getElementById('modelTemp').textContent = `${modelInput.temperature} °C`;
    document.getElementById('modelRain').textContent = `${modelInput.rainfall} mm`;
    document.getElementById('modelYears').textContent = modelInput.years_used.join(', ');
    document.getElementById('weatherNote').textContent = modelInput.note;
    
    // Live conditions
    const live = weatherUsed.live_conditions;
    // A live payload can arrive with individual null fields; label each one.
    function liveText(value, unit) {
        return (value === null || value === undefined) ? 'Unavailable' : `${value} ${unit}`;
    }
    if (live) {
        document.getElementById('liveTemp').textContent = liveText(live.temperature, '°C');
        document.getElementById('liveHumidity').textContent = liveText(live.humidity, '%');
        document.getElementById('livePrecip').textContent = liveText(live.precipitation, 'mm');
        document.getElementById('liveWind').textContent = liveText(live.wind_speed, 'km/h');
    } else {
        // The live API can be unavailable; say so instead of showing blanks.
        document.getElementById('liveTemp').textContent = 'Unavailable';
        document.getElementById('liveHumidity').textContent = 'Unavailable';
        document.getElementById('livePrecip').textContent = 'Unavailable';
        document.getElementById('liveWind').textContent = 'Unavailable';
    }
    
    // 7-day forecast
    const forecastContainer = document.getElementById('forecastContainer');
    forecastContainer.innerHTML = '';
    const forecast = weatherUsed.forecast_7day || [];
    if (forecast.length === 0) {
        forecastContainer.innerHTML = '<p style="color: #666; font-size: 13px;">Live forecast unavailable</p>';
        return;
    }
    forecast.slice(0, 7).forEach(day => {
        const item = document.createElement('div');
        item.className = 'forecast-item';
        // Parse the ISO date as local midnight so the displayed day cannot shift.
        const date = new Date(`${day.date}T00:00:00`);
        item.innerHTML = `
            <div class="forecast-date">${date.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' })}</div>
            <div class="forecast-temp">${day.temp_min}° - ${day.temp_max}°C</div>
            <div class="forecast-rain">${day.precipitation} mm</div>
        `;
        forecastContainer.appendChild(item);
    });
}

// Display top recommendation
function displayTopRecommendation(rec) {
    const container = document.getElementById('topRecommendation');
    const agreementClass = (rec.agreement || '').toLowerCase();
    // Build the highlight card. Suitability is always relative to this crop's
    // own median, so the label says so explicitly.
    container.innerHTML = `
        <div class="top-rec-header">
            <div class="top-rec-copy">
                <span class="top-rec-badge">Top pick</span>
                <div class="top-crop-name">${rec.crop}</div>
                <p class="top-rec-sub">
                    Highest predicted yield relative to this crop's own historical
                    median in this district and season.
                </p>
            </div>
            <div class="suitability-tile">
                <span class="tile-value">${rec.suitability}%</span>
                <span class="tile-label">of historical median</span>
            </div>
        </div>
        <div class="top-detail-grid">
            <div class="detail-card">
                <span class="detail-label">Expected yield</span>
                <span class="detail-value">${rec.expected_yield} <small>t/ha</small></span>
            </div>
            <div class="detail-card">
                <span class="detail-label">Historical median</span>
                <span class="detail-value">${rec.historical_median} <small>t/ha</small></span>
            </div>
            <div class="detail-card">
                <span class="detail-label">Typical range</span>
                <span class="detail-value">${rec.historical_range.p25}–${rec.historical_range.p75} <small>t/ha</small></span>
            </div>
            <div class="detail-card">
                <span class="detail-label">Years of evidence</span>
                <span class="detail-value">${rec.years_of_evidence} <small>years</small></span>
            </div>
        </div>
        <div class="top-rec-footer">
            <span class="agreement-pill ${agreementClass}">
                ${rec.agreement} model agreement
            </span>
            <span class="top-rec-hint">Ranked by suitability, never by raw yield across crops.</span>
        </div>
    `;
}

// Display other recommendations
function displayOtherRecommendations(recs) {
    const container = document.getElementById('otherRecommendations');
    container.innerHTML = '';
    
    if (recs.length === 0) {
        container.innerHTML = '<p style="color: #666; font-size: 13px;">No other recommendations</p>';
        return;
    }
    
    recs.forEach((rec, index) => {
        const item = document.createElement('div');
        item.className = 'other-crop-item';
        // Show every model's own prediction as a compact, colour-coded chip.
        const modelLine = rec.models
            ? `<div class="other-crop-models">
                   <span class="models-label">Model outputs · t/ha</span>
                   <span class="model-chip rf">RF <b>${rec.models.RandomForest}</b></span>
                   <span class="model-chip xgb">XGB <b>${rec.models.XGBoost}</b></span>
                   <span class="model-chip hgb">HGB <b>${rec.models.HistGradientBoosting}</b></span>
                   <span class="model-chip ens">Ens <b>${rec.models.Ensemble}</b></span>
               </div>`
            : '';
        const agreementClass = (rec.agreement || '').toLowerCase();
        item.innerHTML = `
            <div class="other-crop-head">
                <div class="other-crop-title">
                    <span class="rank-badge">#${index + 2}</span>
                    <span class="other-crop-name">${rec.crop}</span>
                </div>
                <span class="suitability-pill">${rec.suitability}% of median</span>
            </div>
            <div class="other-crop-stats">
                <span class="other-stat"><strong>${rec.expected_yield}</strong> t/ha expected</span>
                <span class="other-stat"><strong>${rec.historical_median}</strong> t/ha median</span>
                <span class="other-stat"><strong>${rec.years_of_evidence}</strong> years</span>
                <span class="agreement-dot ${agreementClass}">${rec.agreement} agreement</span>
            </div>
            ${modelLine}
        `;
        container.appendChild(item);
    });
}

// Display reasons
function displayReasons(recommendations) {
    const container = document.getElementById('reasonsSection');
    container.innerHTML = '';

    // Show the computed reasons for every recommended crop, not just the top.
    recommendations.forEach((rec, index) => {
        // Use a native <details> so each explanation can be expanded without JS.
        const block = document.createElement('details');
        block.className = 'reason-block';
        // Keep the top crop's reasoning visible by default.
        if (index === 0) {
            block.open = true;
        }

        const summary = document.createElement('summary');
        const heading = document.createElement('span');
        heading.className = 'reason-heading';

        const rank = document.createElement('span');
        rank.className = 'reason-rank';
        rank.textContent = `#${index + 1}`;

        const title = document.createElement('span');
        title.className = 'reason-title';
        title.textContent = `Why ${rec.crop}`;

        heading.appendChild(rank);
        heading.appendChild(title);

        const meta = document.createElement('span');
        meta.className = 'reason-meta';
        meta.textContent = `${rec.reasons.length} signals`;

        summary.appendChild(heading);
        summary.appendChild(meta);
        block.appendChild(summary);

        const ul = document.createElement('ul');
        rec.reasons.forEach(reason => {
            const li = document.createElement('li');
            li.textContent = reason;
            ul.appendChild(li);
        });
        block.appendChild(ul);
        container.appendChild(block);
    });
}

// Display limitations
function displayLimitations(limitations) {
    const container = document.getElementById('limitationsSection');
    container.innerHTML = '<ul></ul>';
    const ul = container.querySelector('ul');
    
    limitations.forEach(limitation => {
        const li = document.createElement('li');
        li.textContent = limitation;
        ul.appendChild(li);
    });
}

// Display model outputs in sidebar
function displayModelOutputs(recommendations, modelInfo) {
    const container = document.getElementById('modelOutputs');
    container.innerHTML = '';

    if (recommendations.length === 0) return;

    const topRec = recommendations[0];
    // Shared header attributes: the card can be expanded by mouse or keyboard.
    const headerAttrs = 'onclick="toggleModelCard(this)" onkeydown="if(event.key===\'Enter\'||event.key===\' \'){event.preventDefault();toggleModelCard(this);}" role="button" tabindex="0" aria-expanded="false"';
    // Map the API model names to the short class names used by the stylesheet.
    const models = [
        { name: 'RandomForest', label: 'Random Forest', className: 'rf' },
        { name: 'XGBoost', label: 'XGBoost', className: 'xgb' },
        { name: 'HistGradientBoosting', label: 'HistGradientBoosting', className: 'hgb' }
    ];

    // Reusable metric rows so every card renders metrics identically.
    const metricRows = (metric) => {
        if (!metric) {
            return '<div class="model-detail-row"><span>Metrics not available</span></div>';
        }
        return `
            <div class="model-detail-row">
                <span>Test R²</span>
                <span>${metric.R2.toFixed(4)}</span>
            </div>
            <div class="model-detail-row">
                <span>Test MAE</span>
                <span>${metric.MAE.toFixed(4)} t/ha</span>
            </div>
            <div class="model-detail-row">
                <span>Test RMSE</span>
                <span>${metric.RMSE.toFixed(4)} t/ha</span>
            </div>
        `;
    };

    // Build a card for each individual model.
    models.forEach(model => {
        const pred = topRec.models[model.name];
        const weight = modelInfo.weights ? modelInfo.weights[model.name] : 0;
        const testMetric = modelInfo.test_metrics ? modelInfo.test_metrics[model.name] : null;

        const card = document.createElement('div');
        card.className = `model-card ${model.className}`;
        card.innerHTML = `
            <div class="model-card-header" ${headerAttrs}>
                <span class="model-name">${model.label}</span>
                <span class="model-arrow" aria-hidden="true">▸</span>
            </div>
            <div class="model-body">
                <span class="model-prediction">${pred} <span>t/ha</span></span>
                <span class="weight-pill">${(weight * 100).toFixed(1)}% weight</span>
            </div>
            <div class="model-details">
                ${metricRows(testMetric)}
            </div>
        `;
        container.appendChild(card);
    });

    // Add the weighted ensemble, which is the value used for ranking.
    const ensembleMetric = modelInfo.test_metrics
        ? modelInfo.test_metrics.Ensemble
        : null;
    const ensembleCard = document.createElement('div');
    ensembleCard.className = 'model-card ensemble';
    ensembleCard.innerHTML = `
        <div class="model-card-header" ${headerAttrs}>
            <span class="model-name">Ensemble (weighted)</span>
            <span class="model-arrow" aria-hidden="true">▸</span>
        </div>
        <div class="model-body">
            <span class="model-prediction">${topRec.models.Ensemble} <span>t/ha</span></span>
            <span class="weight-pill">Used for ranking</span>
        </div>
        <div class="model-details">
            ${metricRows(ensembleMetric)}
        </div>
    `;
    container.appendChild(ensembleCard);

    // Add the honest baseline so users can compare the ML models against it.
    const baselineMetric = modelInfo.test_metrics
        ? modelInfo.test_metrics['Baseline (District-Crop Median)']
        : null;
    const baselineCard = document.createElement('div');
    baselineCard.className = 'model-card baseline';
    baselineCard.innerHTML = `
        <div class="model-card-header" ${headerAttrs}>
            <span class="model-name">Baseline (district-crop median)</span>
            <span class="model-arrow" aria-hidden="true">▸</span>
        </div>
        <div class="model-body">
            <span class="model-prediction">${topRec.historical_median} <span>t/ha</span></span>
            <span class="weight-pill">No ML</span>
        </div>
        <div class="model-details">
            ${metricRows(baselineMetric)}
        </div>
    `;
    container.appendChild(baselineCard);
}

// Toggle model card expansion
function toggleModelCard(header) {
    const card = header.parentElement;
    card.classList.toggle('expanded');
    // Keep assistive technology in sync with the visual state.
    header.setAttribute('aria-expanded', String(card.classList.contains('expanded')));
}

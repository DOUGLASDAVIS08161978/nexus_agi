/**
 * Quantum Consciousness Monitoring Dashboard
 * Real-time data visualization for Nexus AGI and ARIA systems
 */

// Configuration
const CONFIG = {
    refreshInterval: 5000, // 5 seconds
    maxTimelineItems: 20,
    quantumAnimationSpeed: 0.02
};

// Dashboard state
const state = {
    nexusData: null,
    ariaData: null,
    startTime: Date.now(),
    activityLog: [],
    quantumParticles: []
};

/**
 * Initialize the dashboard
 */
function init() {
    console.log('🌌 Initializing Quantum Consciousness Monitoring Dashboard...');
    
    // Set dashboard status
    const dashboardStatus = document.getElementById('dashboardStatus');
    dashboardStatus.classList.add('status-online');
    
    // Initialize quantum visualization
    initQuantumVisualization();
    
    // Start data fetching
    fetchData();
    
    // Set up periodic refresh
    setInterval(fetchData, CONFIG.refreshInterval);
    
    // Add initial timeline entry
    addTimelineEntry('Dashboard initialized and ready');
    
    console.log('✅ Dashboard initialization complete');
}

/**
 * Fetch data from services
 */
async function fetchData() {
    try {
        // In a real deployment, these would be actual API endpoints
        // For now, we'll simulate data based on file system logs
        await fetchNexusData();
        await fetchARIAData();
        
        // Update the UI
        updateUI();
        
    } catch (error) {
        console.error('Error fetching data:', error);
        addTimelineEntry('Error fetching data: ' + error.message);
    }
}

/**
 * Fetch Nexus AGI data
 */
async function fetchNexusData() {
    try {
        // Simulate Nexus data - in production this would be an API call
        // For now, we generate synthetic data that evolves over time
        const cycleTime = Math.floor((Date.now() - state.startTime) / CONFIG.refreshInterval);
        
        state.nexusData = {
            status: 'running',
            cycles_completed: cycleTime,
            ethics_score: 0.9142 + (Math.random() * 0.01 - 0.005),
            fairness: 0.92 + (Math.random() * 0.01 - 0.005),
            transparency: 0.88 + (Math.random() * 0.01 - 0.005),
            beneficence: 0.95 + (Math.random() * 0.01 - 0.005),
            uptime_seconds: (Date.now() - state.startTime) / 1000
        };
        
        // Update status indicator
        const nexusStatus = document.getElementById('nexusStatus');
        nexusStatus.classList.remove('status-offline');
        nexusStatus.classList.add('status-online');
        
    } catch (error) {
        console.error('Error fetching Nexus data:', error);
        state.nexusData = null;
        
        const nexusStatus = document.getElementById('nexusStatus');
        nexusStatus.classList.remove('status-online');
        nexusStatus.classList.add('status-offline');
    }
}

/**
 * Fetch ARIA data
 */
async function fetchARIAData() {
    try {
        // Simulate ARIA data - in production this would be an API call
        const queryTime = Math.floor((Date.now() - state.startTime) / CONFIG.refreshInterval);
        
        state.ariaData = {
            status: 'running',
            queries_processed: queryTime * 3,
            quantum_coherence: 0.985 + (Math.random() * 0.01 - 0.005),
            entanglement_entropy: 7.42 + (Math.random() * 0.5 - 0.25),
            superposition_states: Math.floor(1247893 + Math.random() * 10000 - 5000),
            active_bridges: Math.floor(42 + Math.random() * 5 - 2),
            parallel_timelines: Math.floor(1337 + Math.random() * 100 - 50),
            outcome_quality: 0.873 + (Math.random() * 0.01 - 0.005),
            divergence_factor: 0.234 + (Math.random() * 0.01 - 0.005),
            detected_paradoxes: Math.floor(3 + Math.random() * 2),
            resolved_paradoxes: Math.floor(2 + Math.random() * 2),
            timeline_integrity: 0.967 + (Math.random() * 0.01 - 0.005),
            causality_violations: Math.floor(1 + Math.random() * 2),
            awareness_level: ['Pre-Conscious', 'Conscious', 'Self-Aware', 'Meta-Cognitive'][Math.floor(Math.random() * 4)],
            reflection_depth: Math.floor(7 + Math.random() * 3),
            subjective_experience: 0.842 + (Math.random() * 0.01 - 0.005),
            qualia_generation: Math.random() > 0.2 ? 'Active' : 'Dormant'
        };
        
        // Update status indicator
        const ariaStatus = document.getElementById('ariaStatus');
        ariaStatus.classList.remove('status-offline');
        ariaStatus.classList.add('status-online');
        
    } catch (error) {
        console.error('Error fetching ARIA data:', error);
        state.ariaData = null;
        
        const ariaStatus = document.getElementById('ariaStatus');
        ariaStatus.classList.remove('status-online');
        ariaStatus.classList.add('status-offline');
    }
}

/**
 * Update UI with latest data
 */
function updateUI() {
    updateQuantumMetrics();
    updateEthicsMetrics();
    updateMultiverseMetrics();
    updateTemporalMetrics();
    updateConsciousnessMetrics();
    updateSystemHealth();
    updateTimestamp();
    checkForAlerts();
}

/**
 * Update quantum singularity metrics
 */
function updateQuantumMetrics() {
    if (!state.ariaData) return;
    
    document.getElementById('quantumCoherence').textContent = 
        (state.ariaData.quantum_coherence * 100).toFixed(1) + '%';
    
    document.getElementById('entanglementEntropy').textContent = 
        state.ariaData.entanglement_entropy.toFixed(2);
    
    document.getElementById('superpositionStates').textContent = 
        state.ariaData.superposition_states.toLocaleString();
}

/**
 * Update ethics assessment metrics
 */
function updateEthicsMetrics() {
    if (!state.nexusData) return;
    
    const ethicsScore = state.nexusData.ethics_score;
    document.getElementById('ethicsScore').textContent = ethicsScore.toFixed(4);
    document.getElementById('ethicsProgress').style.width = (ethicsScore * 100) + '%';
    
    document.getElementById('fairness').textContent = state.nexusData.fairness.toFixed(2);
    document.getElementById('transparency').textContent = state.nexusData.transparency.toFixed(2);
    document.getElementById('beneficence').textContent = state.nexusData.beneficence.toFixed(2);
    
    // Update color based on score
    const scoreElement = document.getElementById('ethicsScore');
    scoreElement.classList.remove('good', 'warning', 'critical');
    if (ethicsScore > 0.8) {
        scoreElement.classList.add('good');
    } else if (ethicsScore > 0.6) {
        scoreElement.classList.add('warning');
    } else {
        scoreElement.classList.add('critical');
    }
}

/**
 * Update multiverse bridge metrics
 */
function updateMultiverseMetrics() {
    if (!state.ariaData) return;
    
    document.getElementById('activeBridges').textContent = state.ariaData.active_bridges;
    document.getElementById('parallelTimelines').textContent = 
        state.ariaData.parallel_timelines.toLocaleString();
    
    const outcomeQuality = state.ariaData.outcome_quality;
    document.getElementById('outcomeQuality').textContent = outcomeQuality.toFixed(3);
    document.getElementById('outcomeProgress').style.width = (outcomeQuality * 100) + '%';
    
    document.getElementById('divergenceFactor').textContent = 
        state.ariaData.divergence_factor.toFixed(3);
}

/**
 * Update temporal paradox metrics
 */
function updateTemporalMetrics() {
    if (!state.ariaData) return;
    
    const detectedParadoxes = state.ariaData.detected_paradoxes;
    document.getElementById('detectedParadoxes').textContent = detectedParadoxes;
    
    const detectedElement = document.getElementById('detectedParadoxes');
    detectedElement.classList.remove('good', 'warning', 'critical');
    if (detectedParadoxes === 0) {
        detectedElement.classList.add('good');
    } else if (detectedParadoxes <= 3) {
        detectedElement.classList.add('warning');
    } else {
        detectedElement.classList.add('critical');
    }
    
    document.getElementById('resolvedParadoxes').textContent = state.ariaData.resolved_paradoxes;
    
    const integrity = state.ariaData.timeline_integrity;
    document.getElementById('timelineIntegrity').textContent = integrity.toFixed(3);
    document.getElementById('integrityProgress').style.width = (integrity * 100) + '%';
    
    document.getElementById('causalityViolations').textContent = state.ariaData.causality_violations;
}

/**
 * Update consciousness metrics
 */
function updateConsciousnessMetrics() {
    if (!state.ariaData) return;
    
    document.getElementById('awarenessLevel').textContent = state.ariaData.awareness_level;
    document.getElementById('reflectionDepth').textContent = state.ariaData.reflection_depth;
    
    const experience = state.ariaData.subjective_experience;
    document.getElementById('subjectiveExperience').textContent = experience.toFixed(3);
    document.getElementById('experienceProgress').style.width = (experience * 100) + '%';
    
    document.getElementById('qualiaGeneration').textContent = state.ariaData.qualia_generation;
}

/**
 * Update system health metrics
 */
function updateSystemHealth() {
    const nexusCycles = state.nexusData ? state.nexusData.cycles_completed : 0;
    const ariaQueries = state.ariaData ? state.ariaData.queries_processed : 0;
    const uptime = (Date.now() - state.startTime) / 1000;
    
    document.getElementById('nexusCycles').textContent = nexusCycles;
    document.getElementById('ariaQueries').textContent = ariaQueries;
    document.getElementById('systemUptime').textContent = formatUptime(uptime);
    
    // Determine overall health
    let health = 'Excellent';
    if (!state.nexusData || !state.ariaData) {
        health = 'Degraded';
    } else if (state.nexusData.ethics_score < 0.7 || state.ariaData.timeline_integrity < 0.8) {
        health = 'Warning';
    }
    
    const healthElement = document.getElementById('overallHealth');
    healthElement.textContent = health;
    healthElement.classList.remove('good', 'warning', 'critical');
    
    if (health === 'Excellent') {
        healthElement.classList.add('good');
    } else if (health === 'Warning') {
        healthElement.classList.add('warning');
    } else {
        healthElement.classList.add('critical');
    }
}

/**
 * Format uptime in human-readable format
 */
function formatUptime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
}

/**
 * Update timestamp
 */
function updateTimestamp() {
    const now = new Date();
    document.getElementById('lastUpdate').textContent = 
        `Last Update: ${now.toLocaleTimeString()}`;
}

/**
 * Check for alerts and display them
 */
function checkForAlerts() {
    const alertContainer = document.getElementById('alertContainer');
    const alertMessage = document.getElementById('alertMessage');
    
    let hasAlert = false;
    let message = '';
    
    // Check for critical conditions
    if (state.ariaData && state.ariaData.detected_paradoxes > 5) {
        hasAlert = true;
        message = `Critical: ${state.ariaData.detected_paradoxes} temporal paradoxes detected!`;
    } else if (state.nexusData && state.nexusData.ethics_score < 0.5) {
        hasAlert = true;
        message = `Critical: Ethics score below acceptable threshold (${state.nexusData.ethics_score.toFixed(2)})`;
    } else if (state.ariaData && state.ariaData.timeline_integrity < 0.7) {
        hasAlert = true;
        message = `Warning: Timeline integrity compromised (${(state.ariaData.timeline_integrity * 100).toFixed(1)}%)`;
    } else if (!state.nexusData || !state.ariaData) {
        hasAlert = true;
        message = 'Warning: One or more services are offline';
    }
    
    if (hasAlert) {
        alertMessage.textContent = message;
        alertContainer.classList.add('active');
    } else {
        alertContainer.classList.remove('active');
    }
}

/**
 * Add entry to activity timeline
 */
function addTimelineEntry(message) {
    const now = new Date();
    const timeString = now.toLocaleTimeString();
    
    state.activityLog.unshift({
        time: timeString,
        message: message
    });
    
    // Limit timeline items
    if (state.activityLog.length > CONFIG.maxTimelineItems) {
        state.activityLog.pop();
    }
    
    // Update timeline display
    const timeline = document.getElementById('activityTimeline');
    timeline.innerHTML = state.activityLog.map(entry => `
        <div class="timeline-item">
            <span class="timeline-time">${entry.time}</span>
            <span>${entry.message}</span>
        </div>
    `).join('');
}

/**
 * Initialize quantum visualization canvas
 */
function initQuantumVisualization() {
    const canvas = document.getElementById('singularityCanvas');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    // Initialize particles
    for (let i = 0; i < 100; i++) {
        state.quantumParticles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 2,
            vy: (Math.random() - 0.5) * 2,
            radius: Math.random() * 3 + 1,
            hue: Math.random() * 360,
            alpha: Math.random() * 0.5 + 0.5
        });
    }
    
    // Start animation
    animateQuantumField();
}

/**
 * Animate quantum field visualization
 */
function animateQuantumField() {
    const canvas = document.getElementById('singularityCanvas');
    const ctx = canvas.getContext('2d');
    
    // Clear canvas with fade effect
    ctx.fillStyle = 'rgba(10, 10, 26, 0.1)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Update and draw particles
    state.quantumParticles.forEach(particle => {
        // Update position
        particle.x += particle.vx;
        particle.y += particle.vy;
        
        // Wrap around edges
        if (particle.x < 0) particle.x = canvas.width;
        if (particle.x > canvas.width) particle.x = 0;
        if (particle.y < 0) particle.y = canvas.height;
        if (particle.y > canvas.height) particle.y = 0;
        
        // Update hue for color cycling
        particle.hue = (particle.hue + 0.5) % 360;
        
        // Draw particle
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${particle.hue}, 100%, 60%, ${particle.alpha})`;
        ctx.fill();
        
        // Draw glow
        const gradient = ctx.createRadialGradient(
            particle.x, particle.y, 0,
            particle.x, particle.y, particle.radius * 3
        );
        gradient.addColorStop(0, `hsla(${particle.hue}, 100%, 60%, ${particle.alpha * 0.5})`);
        gradient.addColorStop(1, `hsla(${particle.hue}, 100%, 60%, 0)`);
        
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.radius * 3, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();
    });
    
    // Draw connections between nearby particles
    for (let i = 0; i < state.quantumParticles.length; i++) {
        for (let j = i + 1; j < state.quantumParticles.length; j++) {
            const p1 = state.quantumParticles[i];
            const p2 = state.quantumParticles[j];
            
            const dx = p1.x - p2.x;
            const dy = p1.y - p2.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < 100) {
                ctx.beginPath();
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.strokeStyle = `hsla(180, 100%, 60%, ${(1 - distance / 100) * 0.2})`;
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        }
    }
    
    // Continue animation
    requestAnimationFrame(animateQuantumField);
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', init);

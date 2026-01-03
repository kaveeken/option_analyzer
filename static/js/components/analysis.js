/**
 * Analysis Component
 *
 * Handles strategy analysis, metrics display, and chart rendering
 */

/**
 * Initialize analysis component
 */
function initAnalysis() {
    const autoAnalyzeCheckbox = getById('auto-analyze');
    const manualAnalyzeButton = getById('manual-analyze');
    const analysisSection = getById('analysis-section');

    // Handle auto-analyze checkbox
    if (autoAnalyzeCheckbox) {
        // Set initial state
        autoAnalyzeCheckbox.checked = state.get('autoAnalyze');

        on(autoAnalyzeCheckbox, 'change', () => {
            const isChecked = autoAnalyzeCheckbox.checked;
            state.setState({ autoAnalyze: isChecked });

            // Enable/disable manual button
            if (manualAnalyzeButton) {
                manualAnalyzeButton.disabled = isChecked;
            }

            // If enabling auto-analyze and we have positions, trigger analysis
            if (isChecked && state.hasPositions()) {
                triggerAnalysis();
            }
        });
    }

    // Handle manual analyze button
    if (manualAnalyzeButton) {
        on(manualAnalyzeButton, 'click', () => {
            triggerAnalysis();
        });

        // Initial state - disabled if auto-analyze is on
        manualAnalyzeButton.disabled = state.get('autoAnalyze');
    }

    // Subscribe to state changes
    state.subscribe((newState, changedKeys) => {
        // Show analysis section when strategy is initialized
        if (changedKeys.includes('symbol') && newState.symbol) {
            show(analysisSection);
        }

        // Auto-analyze when positions change (if enabled)
        if (changedKeys.includes('positions')) {
            if (newState.autoAnalyze && newState.positions.length > 0) {
                // Use setTimeout to avoid triggering during initial load
                setTimeout(() => triggerAnalysis(), 100);
            }
        }

        // Update metrics display when analysis results change
        if (changedKeys.includes('analysis')) {
            if (newState.analysis) {
                renderMetrics(newState.analysis);
                renderChart(newState.analysis.plot_url);
            } else {
                clearMetrics();
                clearChart();
            }
        }
    });
}

/**
 * Trigger strategy analysis
 */
async function triggerAnalysis() {
    // Check if we have positions
    if (!state.hasPositions()) {
        showError('Add at least one position to analyze the strategy');
        return;
    }

    try {
        await analyzeStrategy();
    } catch (error) {
        console.error('Failed to analyze strategy:', error);
    }
}

/**
 * Render metrics display
 * @param {Object} analysis - Analysis results
 */
function renderMetrics(analysis) {
    const metricsDisplay = getById('metrics-display');
    const metricEv = getById('metric-ev');
    const metricPop = getById('metric-pop');
    const metricMaxGain = getById('metric-max-gain');
    const metricMaxLoss = getById('metric-max-loss');

    if (!metricsDisplay) {
        return;
    }

    // Update metric values
    if (metricEv) {
        setText(metricEv, formatCurrency(analysis.expected_value));
        // Color code based on positive/negative
        if (analysis.expected_value > 0) {
            addClass(metricEv, 'text-success');
            removeClass(metricEv, 'text-error');
        } else if (analysis.expected_value < 0) {
            addClass(metricEv, 'text-error');
            removeClass(metricEv, 'text-success');
        }
    }

    if (metricPop) {
        setText(metricPop, formatPercent(analysis.probability_of_profit));
    }

    if (metricMaxGain) {
        const gainText = analysis.max_gain !== null
            ? formatCurrency(analysis.max_gain)
            : 'Unlimited';
        setText(metricMaxGain, gainText);
    }

    if (metricMaxLoss) {
        const lossText = analysis.max_loss !== null
            ? formatCurrency(analysis.max_loss)
            : 'Unlimited';
        setText(metricMaxLoss, lossText);
    }

    // Show metrics display
    show(metricsDisplay);
}

/**
 * Render strategy chart
 * @param {string} plotUrl - URL to the chart image
 */
function renderChart(plotUrl) {
    const chartContainer = getById('chart-container');
    const chartImage = getById('strategy-chart');

    if (!chartContainer || !chartImage) {
        return;
    }

    // Update chart image source
    // Add cache buster to force reload
    chartImage.src = plotUrl + '?t=' + Date.now();

    // Show chart container
    show(chartContainer);
}

/**
 * Clear metrics display
 */
function clearMetrics() {
    const metricsDisplay = getById('metrics-display');
    const metricEv = getById('metric-ev');
    const metricPop = getById('metric-pop');
    const metricMaxGain = getById('metric-max-gain');
    const metricMaxLoss = getById('metric-max-loss');

    if (metricEv) setText(metricEv, '-');
    if (metricPop) setText(metricPop, '-');
    if (metricMaxGain) setText(metricMaxGain, '-');
    if (metricMaxLoss) setText(metricMaxLoss, '-');

    if (metricsDisplay) {
        hide(metricsDisplay);
    }
}

/**
 * Clear chart display
 */
function clearChart() {
    const chartContainer = getById('chart-container');
    const chartImage = getById('strategy-chart');

    if (chartImage) {
        chartImage.src = '';
    }

    if (chartContainer) {
        hide(chartContainer);
    }
}

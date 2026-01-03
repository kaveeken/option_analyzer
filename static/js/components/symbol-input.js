/**
 * Symbol Input Component
 *
 * Handles stock symbol input, initialization, and display
 */

/**
 * Initialize symbol input component
 */
function initSymbolInput() {
    const symbolInput = getById('symbol-input');
    const symbolSubmit = getById('symbol-submit');
    const stockInfo = getById('stock-info');
    const monthSection = getById('month-section');

    if (!symbolInput || !symbolSubmit) {
        console.error('Symbol input elements not found');
        return;
    }

    // Handle form submission
    const handleSubmit = async () => {
        const symbol = symbolInput.value.trim().toUpperCase();

        if (!symbol) {
            showError('Please enter a stock symbol');
            return;
        }

        try {
            // Initialize strategy with the symbol
            const data = await initStrategy(symbol);

            // Render stock information
            renderStockInfo(data);

            // Populate month dropdown
            populateMonthDropdown(data.available_expirations, data.target_date);

            // Show month section
            show(monthSection);

            // Clear input
            symbolInput.value = '';
        } catch (error) {
            // Error already handled by API client
            console.error('Failed to initialize strategy:', error);
        }
    };

    // Submit button click
    on(symbolSubmit, 'click', handleSubmit);

    // Enter key in input
    on(symbolInput, 'keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleSubmit();
        }
    });

    // Subscribe to state changes to update UI
    state.subscribe((newState, changedKeys) => {
        if (changedKeys.includes('symbol') || changedKeys.includes('currentPrice')) {
            if (newState.symbol) {
                renderStockInfo(newState);
            }
        }

        if (changedKeys.includes('availableExpirations')) {
            if (newState.availableExpirations && newState.availableExpirations.length > 0) {
                populateMonthDropdown(newState.availableExpirations, newState.targetDate);
                show(monthSection);
            }
        }
    });
}

/**
 * Render stock information display
 * @param {Object} data - Stock data with symbol and current_price
 */
function renderStockInfo(data) {
    const stockInfo = getById('stock-info');
    const stockSymbol = getById('stock-symbol');
    const stockPrice = getById('stock-price');

    if (!stockInfo || !stockSymbol || !stockPrice) {
        return;
    }

    // Update symbol and price
    setText(stockSymbol, data.symbol);
    setText(stockPrice, formatCurrency(data.current_price || data.currentPrice));

    // Show stock info box
    show(stockInfo);
}

/**
 * Populate month selector dropdown
 * @param {string[]} expirations - Available expiration months
 * @param {string} selectedMonth - Currently selected month
 */
function populateMonthDropdown(expirations, selectedMonth) {
    const monthSelector = getById('month-selector');

    if (!monthSelector) {
        return;
    }

    // Clear existing options
    clearChildren(monthSelector);

    // Add placeholder option
    const placeholder = createElement('option', { value: '' }, 'Select expiration month');
    monthSelector.appendChild(placeholder);

    // Add expiration options
    expirations.forEach(month => {
        const option = createElement('option', { value: month }, month);
        if (month === selectedMonth) {
            option.selected = true;
        }
        monthSelector.appendChild(option);
    });

    // If a month is selected, enable the load button
    const monthLoad = getById('month-load');
    if (monthLoad) {
        monthLoad.disabled = !selectedMonth;
    }
}

/**
 * Clear symbol input and related UI
 */
function clearSymbolInput() {
    const symbolInput = getById('symbol-input');
    const stockInfo = getById('stock-info');
    const monthSection = getById('month-section');

    if (symbolInput) {
        symbolInput.value = '';
    }

    if (stockInfo) {
        hide(stockInfo);
    }

    if (monthSection) {
        hide(monthSection);
    }
}

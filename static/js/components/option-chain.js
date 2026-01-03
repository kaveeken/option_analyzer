/**
 * Option Chain Table Component
 *
 * Displays option chain with calls and puts, allows adding positions
 */

/**
 * Initialize option chain component
 */
function initOptionChain() {
    const monthSelector = getById('month-selector');
    const monthLoad = getById('month-load');
    const optionChainSection = getById('option-chain-section');

    if (!monthSelector || !monthLoad) {
        console.error('Option chain elements not found');
        return;
    }

    // Enable/disable load button based on selection
    on(monthSelector, 'change', () => {
        monthLoad.disabled = !monthSelector.value;
    });

    // Handle load chain button click
    on(monthLoad, 'click', async () => {
        const symbol = state.get('symbol');
        const month = monthSelector.value;
        const currentTargetDate = state.get('targetDate');

        if (!symbol || !month) {
            showError('Please select an expiration month');
            return;
        }

        try {
            // Update target date if it's different from current
            if (month !== currentTargetDate) {
                await updateTargetDate(month);
            }

            // Load option chain from API
            await getOptionChain(symbol, month);

            // Show option chain section
            show(optionChainSection);
        } catch (error) {
            // Error already handled by API client
            console.error('Failed to load option chain:', error);
        }
    });

    // Subscribe to state changes
    state.subscribe((newState, changedKeys) => {
        // Render table when option chain is loaded
        if (changedKeys.includes('optionChain')) {
            if (newState.optionChain) {
                renderOptionChainTable(newState.optionChain);
                show(optionChainSection);
            }
        }

        // Re-render when positions change to update highlights
        if (changedKeys.includes('positions')) {
            if (newState.optionChain) {
                renderOptionChainTable(newState.optionChain);
            }
        }
    });
}

/**
 * Merge calls and puts by strike price
 * @param {Array} calls - Array of call option contracts
 * @param {Array} puts - Array of put option contracts
 * @returns {Array} Merged array sorted by strike
 */
function mergeOptionsByStrike(calls, puts) {
    const strikeMap = new Map();

    // Add calls to map
    calls.forEach(call => {
        if (!strikeMap.has(call.strike)) {
            strikeMap.set(call.strike, { strike: call.strike });
        }
        strikeMap.get(call.strike).call = call;
    });

    // Add puts to map
    puts.forEach(put => {
        if (!strikeMap.has(put.strike)) {
            strikeMap.set(put.strike, { strike: put.strike });
        }
        strikeMap.get(put.strike).put = put;
    });

    // Convert to array and sort by strike
    return Array.from(strikeMap.values()).sort((a, b) => a.strike - b.strike);
}

/**
 * Check if a contract is in current positions
 * @param {number} conid - Contract ID
 * @returns {Object|null} Position object if found, null otherwise
 */
function findPositionByConid(conid) {
    const positions = state.get('positions') || [];
    return positions.find(pos => pos.conid === conid) || null;
}

/**
 * Render option chain table
 * @param {Object} optionChain - Option chain data with calls and puts
 */
function renderOptionChainTable(optionChain) {
    const tableBody = getById('option-chain-body');

    if (!tableBody) {
        return;
    }

    // Clear existing rows
    clearChildren(tableBody);

    // Merge calls and puts by strike
    const merged = mergeOptionsByStrike(optionChain.calls, optionChain.puts);

    // Render each row
    merged.forEach(row => {
        const tr = createElement('tr');

        // Call columns
        if (row.call) {
            tr.appendChild(createPriceCell(row.call.bid));
            tr.appendChild(createPriceCell(row.call.ask));
            tr.appendChild(createActionCell(row.call, 'C'));
        } else {
            tr.appendChild(createElement('td', {}, '-'));
            tr.appendChild(createElement('td', {}, '-'));
            tr.appendChild(createElement('td', {}, '-'));
        }

        // Strike column
        tr.appendChild(createStrikeCell(row.strike));

        // Put columns
        if (row.put) {
            tr.appendChild(createActionCell(row.put, 'P'));
            tr.appendChild(createPriceCell(row.put.bid));
            tr.appendChild(createPriceCell(row.put.ask));
        } else {
            tr.appendChild(createElement('td', {}, '-'));
            tr.appendChild(createElement('td', {}, '-'));
            tr.appendChild(createElement('td', {}, '-'));
        }

        tableBody.appendChild(tr);
    });
}

/**
 * Create price cell with formatted value
 * @param {number|null} price - Price value
 * @returns {HTMLElement} Table cell
 */
function createPriceCell(price) {
    const value = price !== null && price !== undefined
        ? formatPrice(price, 2)
        : '-';
    return createElement('td', {}, value);
}

/**
 * Create strike price cell
 * @param {number} strike - Strike price
 * @returns {HTMLElement} Table cell with strike-price class
 */
function createStrikeCell(strike) {
    return createElement('td', { class: 'strike-price' }, formatStrike(strike));
}

/**
 * Create action cell with Add button
 * @param {Object} contract - Option contract data
 * @param {string} right - 'C' for call, 'P' for put
 * @returns {HTMLElement} Table cell with action buttons
 */
function createActionCell(contract, right) {
    const td = createElement('td');

    // Check if this contract is already in positions
    const existingPosition = findPositionByConid(contract.conid);

    if (existingPosition) {
        // Show indicator that position exists
        const indicator = createElement(
            'span',
            { class: 'position-indicator' },
            `${formatQuantity(existingPosition.quantity)}`
        );
        td.appendChild(indicator);
    } else {
        // Show Add button
        const addButton = createElement(
            'button',
            {
                class: 'btn-add-position',
                'data-conid': contract.conid,
                'data-strike': contract.strike,
                'data-right': right,
            },
            'Add'
        );

        // Click handler to add position
        on(addButton, 'click', async (e) => {
            e.preventDefault();
            const conid = parseInt(addButton.dataset.conid);

            // Prompt for quantity
            const quantityStr = prompt('Enter quantity (positive for long, negative for short):', '1');
            if (quantityStr === null) {
                return; // User cancelled
            }

            const quantity = parseInt(quantityStr);
            if (isNaN(quantity) || quantity === 0) {
                showError('Invalid quantity. Must be a non-zero integer.');
                return;
            }

            try {
                await addPosition(conid, quantity);
            } catch (error) {
                // Error already handled by API client
                console.error('Failed to add position:', error);
            }
        });

        td.appendChild(addButton);
    }

    return td;
}

/**
 * Clear option chain display
 */
function clearOptionChain() {
    const tableBody = getById('option-chain-body');
    const optionChainSection = getById('option-chain-section');

    if (tableBody) {
        clearChildren(tableBody);
    }

    if (optionChainSection) {
        hide(optionChainSection);
    }
}

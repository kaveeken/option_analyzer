/**
 * Data Formatting Utilities
 *
 * Helper functions for formatting numbers, currency, and percentages
 */

/**
 * Format a number as currency
 * @param {number} value - Numeric value
 * @param {string} currency - Currency code (default: 'USD')
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} Formatted currency string
 */
function formatCurrency(value, currency = 'USD', decimals = 2) {
    if (value === null || value === undefined || isNaN(value)) {
        return 'N/A';
    }

    const formatter = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: currency,
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
    });

    return formatter.format(value);
}

/**
 * Format a number as percentage
 * @param {number} value - Numeric value (0.0 to 1.0 for 0% to 100%)
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} Formatted percentage string
 */
function formatPercent(value, decimals = 2) {
    if (value === null || value === undefined || isNaN(value)) {
        return 'N/A';
    }

    const formatter = new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
    });

    return formatter.format(value);
}

/**
 * Format a number with thousands separators
 * @param {number} value - Numeric value
 * @param {number} decimals - Number of decimal places (default: 0)
 * @returns {string} Formatted number string
 */
function formatNumber(value, decimals = 0) {
    if (value === null || value === undefined || isNaN(value)) {
        return 'N/A';
    }

    const formatter = new Intl.NumberFormat('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
    });

    return formatter.format(value);
}

/**
 * Format a price value (currency without symbol)
 * @param {number} value - Price value
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} Formatted price string
 */
function formatPrice(value, decimals = 2) {
    if (value === null || value === undefined || isNaN(value)) {
        return 'N/A';
    }

    return formatNumber(value, decimals);
}

/**
 * Format a strike price
 * @param {number} value - Strike price
 * @returns {string} Formatted strike price
 */
function formatStrike(value) {
    if (value === null || value === undefined || isNaN(value)) {
        return 'N/A';
    }

    // Use 0 decimals for whole numbers, 2 for decimals
    const decimals = value % 1 === 0 ? 0 : 2;
    return formatNumber(value, decimals);
}

/**
 * Format a date string
 * @param {string|Date} date - Date to format
 * @param {string} format - Format style ('short', 'medium', 'long')
 * @returns {string} Formatted date string
 */
function formatDate(date, format = 'short') {
    if (!date) {
        return 'N/A';
    }

    const dateObj = typeof date === 'string' ? new Date(date) : date;

    if (isNaN(dateObj.getTime())) {
        return 'Invalid Date';
    }

    const options = {
        short: { year: 'numeric', month: 'numeric', day: 'numeric' },
        medium: { year: 'numeric', month: 'short', day: 'numeric' },
        long: { year: 'numeric', month: 'long', day: 'numeric' },
    };

    const formatter = new Intl.DateTimeFormat('en-US', options[format] || options.short);
    return formatter.format(dateObj);
}

/**
 * Format quantity with sign (+/-)
 * @param {number} value - Quantity value
 * @returns {string} Formatted quantity with sign
 */
function formatQuantity(value) {
    if (value === null || value === undefined || isNaN(value)) {
        return 'N/A';
    }

    const sign = value > 0 ? '+' : '';
    return `${sign}${value}`;
}

/**
 * Format position type (Call/Put) based on right
 * @param {string} right - Option right ('C' or 'P')
 * @returns {string} Formatted position type
 */
function formatPositionType(right) {
    return right === 'C' ? 'Call' : right === 'P' ? 'Put' : 'Unknown';
}

/**
 * Truncate text to max length with ellipsis
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length
 * @returns {string} Truncated text
 */
function truncate(text, maxLength = 50) {
    if (!text || text.length <= maxLength) {
        return text;
    }
    return text.slice(0, maxLength - 3) + '...';
}

/**
 * Format large numbers with K, M, B suffixes
 * @param {number} value - Numeric value
 * @param {number} decimals - Number of decimal places (default: 1)
 * @returns {string} Formatted compact number
 */
function formatCompact(value, decimals = 1) {
    if (value === null || value === undefined || isNaN(value)) {
        return 'N/A';
    }

    const absValue = Math.abs(value);
    const sign = value < 0 ? '-' : '';

    if (absValue >= 1e9) {
        return sign + (absValue / 1e9).toFixed(decimals) + 'B';
    } else if (absValue >= 1e6) {
        return sign + (absValue / 1e6).toFixed(decimals) + 'M';
    } else if (absValue >= 1e3) {
        return sign + (absValue / 1e3).toFixed(decimals) + 'K';
    } else {
        return sign + absValue.toFixed(decimals);
    }
}

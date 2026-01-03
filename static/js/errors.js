/**
 * Error Handling and Loading Utilities
 *
 * Functions for displaying errors and managing loading states
 */

/**
 * Error code to user-friendly message mappings
 */
const ERROR_MESSAGES = {
    // Client errors (400s)
    NO_EXPIRATIONS_AVAILABLE: 'No option expirations available for this symbol.',
    INVALID_TARGET_DATE: 'Invalid expiration date selected.',
    POSITIONS_EXIST: 'Cannot change expiration while positions exist. Please delete all positions first.',
    DUPLICATE_POSITION: 'This position already exists. Use modify to change the quantity.',
    POSITION_NOT_FOUND: 'Position not found.',
    CONTRACT_NOT_FOUND: 'Option contract not found in the selected expiration.',
    INVALID_QUANTITY: 'Quantity cannot be zero.',
    MIXED_EXPIRATIONS: 'All positions must have the same expiration date.',
    VALIDATION_ERROR: 'Invalid input. Please check your data and try again.',

    // Server errors (500s)
    IBKR_API_ERROR: 'Unable to connect to Interactive Brokers. Please try again.',
    IBKR_TIMEOUT: 'Request to Interactive Brokers timed out. Please try again.',
    IBKR_RATE_LIMIT: 'Too many requests to Interactive Brokers. Please wait a moment.',
    SYMBOL_NOT_FOUND: 'Stock symbol not found.',
    INSUFFICIENT_DATA: 'Insufficient historical data for analysis.',

    // Session errors
    SESSION_EXPIRED: 'Your session has expired. Please start a new strategy.',
    NO_STRATEGY: 'No strategy initialized. Please load a stock symbol first.',
};

/**
 * HTTP status code to user-friendly message mappings
 */
const STATUS_MESSAGES = {
    400: 'Invalid request. Please check your input.',
    401: 'Session expired. Please refresh and try again.',
    404: 'Resource not found.',
    429: 'Too many requests. Please wait a moment and try again.',
    500: 'Server error. Please try again later.',
    502: 'Service temporarily unavailable. Please try again.',
    503: 'Service temporarily unavailable. Please try again.',
};

/**
 * Get user-friendly error message from error object
 * @param {Error} error - Error object with message, code, and status
 * @returns {string} User-friendly error message
 */
function getErrorMessage(error) {
    // If error has a code, try to get mapped message
    if (error.code && ERROR_MESSAGES[error.code]) {
        return ERROR_MESSAGES[error.code];
    }

    // If error has a status code, try to get status message
    if (error.status && STATUS_MESSAGES[error.status]) {
        return STATUS_MESSAGES[error.status];
    }

    // Fall back to error message or generic message
    return error.message || 'An unexpected error occurred. Please try again.';
}

/**
 * Show error banner with message
 * @param {string|Error} error - Error message or Error object
 */
function showError(error) {
    const errorBanner = getById('error-banner');
    const errorMessage = getById('error-message');

    if (!errorBanner || !errorMessage) {
        console.error('Error banner elements not found');
        return;
    }

    // Get user-friendly message
    const message = typeof error === 'string'
        ? error
        : getErrorMessage(error);

    // Update message and show banner
    setText(errorMessage, message);
    removeClass(errorBanner, 'hidden');

    // Auto-hide after 10 seconds
    setTimeout(() => {
        hideError();
    }, 10000);
}

/**
 * Hide error banner
 */
function hideError() {
    const errorBanner = getById('error-banner');
    if (errorBanner) {
        addClass(errorBanner, 'hidden');
    }
}

/**
 * Clear error banner
 * Alias for hideError for consistency with state.clearError()
 */
function clearError() {
    hideError();
}

/**
 * Handle API error
 * Maps error to user-friendly message and displays it
 * @param {Error} error - Error from API call
 */
function handleApiError(error) {
    console.error('API Error:', error);

    // Show error banner with user-friendly message
    showError(error);

    // Update state
    if (typeof state !== 'undefined') {
        state.setError(getErrorMessage(error));
    }
}

/**
 * Show loading overlay with message
 * @param {string} message - Loading message to display
 */
function showLoading(message = 'Loading...') {
    const loadingOverlay = getById('loading-overlay');
    const loadingMessage = getById('loading-message');

    if (!loadingOverlay) {
        console.error('Loading overlay not found');
        return;
    }

    // Update message if provided
    if (loadingMessage && message) {
        setText(loadingMessage, message);
    }

    // Show overlay
    removeClass(loadingOverlay, 'hidden');

    // Update state
    if (typeof state !== 'undefined') {
        state.setLoading(true, message);
    }
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    const loadingOverlay = getById('loading-overlay');
    if (loadingOverlay) {
        addClass(loadingOverlay, 'hidden');
    }

    // Update state
    if (typeof state !== 'undefined') {
        state.setLoading(false, '');
    }
}

/**
 * Initialize error handling
 * Sets up event listeners for error banner close button
 */
function initErrorHandling() {
    // Close button handler
    const closeButton = getById('error-close');
    if (closeButton) {
        on(closeButton, 'click', () => {
            hideError();
            if (typeof state !== 'undefined') {
                state.clearError();
            }
        });
    }

    // Subscribe to state changes for error/loading
    if (typeof state !== 'undefined') {
        state.subscribe((newState, changedKeys) => {
            // Handle error state changes
            if (changedKeys.includes('error')) {
                if (newState.error) {
                    showError(newState.error);
                } else {
                    hideError();
                }
            }

            // Handle loading state changes
            if (changedKeys.includes('isLoading') || changedKeys.includes('loadingMessage')) {
                if (newState.isLoading) {
                    showLoading(newState.loadingMessage);
                } else {
                    hideLoading();
                }
            }
        });
    }
}

/**
 * Wrap async function with error handling
 * Automatically catches and displays errors
 * @param {Function} fn - Async function to wrap
 * @returns {Function} Wrapped function
 */
function withErrorHandling(fn) {
    return async function(...args) {
        try {
            return await fn.apply(this, args);
        } catch (error) {
            handleApiError(error);
            throw error;
        }
    };
}

/**
 * Show success message (using error banner with success styling)
 * @param {string} message - Success message
 */
function showSuccess(message) {
    const errorBanner = getById('error-banner');
    const errorMessage = getById('error-message');

    if (!errorBanner || !errorMessage) {
        return;
    }

    // Temporarily add success class (you can style this in CSS)
    addClass(errorBanner, 'success');
    setText(errorMessage, message);
    removeClass(errorBanner, 'hidden');

    // Auto-hide after 3 seconds and remove success class
    setTimeout(() => {
        hideError();
        removeClass(errorBanner, 'success');
    }, 3000);
}

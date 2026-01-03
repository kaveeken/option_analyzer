/**
 * DOM Manipulation Utilities
 *
 * Helper functions for creating and manipulating DOM elements
 */

/**
 * Create a DOM element with attributes and children
 * @param {string} tag - HTML tag name
 * @param {Object} attributes - Element attributes (e.g., {class: 'btn', id: 'submit'})
 * @param {Array|string|HTMLElement} children - Child elements or text content
 * @returns {HTMLElement} Created element
 */
function createElement(tag, attributes = {}, children = []) {
    const element = document.createElement(tag);

    // Set attributes
    setAttributes(element, attributes);

    // Add children
    if (!Array.isArray(children)) {
        children = [children];
    }

    children.forEach(child => {
        if (typeof child === 'string') {
            element.appendChild(document.createTextNode(child));
        } else if (child instanceof HTMLElement) {
            element.appendChild(child);
        }
    });

    return element;
}

/**
 * Set multiple attributes on an element
 * @param {HTMLElement} element - Target element
 * @param {Object} attributes - Attributes to set
 */
function setAttributes(element, attributes) {
    Object.entries(attributes).forEach(([key, value]) => {
        if (key === 'class') {
            element.className = value;
        } else if (key === 'style' && typeof value === 'object') {
            Object.assign(element.style, value);
        } else if (key.startsWith('data-')) {
            const dataKey = key.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
            element.dataset[dataKey] = value;
        } else if (typeof value === 'boolean') {
            if (value) {
                element.setAttribute(key, '');
            }
        } else {
            element.setAttribute(key, value);
        }
    });
}

/**
 * Remove all child elements from a parent
 * @param {HTMLElement} element - Parent element to clear
 */
function clearChildren(element) {
    while (element.firstChild) {
        element.removeChild(element.firstChild);
    }
}

/**
 * Toggle a CSS class on an element
 * @param {HTMLElement} element - Target element
 * @param {string} className - Class name to toggle
 * @param {boolean} force - Optional force add (true) or remove (false)
 */
function toggleClass(element, className, force) {
    if (force === undefined) {
        element.classList.toggle(className);
    } else {
        element.classList.toggle(className, force);
    }
}

/**
 * Add a CSS class to an element
 * @param {HTMLElement} element - Target element
 * @param {string} className - Class name to add
 */
function addClass(element, className) {
    element.classList.add(className);
}

/**
 * Remove a CSS class from an element
 * @param {HTMLElement} element - Target element
 * @param {string} className - Class name to remove
 */
function removeClass(element, className) {
    element.classList.remove(className);
}

/**
 * Check if element has a CSS class
 * @param {HTMLElement} element - Target element
 * @param {string} className - Class name to check
 * @returns {boolean} True if element has the class
 */
function hasClass(element, className) {
    return element.classList.contains(className);
}

/**
 * Show an element (remove 'hidden' class)
 * @param {HTMLElement} element - Element to show
 */
function show(element) {
    removeClass(element, 'hidden');
}

/**
 * Hide an element (add 'hidden' class)
 * @param {HTMLElement} element - Element to hide
 */
function hide(element) {
    addClass(element, 'hidden');
}

/**
 * Get element by ID (shorthand for document.getElementById)
 * @param {string} id - Element ID
 * @returns {HTMLElement} Element or null
 */
function getById(id) {
    return document.getElementById(id);
}

/**
 * Query selector (shorthand for document.querySelector)
 * @param {string} selector - CSS selector
 * @returns {HTMLElement} Element or null
 */
function query(selector) {
    return document.querySelector(selector);
}

/**
 * Query selector all (shorthand for document.querySelectorAll)
 * @param {string} selector - CSS selector
 * @returns {NodeList} List of elements
 */
function queryAll(selector) {
    return document.querySelectorAll(selector);
}

/**
 * Add event listener to element
 * @param {HTMLElement} element - Target element
 * @param {string} event - Event name
 * @param {Function} handler - Event handler
 * @param {Object} options - Event listener options
 */
function on(element, event, handler, options) {
    element.addEventListener(event, handler, options);
}

/**
 * Remove event listener from element
 * @param {HTMLElement} element - Target element
 * @param {string} event - Event name
 * @param {Function} handler - Event handler
 */
function off(element, event, handler) {
    element.removeEventListener(event, handler);
}

/**
 * Set text content of an element
 * @param {HTMLElement} element - Target element
 * @param {string} text - Text content
 */
function setText(element, text) {
    element.textContent = text;
}

/**
 * Set HTML content of an element
 * @param {HTMLElement} element - Target element
 * @param {string} html - HTML content
 */
function setHTML(element, html) {
    element.innerHTML = html;
}

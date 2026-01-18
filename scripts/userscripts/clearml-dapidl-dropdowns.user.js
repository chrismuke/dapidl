// ==UserScript==
// @name         ClearML DAPIDL Parameter Dropdowns
// @namespace    https://github.com/chrismuke/dapidl
// @version      1.0.0
// @description  Enhance ClearML UI with dropdown selects for DAPIDL pipeline parameters
// @author       DAPIDL Project
// @match        *://app.clear.ml/*
// @match        *://localhost:8080/*
// @match        *://localhost:8008/*
// @match        *://*.clear.ml/*
// @grant        GM_addStyle
// @run-at       document-idle
// ==/UserScript==

(function() {
    'use strict';

    // ===========================================
    // DAPIDL Parameter Definitions
    // ===========================================
    // Define allowed values for each parameter
    // Format: { paramName: { values: [...], description: "..." } }

    const DAPIDL_PARAMS = {
        // Platform selection
        'platform': {
            values: ['auto', 'xenium', 'merscope'],
            description: 'Spatial transcriptomics platform'
        },

        // Segmentation options
        'segmenter': {
            values: ['cellpose', 'native'],
            description: 'Nucleus segmentation method'
        },

        // Annotation options
        'annotator': {
            values: ['celltypist', 'ground_truth', 'popv'],
            description: 'Cell type annotation method'
        },
        'strategy': {
            values: ['single', 'consensus', 'hierarchical'],
            description: 'CellTypist voting strategy'
        },
        'annotation_strategy': {
            values: ['single', 'consensus', 'hierarchical'],
            description: 'CellTypist voting strategy'
        },

        // Patch extraction
        'patch_size': {
            values: ['32', '64', '128', '256'],
            description: 'Patch size in pixels'
        },
        'output_format': {
            values: ['lmdb', 'zarr'],
            description: 'Output dataset format'
        },
        'normalization': {
            values: ['adaptive', 'fixed', 'none'],
            description: 'Image normalization method'
        },
        'normalization_method': {
            values: ['adaptive', 'fixed', 'none'],
            description: 'Image normalization method'
        },

        // Training
        'backbone': {
            values: [
                'efficientnetv2_rw_s',
                'efficientnet_b0',
                'efficientnet_b1',
                'convnext_tiny',
                'convnext_small',
                'resnet50',
                'resnet34'
            ],
            description: 'CNN backbone architecture'
        },

        // Boolean options (rendered as dropdown)
        'fine_grained': {
            values: ['true', 'false'],
            description: 'Use fine-grained cell types'
        },
        'majority_voting': {
            values: ['true', 'false'],
            description: 'Enable majority voting'
        },
        'extended_consensus': {
            values: ['true', 'false'],
            description: 'Use extended consensus (6 models)'
        },
        'use_dali': {
            values: ['true', 'false'],
            description: 'Use NVIDIA DALI for data loading'
        },
        'pretrained': {
            values: ['true', 'false'],
            description: 'Use pretrained weights'
        },

        // Documentation
        'doc_template': {
            values: ['default', 'minimal', 'detailed'],
            description: 'Documentation template style'
        },
        'template': {
            values: ['default', 'minimal', 'detailed'],
            description: 'Documentation template style'
        },

        // Transfer platform
        'target_platform': {
            values: ['auto', 'xenium', 'merscope'],
            description: 'Target platform for transfer'
        },
        'transfer_target_platform': {
            values: ['auto', 'xenium', 'merscope'],
            description: 'Target platform for transfer'
        }
    };

    // ===========================================
    // Custom Styles
    // ===========================================

    GM_addStyle(`
        /* Dropdown styling to match ClearML UI */
        .dapidl-dropdown {
            background-color: #1a1a2e;
            color: #e0e0e0;
            border: 1px solid #3a3a5a;
            border-radius: 4px;
            padding: 6px 10px;
            font-size: 13px;
            font-family: inherit;
            min-width: 150px;
            cursor: pointer;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        .dapidl-dropdown:hover {
            border-color: #5a5a8a;
        }

        .dapidl-dropdown:focus {
            outline: none;
            border-color: #7b68ee;
            box-shadow: 0 0 0 2px rgba(123, 104, 238, 0.2);
        }

        .dapidl-dropdown option {
            background-color: #1a1a2e;
            color: #e0e0e0;
            padding: 8px;
        }

        /* Tooltip for parameter description */
        .dapidl-param-wrapper {
            position: relative;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .dapidl-info-icon {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #3a3a5a;
            color: #a0a0c0;
            font-size: 11px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            cursor: help;
        }

        .dapidl-info-icon:hover {
            background: #5a5a8a;
            color: #e0e0e0;
        }

        .dapidl-tooltip {
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: #2a2a4e;
            color: #e0e0e0;
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 12px;
            white-space: nowrap;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.2s, visibility 0.2s;
            z-index: 1000;
            margin-bottom: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }

        .dapidl-info-icon:hover + .dapidl-tooltip,
        .dapidl-tooltip:hover {
            opacity: 1;
            visibility: visible;
        }

        /* Badge to indicate enhanced parameter */
        .dapidl-badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 9px;
            padding: 2px 6px;
            border-radius: 10px;
            margin-left: 6px;
            font-weight: 600;
            letter-spacing: 0.5px;
        }

        /* Validation error styling */
        .dapidl-dropdown.invalid {
            border-color: #e74c3c;
            box-shadow: 0 0 0 2px rgba(231, 76, 60, 0.2);
        }
    `);

    // ===========================================
    // Core Functions
    // ===========================================

    /**
     * Extract parameter name from a label or input element
     */
    function extractParamName(element) {
        // Try various methods to find the parameter name
        const possibleSources = [
            // Direct text content
            element.textContent?.trim(),
            // aria-label
            element.getAttribute('aria-label'),
            // placeholder
            element.getAttribute('placeholder'),
            // name attribute
            element.getAttribute('name'),
            // Previous sibling label
            element.previousElementSibling?.textContent?.trim(),
            // Parent's label child
            element.closest('.param-row, .parameter-row, [class*="param"]')
                ?.querySelector('label, .label, [class*="label"], [class*="name"]')
                ?.textContent?.trim()
        ];

        for (const source of possibleSources) {
            if (source) {
                // Clean up the name - remove colons, whitespace, etc.
                const cleanName = source
                    .replace(/[:\s]+$/, '')
                    .replace(/^step_config\//, '')
                    .replace(/^General\//, '')
                    .replace(/^Args\//, '')
                    .trim()
                    .toLowerCase();

                // Check if this matches any of our parameters
                for (const paramKey of Object.keys(DAPIDL_PARAMS)) {
                    if (cleanName === paramKey.toLowerCase() ||
                        cleanName.endsWith('/' + paramKey.toLowerCase()) ||
                        cleanName.includes(paramKey.toLowerCase())) {
                        return paramKey;
                    }
                }
            }
        }
        return null;
    }

    /**
     * Create a dropdown element for a parameter
     */
    function createDropdown(paramName, currentValue, originalInput) {
        const paramConfig = DAPIDL_PARAMS[paramName];
        if (!paramConfig) return null;

        // Create wrapper
        const wrapper = document.createElement('div');
        wrapper.className = 'dapidl-param-wrapper';

        // Create select element
        const select = document.createElement('select');
        select.className = 'dapidl-dropdown';
        select.dataset.dapidlParam = paramName;
        select.dataset.originalInputId = originalInput.id || '';

        // Add options
        paramConfig.values.forEach(value => {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = value;

            // Handle case-insensitive comparison for current value
            const normalizedCurrent = String(currentValue).toLowerCase().trim();
            const normalizedValue = String(value).toLowerCase().trim();
            if (normalizedCurrent === normalizedValue) {
                option.selected = true;
            }

            select.appendChild(option);
        });

        // If current value is not in options, add it as first option (invalid)
        const normalizedCurrent = String(currentValue).toLowerCase().trim();
        const hasMatch = paramConfig.values.some(v =>
            String(v).toLowerCase().trim() === normalizedCurrent
        );

        if (currentValue && !hasMatch) {
            const invalidOption = document.createElement('option');
            invalidOption.value = currentValue;
            invalidOption.textContent = `${currentValue} (invalid)`;
            invalidOption.selected = true;
            invalidOption.style.color = '#e74c3c';
            select.insertBefore(invalidOption, select.firstChild);
            select.classList.add('invalid');
        }

        // Event handler to sync with original input
        select.addEventListener('change', () => {
            const newValue = select.value;

            // Update original input
            originalInput.value = newValue;

            // Trigger input event for Angular to pick up
            originalInput.dispatchEvent(new Event('input', { bubbles: true }));
            originalInput.dispatchEvent(new Event('change', { bubbles: true }));

            // Also try to trigger Angular-specific events
            const ngModelEvent = new CustomEvent('ngModelChange', {
                bubbles: true,
                detail: newValue
            });
            originalInput.dispatchEvent(ngModelEvent);

            // Remove invalid class if selecting a valid option
            if (paramConfig.values.includes(newValue)) {
                select.classList.remove('invalid');
                // Remove invalid option if it exists
                const invalidOpt = select.querySelector('option[style*="color"]');
                if (invalidOpt) invalidOpt.remove();
            }

            console.log(`[DAPIDL] Updated ${paramName}: ${newValue}`);
        });

        wrapper.appendChild(select);

        // Add info icon with tooltip
        const infoIcon = document.createElement('span');
        infoIcon.className = 'dapidl-info-icon';
        infoIcon.textContent = '?';

        const tooltip = document.createElement('span');
        tooltip.className = 'dapidl-tooltip';
        tooltip.textContent = paramConfig.description;

        wrapper.appendChild(infoIcon);
        wrapper.appendChild(tooltip);

        return wrapper;
    }

    /**
     * Process an input element and convert to dropdown if applicable
     */
    function processInput(input) {
        // Skip if already processed
        if (input.dataset.dapidlProcessed === 'true') return;

        // Skip if not a text input
        if (input.type !== 'text' && input.tagName !== 'INPUT') return;

        // Try to find parameter name
        const paramName = extractParamName(input);
        if (!paramName || !DAPIDL_PARAMS[paramName]) return;

        // Mark as processed
        input.dataset.dapidlProcessed = 'true';

        // Create dropdown
        const dropdown = createDropdown(paramName, input.value, input);
        if (!dropdown) return;

        // Insert dropdown after input
        input.parentNode.insertBefore(dropdown, input.nextSibling);

        // Hide original input but keep it for form submission
        input.style.display = 'none';

        // Add badge to label if found
        const label = input.closest('.param-row, .parameter-row, [class*="param"]')
            ?.querySelector('label, .label, [class*="label"]');
        if (label && !label.querySelector('.dapidl-badge')) {
            const badge = document.createElement('span');
            badge.className = 'dapidl-badge';
            badge.textContent = 'DAPIDL';
            label.appendChild(badge);
        }

        console.log(`[DAPIDL] Enhanced parameter: ${paramName}`);
    }

    /**
     * Scan the page for parameter inputs
     */
    function scanForParameters() {
        // Target various ClearML UI elements that might contain parameters
        const selectors = [
            // Hyperparameters section
            'input[type="text"]',
            '.hyper-param input',
            '.parameter-value input',
            '[class*="param"] input',
            '[class*="config"] input',
            // Pipeline parameters
            '.pipeline-param input',
            '[class*="pipeline"] input',
            // General form inputs in configuration sections
            '.configuration input',
            '[class*="Configuration"] input'
        ];

        selectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(processInput);
        });
    }

    // ===========================================
    // Initialization
    // ===========================================

    /**
     * Initialize the userscript
     */
    function init() {
        console.log('[DAPIDL] ClearML Parameter Enhancement loaded');

        // Initial scan
        setTimeout(scanForParameters, 1000);

        // Set up MutationObserver for dynamic content
        const observer = new MutationObserver((mutations) => {
            let shouldScan = false;

            for (const mutation of mutations) {
                if (mutation.addedNodes.length > 0) {
                    shouldScan = true;
                    break;
                }
            }

            if (shouldScan) {
                // Debounce scans
                clearTimeout(window.dapidlScanTimeout);
                window.dapidlScanTimeout = setTimeout(scanForParameters, 200);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        // Also scan on URL changes (SPA navigation)
        let lastUrl = location.href;
        setInterval(() => {
            if (location.href !== lastUrl) {
                lastUrl = location.href;
                console.log('[DAPIDL] URL changed, rescanning...');
                setTimeout(scanForParameters, 500);
            }
        }, 1000);
    }

    // Start when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();

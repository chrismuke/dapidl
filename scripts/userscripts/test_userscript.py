#!/usr/bin/env python
"""Test the ClearML DAPIDL userscript with headless Firefox.

This script:
1. Creates a mock HTML page simulating ClearML's parameter UI
2. Loads the userscript into the page
3. Verifies dropdown conversion works correctly
"""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


# Mock HTML that simulates ClearML's parameter UI structure
MOCK_CLEARML_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>ClearML Mock - Pipeline Parameters</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            padding: 20px;
        }
        h1 { color: #7b68ee; }
        .configuration {
            background: #1a1a2e;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .param-row {
            display: flex;
            align-items: center;
            margin: 12px 0;
            gap: 16px;
        }
        .param-row label {
            width: 200px;
            font-weight: 500;
            color: #a0a0c0;
        }
        .param-row input {
            background: #2a2a4e;
            border: 1px solid #3a3a5a;
            border-radius: 4px;
            padding: 8px 12px;
            color: #e0e0e0;
            font-size: 14px;
            min-width: 200px;
        }
        .param-row input:focus {
            outline: none;
            border-color: #7b68ee;
        }
        .section-title {
            color: #7b68ee;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 16px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        #results {
            margin-top: 30px;
            padding: 20px;
            background: #1a2a1a;
            border-radius: 8px;
            font-family: monospace;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <h1>ClearML Mock - DAPIDL Pipeline Parameters</h1>

    <div class="configuration">
        <div class="section-title">Pipeline Parameters</div>

        <div class="param-row">
            <label>platform</label>
            <input type="text" id="param-platform" value="auto">
        </div>

        <div class="param-row">
            <label>segmenter</label>
            <input type="text" id="param-segmenter" value="cellpose">
        </div>

        <div class="param-row">
            <label>annotator</label>
            <input type="text" id="param-annotator" value="celltypist">
        </div>

        <div class="param-row">
            <label>patch_size</label>
            <input type="text" id="param-patch_size" value="128">
        </div>

        <div class="param-row">
            <label>backbone</label>
            <input type="text" id="param-backbone" value="efficientnetv2_rw_s">
        </div>

        <div class="param-row">
            <label>fine_grained</label>
            <input type="text" id="param-fine_grained" value="false">
        </div>

        <div class="param-row">
            <label>strategy</label>
            <input type="text" id="param-strategy" value="consensus">
        </div>

        <div class="param-row">
            <label>normalization</label>
            <input type="text" id="param-normalization" value="adaptive">
        </div>
    </div>

    <div class="configuration">
        <div class="section-title">Test Cases</div>

        <div class="param-row">
            <label>segmenter (invalid)</label>
            <input type="text" id="param-segmenter-invalid" value="stardist">
        </div>

        <div class="param-row">
            <label>epochs (not a dropdown param)</label>
            <input type="text" id="param-epochs" value="50">
        </div>
    </div>

    <div id="results">Test results will appear here...</div>

    <script>
        // Function to collect test results
        window.collectResults = function() {
            const results = {
                dropdownsCreated: document.querySelectorAll('.dapidl-dropdown').length,
                badgesAdded: document.querySelectorAll('.dapidl-badge').length,
                hiddenInputs: document.querySelectorAll('input[style*="display: none"]').length,
                parameters: {}
            };

            document.querySelectorAll('.dapidl-dropdown').forEach(select => {
                const paramName = select.dataset.dapidlParam;
                results.parameters[paramName] = {
                    currentValue: select.value,
                    options: Array.from(select.options).map(o => o.value),
                    hasInvalidClass: select.classList.contains('invalid')
                };
            });

            return results;
        };

        // Function to change a dropdown value
        window.changeDropdownValue = function(paramName, newValue) {
            const select = document.querySelector(`select[data-dapidl-param="${paramName}"]`);
            if (select) {
                select.value = newValue;
                select.dispatchEvent(new Event('change', { bubbles: true }));
                return true;
            }
            return false;
        };

        // Function to get original input value
        window.getOriginalInputValue = function(paramName) {
            const input = document.querySelector(`input[id*="${paramName}"]`);
            return input ? input.value : null;
        };
    </script>
</body>
</html>
"""


async def test_userscript():
    """Run the userscript test."""
    print("=" * 60)
    print("ClearML DAPIDL Userscript Test")
    print("=" * 60)

    # Load userscript content
    userscript_path = Path(__file__).parent / "clearml-dapidl-dropdowns.user.js"
    userscript_content = userscript_path.read_text()

    # Remove userscript metadata header for injection
    # Find the end of the metadata block
    metadata_end = userscript_content.find("// ==/UserScript==")
    if metadata_end != -1:
        # Skip the metadata and get the actual code
        code_start = userscript_content.find("(function()", metadata_end)
        if code_start != -1:
            userscript_code = userscript_content[code_start:]
        else:
            userscript_code = userscript_content
    else:
        userscript_code = userscript_content

    # We need to provide GM_addStyle function
    gm_addstyle_shim = """
    window.GM_addStyle = function(css) {
        const style = document.createElement('style');
        style.textContent = css;
        document.head.appendChild(style);
    };
    """

    async with async_playwright() as p:
        # Launch Firefox in headless mode
        print("\n[1/5] Launching headless Firefox...")
        browser = await p.firefox.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # Listen to console messages
        page.on("console", lambda msg: print(f"      [Browser] {msg.text}"))

        # Create mock page
        print("[2/5] Loading mock ClearML page...")
        await page.set_content(MOCK_CLEARML_HTML)

        # Inject GM_addStyle shim
        await page.evaluate(gm_addstyle_shim)

        # Inject userscript
        print("[3/5] Injecting userscript...")
        await page.evaluate(userscript_code)

        # Wait for userscript to process
        print("[4/5] Waiting for userscript to process parameters...")
        await asyncio.sleep(2)  # Give it time to run

        # Collect results
        print("[5/5] Collecting test results...\n")
        results = await page.evaluate("window.collectResults()")

        # Print results
        print("=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"\nDropdowns created: {results['dropdownsCreated']}")
        print(f"DAPIDL badges added: {results['badgesAdded']}")
        print(f"Original inputs hidden: {results['hiddenInputs']}")

        print("\n--- Parameter Details ---")
        for param_name, details in results.get("parameters", {}).items():
            invalid_marker = " [INVALID]" if details.get("hasInvalidClass") else ""
            print(f"\n  {param_name}{invalid_marker}:")
            print(f"    Current value: {details['currentValue']}")
            print(f"    Options: {', '.join(details['options'][:5])}{'...' if len(details['options']) > 5 else ''}")

        # Test value change
        print("\n--- Testing Value Change ---")
        print("  Changing 'segmenter' from 'cellpose' to 'native'...")
        changed = await page.evaluate('window.changeDropdownValue("segmenter", "native")')
        if changed:
            original_value = await page.evaluate('window.getOriginalInputValue("segmenter")')
            print(f"  Original input value after change: {original_value}")
            if original_value == "native":
                print("  ✓ Value sync working correctly!")
            else:
                print("  ✗ Value sync FAILED!")
        else:
            print("  ✗ Could not find segmenter dropdown!")

        # Screenshot for visual verification
        screenshot_path = Path(__file__).parent / "test_screenshot.png"
        await page.screenshot(path=str(screenshot_path))
        print(f"\n[Screenshot saved to {screenshot_path}]")

        # Evaluate test success
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        expected_dropdowns = 9  # 8 valid params + 1 invalid
        success = (
            results["dropdownsCreated"] >= 8  # At least 8 dropdowns
            and results["hiddenInputs"] >= 8  # Original inputs hidden
        )

        if success:
            print("✓ All tests PASSED!")
            print(f"  - Created {results['dropdownsCreated']} dropdowns (expected >= 8)")
            print(f"  - Hidden {results['hiddenInputs']} original inputs")

            # Check invalid parameter handling
            invalid_params = [
                name for name, details in results.get("parameters", {}).items()
                if details.get("hasInvalidClass")
            ]
            if invalid_params:
                print(f"  - Correctly marked invalid parameters: {', '.join(invalid_params)}")
        else:
            print("✗ Some tests FAILED!")
            print(f"  - Created {results['dropdownsCreated']} dropdowns (expected >= 8)")
            print(f"  - Hidden {results['hiddenInputs']} original inputs (expected >= 8)")

        await browser.close()
        return success


if __name__ == "__main__":
    success = asyncio.run(test_userscript())
    exit(0 if success else 1)

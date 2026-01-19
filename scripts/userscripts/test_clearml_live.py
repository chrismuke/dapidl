#!/usr/bin/env python
"""Test the ClearML DAPIDL userscript on a live ClearML instance.

This script:
1. Launches Firefox with the userscript injected
2. Navigates to a real ClearML task page
3. Takes screenshots to verify dropdown conversion

Note: For hosted ClearML (app.clear.ml), you need to log in manually
or use browser profile with saved session.
"""

import asyncio
import os
from pathlib import Path

from playwright.async_api import async_playwright


# Userscript with GM_addStyle shim included
def get_userscript_with_shim():
    """Load userscript and add GM_addStyle shim."""
    userscript_path = Path(__file__).parent / "clearml-dapidl-dropdowns.user.js"
    userscript_content = userscript_path.read_text()

    # Find the actual code (after metadata)
    metadata_end = userscript_content.find("// ==/UserScript==")
    if metadata_end != -1:
        code_start = userscript_content.find("(function()", metadata_end)
        if code_start != -1:
            userscript_code = userscript_content[code_start:]
        else:
            userscript_code = userscript_content
    else:
        userscript_code = userscript_content

    # Add GM_addStyle shim
    gm_shim = """
    if (typeof GM_addStyle === 'undefined') {
        window.GM_addStyle = function(css) {
            const style = document.createElement('style');
            style.textContent = css;
            document.head.appendChild(style);
        };
    }
    """

    return gm_shim + "\n" + userscript_code


async def test_live_clearml(headless: bool = True, task_id: str = None):
    """Test userscript on live ClearML instance.

    Args:
        headless: Run in headless mode (default True)
        task_id: Specific task ID to navigate to
    """
    print("=" * 60)
    print("ClearML Live Userscript Test")
    print("=" * 60)

    # Default to step-annotation task which has good parameters
    if not task_id:
        task_id = "6e156d458ff2459b84462eb9ddb0f28b"  # step-annotation

    # ClearML task URL
    clearml_url = f"https://app.clear.ml/projects/*/experiments/{task_id}/execution/parameters"

    print(f"\nTarget URL: {clearml_url}")

    userscript_code = get_userscript_with_shim()

    async with async_playwright() as p:
        # Check for existing Firefox profile with ClearML session
        firefox_profile = Path.home() / ".mozilla" / "firefox"

        # Launch Firefox
        print(f"\n[1/4] Launching Firefox (headless={headless})...")

        browser = await p.firefox.launch(
            headless=headless,
            # Use slower motion for visibility in non-headless mode
            slow_mo=100 if not headless else 0,
        )

        context = await browser.new_context(
            viewport={"width": 1400, "height": 900},
            # Accept all cookies
            ignore_https_errors=True,
        )

        page = await context.new_page()

        # Listen to console
        page.on("console", lambda msg: print(f"      [Browser] {msg.text}") if "DAPIDL" in msg.text else None)

        # Inject userscript on every navigation
        async def inject_userscript():
            try:
                await page.evaluate(userscript_code)
                print("      [Injected userscript]")
            except Exception as e:
                print(f"      [Injection error: {e}]")

        # Navigate to ClearML
        print(f"[2/4] Navigating to ClearML...")
        try:
            await page.goto(clearml_url, wait_until="networkidle", timeout=30000)
        except Exception as e:
            print(f"      Navigation timeout (this is normal): {e}")

        # Wait for page to load
        await asyncio.sleep(2)

        # Check if we hit login page
        current_url = page.url
        print(f"      Current URL: {current_url}")

        if "login" in current_url.lower() or "auth" in current_url.lower():
            print("\n" + "=" * 60)
            print("LOGIN REQUIRED")
            print("=" * 60)
            print("""
The ClearML instance requires authentication.

Options:
1. Run with headless=False to log in manually:
   python test_clearml_live.py --visible

2. Use browser with saved session (Firefox profile)

3. Test with self-hosted ClearML (no auth required)
""")

            if not headless:
                print("Waiting 60 seconds for manual login...")
                await asyncio.sleep(60)
                # Re-navigate after login
                await page.goto(clearml_url, wait_until="networkidle", timeout=30000)
            else:
                # Take screenshot of login page
                screenshot_path = Path(__file__).parent / "test_clearml_login.png"
                await page.screenshot(path=str(screenshot_path))
                print(f"\n[Screenshot of login page saved to {screenshot_path}]")
                await browser.close()
                return False

        # Inject userscript
        print("[3/4] Injecting userscript...")
        await inject_userscript()

        # Wait for userscript to process
        await asyncio.sleep(3)

        # Take screenshot
        print("[4/4] Taking screenshot...")
        screenshot_path = Path(__file__).parent / "test_clearml_live.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"\n[Screenshot saved to {screenshot_path}]")

        # Check if dropdowns were created
        dropdown_count = await page.evaluate(
            "document.querySelectorAll('.dapidl-dropdown').length"
        )
        print(f"\nDropdowns created: {dropdown_count}")

        if dropdown_count > 0:
            print("✓ Userscript working on live ClearML!")

            # Get details
            details = await page.evaluate("""
                () => {
                    const results = [];
                    document.querySelectorAll('.dapidl-dropdown').forEach(select => {
                        results.push({
                            param: select.dataset.dapidlParam,
                            value: select.value,
                            options: Array.from(select.options).map(o => o.value)
                        });
                    });
                    return results;
                }
            """)

            print("\nEnhanced parameters:")
            for d in details:
                print(f"  - {d['param']}: {d['value']} (options: {', '.join(d['options'][:4])}...)")
        else:
            print("✗ No dropdowns created - userscript may need adjustment for ClearML's DOM structure")

        # Keep browser open in visible mode for inspection
        if not headless:
            print("\nBrowser open for inspection. Press Ctrl+C to close...")
            try:
                await asyncio.sleep(300)  # 5 minutes
            except KeyboardInterrupt:
                pass

        await browser.close()
        return dropdown_count > 0


async def test_with_multiple_tasks():
    """Test userscript with multiple task types."""
    tasks = {
        "step-annotation": "6e156d458ff2459b84462eb9ddb0f28b",
        "step-segmentation": "5435df58c40f4527a33fbd52f93107d2",
        "step-training": "efeba9d247104a2080afd0070ac0b56f",
        "step-patch_extraction": "8447635f9517435e871d579188697c17",
    }

    print("Testing multiple task types...\n")

    for task_name, task_id in tasks.items():
        print(f"\n--- Testing {task_name} ---")
        success = await test_live_clearml(headless=True, task_id=task_id)
        if not success:
            print(f"Stopping tests - authentication required")
            break


if __name__ == "__main__":
    import sys

    headless = "--visible" not in sys.argv
    task_id = None

    # Parse task ID from args
    for arg in sys.argv[1:]:
        if arg.startswith("--task="):
            task_id = arg.split("=")[1]

    if "--multi" in sys.argv:
        asyncio.run(test_with_multiple_tasks())
    else:
        asyncio.run(test_live_clearml(headless=headless, task_id=task_id))

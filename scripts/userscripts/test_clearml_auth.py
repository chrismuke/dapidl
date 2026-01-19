#!/usr/bin/env python
"""Test the ClearML DAPIDL userscript with API authentication.

This script authenticates using ClearML API credentials and injects
the session cookies into the browser.
"""

import asyncio
from pathlib import Path

from clearml.backend_api import Session
from playwright.async_api import async_playwright


def get_userscript_with_shim():
    """Load userscript and add GM_addStyle shim."""
    userscript_path = Path(__file__).parent / "clearml-dapidl-dropdowns.user.js"
    userscript_content = userscript_path.read_text()

    metadata_end = userscript_content.find("// ==/UserScript==")
    if metadata_end != -1:
        code_start = userscript_content.find("(function()", metadata_end)
        if code_start != -1:
            userscript_code = userscript_content[code_start:]
        else:
            userscript_code = userscript_content
    else:
        userscript_code = userscript_content

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


async def test_with_auth():
    """Test userscript with ClearML API authentication."""
    print("=" * 60)
    print("ClearML Userscript Test (with API Auth)")
    print("=" * 60)

    # Get ClearML session
    print("\n[1/6] Authenticating with ClearML API...")
    try:
        session = Session()
        # Get auth token
        auth_token = session.token
        print(f"      Got auth token: {auth_token[:20]}...")
    except Exception as e:
        print(f"      Failed to authenticate: {e}")
        return False

    # Task to test
    task_id = "6e156d458ff2459b84462eb9ddb0f28b"  # step-annotation
    clearml_url = f"https://app.clear.ml/projects/*/experiments/{task_id}/execution/parameters"

    userscript_code = get_userscript_with_shim()

    async with async_playwright() as p:
        print("[2/6] Launching Firefox...")
        browser = await p.firefox.launch(headless=True)

        context = await browser.new_context(
            viewport={"width": 1400, "height": 900},
        )

        # Set auth cookie/header
        print("[3/6] Setting authentication...")

        # ClearML uses Bearer token in Authorization header
        # We need to set cookies that ClearML web app uses
        await context.add_cookies([
            {
                "name": "clearml_token_basic",
                "value": auth_token,
                "domain": "app.clear.ml",
                "path": "/",
                "httpOnly": False,
                "secure": True,
                "sameSite": "Lax",
            },
            {
                "name": "clearml_token_bearer",
                "value": auth_token,
                "domain": "app.clear.ml",
                "path": "/",
                "httpOnly": False,
                "secure": True,
                "sameSite": "Lax",
            },
        ])

        # Also set in local storage (ClearML web app might use this)
        page = await context.new_page()

        # Listen to console
        page.on("console", lambda msg: print(f"      [Browser] {msg.text}") if "DAPIDL" in msg.text else None)

        # First navigate to ClearML to set local storage
        print("[4/6] Navigating to ClearML...")
        await page.goto("https://app.clear.ml", wait_until="domcontentloaded", timeout=30000)

        # Try to set token in localStorage
        await page.evaluate(f"""
            localStorage.setItem('_auth_token', '{auth_token}');
            localStorage.setItem('clearml_token', '{auth_token}');
        """)

        # Now navigate to the task
        await page.goto(clearml_url, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(2)

        current_url = page.url
        print(f"      Current URL: {current_url}")

        if "login" in current_url.lower():
            print("\n      Authentication via API token didn't work for web UI")
            print("      (ClearML web uses different auth than API)")

            # Take screenshot of where we ended up
            screenshot_path = Path(__file__).parent / "test_clearml_auth_result.png"
            await page.screenshot(path=str(screenshot_path))
            print(f"\n[Screenshot saved to {screenshot_path}]")

            await browser.close()
            return False

        # Inject userscript
        print("[5/6] Injecting userscript...")
        await page.evaluate(userscript_code)
        await asyncio.sleep(3)

        # Check results
        print("[6/6] Checking results...")
        dropdown_count = await page.evaluate(
            "document.querySelectorAll('.dapidl-dropdown').length"
        )

        screenshot_path = Path(__file__).parent / "test_clearml_auth_result.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"\n[Screenshot saved to {screenshot_path}]")

        print(f"\nDropdowns created: {dropdown_count}")

        if dropdown_count > 0:
            print("✓ SUCCESS! Userscript working on live ClearML!")
        else:
            print("✗ No dropdowns - check screenshot for page state")

        await browser.close()
        return dropdown_count > 0


if __name__ == "__main__":
    asyncio.run(test_with_auth())

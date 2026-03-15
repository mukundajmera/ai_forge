const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  let hasErrors = false;
  
  page.on('console', msg => {
    if (msg.type() === 'error') {
      console.error(`PAGE ERROR: ${msg.text()}`);
      hasErrors = true;
    } else {
      console.log(`PAGE LOG: ${msg.text()}`);
    }
  });

  page.on('pageerror', error => {
    console.error(`UNCAUGHT EXCEPTION: ${error.message}`);
    hasErrors = true;
  });

  console.log("Navigating to http://localhost:3000...");
  try {
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle', timeout: 10000 });
    console.log("Page loaded. Checking for Vite error overlay...");
    
    // Check if the Vite error overlay is present
    const overlay = await page.locator('vite-error-overlay').count();
    if (overlay > 0) {
      console.error("FAIL: Vite error overlay is visible! UI is broken.");
      hasErrors = true;
    } else {
      console.log("No Vite error overlay found.");
      
      // Also check if main content is rendered (e.g. #root has children)
      const contentText = await page.evaluate(() => document.body.innerText);
      console.log("Body text preview:", contentText.substring(0, 100).replace(/\n/g, ' '));
      if (!contentText.trim()) {
        console.error("FAIL: Page is blank!");
        hasErrors = true;
      }
    }
  } catch (e) {
    console.error(`Failed to load page: ${e}`);
    hasErrors = true;
  }

  await browser.close();
  
  if (hasErrors) {
    console.error("UI is broken.");
    process.exit(1);
  } else {
    console.log("UI appears to be working.");
    process.exit(0);
  }
})();

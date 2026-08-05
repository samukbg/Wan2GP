const { chromium } = require('playwright');
const { createServer } = require('vite');
const fs = require('fs');
const path = require('path');

async function render() {
  const args = process.argv.slice(2);
  const templateName = args[0];
  const outputPath = args[1];
  const propsArgIndex = args.indexOf('--props');
  
  if (!templateName || !outputPath || propsArgIndex === -1 || !args[propsArgIndex + 1]) {
    console.error("Usage: node render.js <TemplateName> <OutputPath> --props <PropsJsonPath>");
    process.exit(1);
  }
  
  const propsPath = args[propsArgIndex + 1];
  let props = {};
  if (fs.existsSync(propsPath)) {
    const rawProps = JSON.parse(fs.readFileSync(propsPath, 'utf8'));
    props = rawProps.input_props ? rawProps.input_props : rawProps;
  } else {
    console.error(`Props file not found: ${propsPath}`);
    process.exit(1);
  }

  console.log(`[Playwright] Starting Vite dev server...`);
  const server = await createServer({
    root: __dirname,
    server: { port: 3000 },
    // Suppress logs for cleaner output
    logLevel: 'error' 
  });
  await server.listen();
  
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1080, height: 1350 },
    deviceScaleFactor: 1
  });
  
  const page = await context.newPage();
  
  console.log(`[Playwright] Loading page...`);
  await page.goto('http://localhost:3000/');
  
  // Wait for React to mount and expose window.reactReady
  await page.waitForFunction(() => window.reactReady === true);
  
  console.log(`[Playwright] Rendering template: ${templateName}`);
  await page.evaluate(({ templateName, props }) => {
    window.renderTemplate(templateName, props);
  }, { templateName, props });
  
  // Wait a moment for images to load (a more robust approach would check network idle)
  await page.waitForLoadState('networkidle', { timeout: 10000 }).catch(() => {});
  // Extra wait to ensure CSS animations (if any) are at 0s, though we mocked spring to 1
  await page.waitForTimeout(500);

  console.log(`[Playwright] Capturing screenshot to ${outputPath}`);
  await page.screenshot({ path: outputPath, type: 'jpeg', quality: 90 });
  
  await browser.close();
  await server.close();
  
  console.log(`[Playwright] Successfully rendered!`);
}

render().catch(err => {
  console.error("Rendering failed:", err);
  process.exit(1);
});

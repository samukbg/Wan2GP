const { chromium } = require('playwright');
const http = require('http');
const fs = require('fs');
const path = require('path');

function createStaticServer(rootDir) {
  const mimeTypes = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
  };

  const server = http.createServer((req, res) => {
    let reqUrl = req.url.split('?')[0];
    if (reqUrl === '/') reqUrl = '/index.html';

    let filePath = path.join(rootDir, reqUrl);

    // Try serving from dist/ first if it exists
    const distPath = path.join(rootDir, 'dist', reqUrl);
    if (fs.existsSync(distPath) && fs.statSync(distPath).isFile()) {
      filePath = distPath;
    }

    if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('Not Found');
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    const contentType = mimeTypes[ext] || 'application/octet-stream';
    res.writeHead(200, { 'Content-Type': contentType, 'Access-Control-Allow-Origin': '*' });
    fs.createReadStream(filePath).pipe(res);
  });

  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const port = server.address().port;
      resolve({ server, url: `http://127.0.0.1:${port}` });
    });
  });
}

async function render() {
  const args = process.argv.slice(2);
  const templateName = args[0];
  const outputPath = args[1];
  const propsArgIndex = args.indexOf('--props');
  
  if (!templateName || !outputPath || propsArgIndex === -1 || !args[propsArgIndex + 1]) {
    console.error("Usage: node render.js <TemplateName> <OutputPath> --props <PropsJsonPath>");
    process.exit(1);
  }
  
  if (templateName === 'scrape') {
    console.log(`[Playwright] Delegating 'scrape' command to scrape.js...`);
    const { spawnSync } = require('child_process');
    const result = spawnSync('node', [path.join(__dirname, 'scrape.js'), ...args], { stdio: 'inherit' });
    process.exit(result.status !== null ? result.status : 1);
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

  // Ensure output directory exists
  const outDir = path.dirname(path.resolve(outputPath));
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }

  let server = null;
  let localUrl = '';

  // Try Vite first if available, otherwise use built-in zero-dependency static server
  try {
    const { createServer } = require('vite');
    const viteServer = await createServer({
      root: __dirname,
      server: { port: 0, strictPort: false },
      logLevel: 'warn'
    });
    await viteServer.listen();
    server = viteServer;
    localUrl = viteServer.resolvedUrls.local[0];
    console.log(`[Playwright] Vite dev server running at ${localUrl}`);
  } catch (viteErr) {
    console.log(`[Playwright] Using built-in HTTP server (Vite not installed or skipped)`);
    const staticSrv = await createStaticServer(__dirname);
    server = staticSrv.server;
    localUrl = staticSrv.url;
    console.log(`[Playwright] Built-in static server running at ${localUrl}`);
  }
  
  const execPath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || (fs.existsSync('/usr/bin/chromium-browser') ? '/usr/bin/chromium-browser' : (fs.existsSync('/usr/bin/chromium') ? '/usr/bin/chromium' : undefined));
  const browser = await chromium.launch({ 
    headless: true,
    executablePath: execPath,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
  });
  const context = await browser.newContext({
    viewport: { width: 1080, height: 1350 },
    deviceScaleFactor: 1
  });
  
  const page = await context.newPage();
  page.on('console', msg => console.log(`[Browser Console] ${msg.type()}: ${msg.text()}`));
  page.on('pageerror', error => console.error(`[Browser Error] ${error.message}`));
  
  console.log(`[Playwright] Loading page: ${localUrl}`);
  await page.goto(localUrl, { waitUntil: 'domcontentloaded', timeout: 15000 });
  
  // Wait for React to mount and expose window.reactReady
  try {
    await page.waitForFunction(() => window.reactReady === true, { timeout: 10000 });
  } catch (waitErr) {
    console.warn('[Playwright] window.reactReady wait timed out, proceeding with render attempt...');
  }
  
  console.log(`[Playwright] Rendering template: ${templateName}`);
  await page.evaluate(({ templateName, props }) => {
    if (typeof window.renderTemplate === 'function') {
      window.renderTemplate(templateName, props);
    }
  }, { templateName, props });
  
  // Wait a moment for images to load
  await page.waitForLoadState('networkidle', { timeout: 8000 }).catch(() => {});
  await page.waitForTimeout(600);

  console.log(`[Playwright] Capturing screenshot to ${outputPath}`);
  await page.screenshot({ path: outputPath, type: 'jpeg', quality: 90 });
  
  await browser.close();
  if (server && typeof server.close === 'function') {
    server.close();
  }
  
  console.log(`[Playwright] Successfully rendered still to ${outputPath}!`);
}

render().catch(err => {
  console.error("Rendering failed:", err);
  process.exit(1);
});

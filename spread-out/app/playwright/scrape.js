const { chromium } = require('playwright');
const fs = require('fs');

async function scrape() {
  const args = process.argv.slice(2);
  const templateName = args[0]; // Ignored
  const outputPath = args[1];
  const propsArgIndex = args.indexOf('--props');
  
  if (!outputPath || propsArgIndex === -1 || !args[propsArgIndex + 1]) {
    console.error("Usage: node scrape.js <TemplateName> <OutputPath> --props <PropsJsonPath>");
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

  const targetUrl = props.url;
  if (!targetUrl) {
    fs.writeFileSync(outputPath, JSON.stringify([]));
    process.exit(0);
  }

  console.log(`[Scrape] Launching Playwright to scrape ${targetUrl}`);
  
  // Try to use a realistic user agent
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    viewport: { width: 1280, height: 800 },
    deviceScaleFactor: 1
  });
  
  const page = await context.newPage();
  
  let imageUrls = [];
  try {
    await page.goto(targetUrl, { waitUntil: 'domcontentloaded', timeout: 15000 });
    
    // Wait a little for images to load
    await page.waitForTimeout(3000);
    
    // Smooth scroll down to trigger lazy loading
    await page.evaluate(async () => {
        await new Promise((resolve) => {
            let totalHeight = 0;
            const distance = 300;
            const timer = setInterval(() => {
                const scrollHeight = document.body.scrollHeight;
                window.scrollBy(0, distance);
                totalHeight += distance;
                if (totalHeight >= scrollHeight || totalHeight > 5000) {
                    clearInterval(timer);
                    resolve(null);
                }
            }, 150);
        });
    });
    
    await page.waitForTimeout(2000);

    // Extract large images
    imageUrls = await page.evaluate(() => {
      const imgs = Array.from(document.querySelectorAll('img'));
      return imgs
        .filter(img => {
            const width = img.naturalWidth || img.width || img.clientWidth || 0;
            const height = img.naturalHeight || img.height || img.clientHeight || 0;
            // Ignore icons, logos, tracking pixels, and very small images
            const src = img.src || img.getAttribute('data-src') || '';
            const isSmall = width > 0 && width < 150;
            const isLogo = src.toLowerCase().includes('logo') || src.toLowerCase().includes('icon');
            return src.startsWith('http') && !isSmall && !isLogo;
        })
        .map(img => img.src || img.getAttribute('data-src'))
        .filter(Boolean);
    });
    
  } catch (err) {
    console.error('[Scrape] Error during scraping:', err);
  } finally {
    await browser.close();
  }
  
  // Remove duplicates and limit to 10
  const uniqueUrls = [...new Set(imageUrls)].slice(0, 10);
  
  console.log(`[Scrape] Found ${uniqueUrls.length} images. Writing to ${outputPath}`);
  fs.writeFileSync(outputPath, JSON.stringify(uniqueUrls));
  process.exit(0);
}

scrape().catch(err => {
  console.error("Scraping failed:", err);
  process.exit(1);
});

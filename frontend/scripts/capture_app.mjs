// Capture actual browser inference for the report after starting Docker Compose.
import { chromium } from 'playwright-core'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  headless: true,
})
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 })
  await page.goto('http://localhost:8080/', { waitUntil: 'domcontentloaded', timeout: 60000 })
  await page.getByText('Ready for inference').waitFor()
  await page.locator('input[type=file]').setInputFiles(path.join(root, 'data/processed/pets/images/Abyssinian_1.png'))
  await page.locator('#corruption').selectOption('noise')
  await page.locator('#severity').selectOption('high')
  await page.getByRole('button', { name: /Run restoration/ }).click()
  await page.locator('img[alt="Restored output"]').waitFor()
  await page.locator('main section').first().screenshot({
    path: path.join(root, 'report/figures/application_universal_result.png'),
  })

  await page.getByRole('button', { name: /Face-to-Sketch Generator/ }).click()
  await page.getByText('Ready for inference').waitFor()
  await page.locator('input[type=file]').setInputFiles(path.join(root, 'data/processed/fs2k/photo/photo1_image0001.png'))
  await page.locator('#style').selectOption('2')
  await page.getByRole('button', { name: /Generate sketch/ }).click()
  await page.locator('img[alt="Generated sketch"]').waitFor()
  await page.locator('main section').first().screenshot({
    path: path.join(root, 'report/figures/application_sketch_result.png'),
  })
  console.log('Captured trained Task 1 and Task 4 browser inference screenshots')
} finally {
  await browser.close()
}

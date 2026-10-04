// Record a silent, captioned walkthrough of the actual Docker Compose application.
// Run from the project root after all seven trained ONNX graphs are in models/:
//   node frontend/scripts/record_submission_demo.mjs
import { chromium } from 'playwright-core'
import { execFileSync } from 'node:child_process'
import { readFileSync, mkdirSync, readdirSync, renameSync, statSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const out = path.join(root, 'artifacts/demo')
mkdirSync(out, { recursive: true })
const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  headless: true,
})
const context = await browser.newContext({
  viewport: { width: 1440, height: 900 },
  deviceScaleFactor: 1,
  acceptDownloads: true,
  recordVideo: { dir: out, size: { width: 1440, height: 900 } },
})
const page = await context.newPage()
const pause = ms => page.waitForTimeout(ms)
const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')

async function slide(title, body, ms) {
  await page.setContent(`<html><body style="margin:0;background:#101827;color:white;font:24px Arial;padding:80px">
    <div style="font-size:16px;color:#a5b4fc;margin-bottom:20px">Generative AI Assignment 1 · Syed Ashher Majid · 23i-0007</div>
    <h1 style="font-size:48px;max-width:1100px">${escape(title)}</h1>
    <pre style="font:22px/1.6 Consolas,monospace;white-space:pre-wrap;background:#1f2937;padding:30px;border-radius:18px;max-width:1180px">${escape(body)}</pre>
    <div style="position:absolute;bottom:42px;color:#9ca3af;font-size:17px">Recorded from the running local application and saved experiment files</div>
    </body></html>`)
  await pause(ms)
}

async function caption(message, ms) {
  await page.evaluate(text => {
    document.querySelector('#demo-caption')?.remove()
    const box = document.createElement('div')
    box.id = 'demo-caption'
    box.textContent = text
    Object.assign(box.style, {
      position: 'fixed', zIndex: 9999, left: '300px', right: '26px', bottom: '18px',
      padding: '16px 20px', borderRadius: '12px', background: 'rgba(15,23,42,.92)',
      color: 'white', font: 'bold 19px Arial', boxShadow: '0 8px 30px #0005',
    })
    document.body.append(box)
  }, message)
  await pause(ms)
}

async function upload(file) {
  await page.locator('input[type=file]').setInputFiles(path.join(root, file))
}

async function runAndDownload(name) {
  await Promise.all([
    page.waitForResponse(response => response.url().includes('/api/infer/') && response.status() === 200, { timeout: 90000 }),
    page.getByRole('button', { name: /Run restoration|Generate sketch/ }).click(),
  ])
  await page.locator('img[alt="Restored output"],img[alt="Generated sketch"]').first().waitFor({ timeout: 90000 })
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: /Download PNG/ }).click()
  await (await download).saveAs(path.join(out, `${name}.png`))
}

try {
  const health = await (await fetch('http://localhost:8080/api/health')).json()
  if (!Object.values(health.workspaces).every(Boolean)) throw new Error(`Not all workspaces are ready: ${JSON.stringify(health.workspaces)}`)
  const compose = execFileSync('docker', ['compose', 'ps'], { cwd: root, encoding: 'utf8' })
  await slide('One-command Docker Compose startup', `docker compose up -d --build\n\n${compose}`, 36000)

  await page.goto('http://localhost:8080/', { waitUntil: 'domcontentloaded' })
  await page.getByText('Ready for inference').waitFor()
  await caption('Task 1: one universal autoencoder serves clean, noise, blur, and occlusion inputs.', 18000)
  await upload('data/processed/pets/images/Abyssinian_1.png')
  await page.locator('#corruption').selectOption('noise')
  await page.locator('#severity').selectOption('high')
  await caption('The corruption is generated at inference time. Here: high-severity salt-and-pepper noise.', 10000)
  await runAndDownload('task1_restoration')
  await caption('The processed input, restored output, inference time, and downloaded PNG are visible.', 30000)

  await page.getByRole('button', { name: /Hard-Routed Restoration/ }).click()
  await upload('data/processed/pets/images/Abyssinian_1.png')
  await page.locator('#corruption').selectOption('blur')
  await page.locator('#severity').selectOption('high')
  await caption('Task 2: the classifier chooses one trained specialist, or an identity bypass for clean images.', 14000)
  await runAndDownload('task2_hard_route')
  await page.locator('text=Classifier probabilities').first().scrollIntoViewIfNeeded()
  await caption('The chart reports four class probabilities and the selected restoration branch.', 30000)

  await page.getByRole('button', { name: /Soft Mixture of Experts/ }).click()
  await upload('data/processed/pets/images/Abyssinian_1.png')
  await page.locator('#corruption').selectOption('occlusion')
  await page.locator('#severity').selectOption('medium')
  await caption('Task 3: the soft gate blends identity, noise, blur, and occlusion branches.', 14000)
  await runAndDownload('task3_soft_mixture')
  await page.locator('text=Soft mixture contributions').first().scrollIntoViewIfNeeded()
  await caption('These four displayed weights are the model output for this uploaded image.', 31000)

  await page.getByRole('button', { name: /Face-to-Sketch Generator/ }).click()
  await upload('data/processed/fs2k/photo/photo1_image0001.png')
  await caption('Task 4: the conditional generator renders one uploaded photograph in three FS2K styles.', 13000)
  for (const style of [1, 2, 3]) {
    await page.locator('#style').selectOption(String(style))
    await runAndDownload(`task4_style_${style}`)
    await caption(`Style ${style}: generated result and downloadable PNG.`, 12000)
  }
  await page.locator('text=Model files').first().scrollIntoViewIfNeeded()
  await caption('Each workspace also exposes its trained ONNX inference graph for download.', 16000)

  const history = [
    ['Task 1 universal', 'artifacts/task1/history.jsonl'],
    ['Task 2 classifier', 'artifacts/task2/classifier/history.jsonl'],
    ['Task 2 salt expert', 'artifacts/task2/expert_1/history.jsonl'],
    ['Task 2 blur expert', 'artifacts/task2/expert_2/history.jsonl'],
    ['Task 2 occlusion expert', 'artifacts/task2/expert_3/history.jsonl'],
    ['Task 3 soft mixture', 'artifacts/task3_submission/history.jsonl'],
    ['Task 4 GAN', 'artifacts/task4/history.jsonl'],
  ].map(([label, filename]) => {
    const lines = readFileSync(path.join(root, filename), 'utf8').trim().split(/\r?\n/)
    const last = JSON.parse(lines.at(-1).replaceAll('Infinity', 'null'))
    return `${label}: ${lines.length} logged epochs; latest epoch ${last.epoch}; validation score ${Number(last.score).toFixed(5)}`
  }).join('\n')
  await slide('Saved experiment tracking records', `MLflow database: mlflow.db\nStudy files: artifacts/studies/\nPer-epoch JSONL history:\n\n${history}`, 48000)
  await slide('Source and reproducibility', 'https://github.com/SyedAshherMajid/i230007-genai-assignment1\n\nREADME: data preparation, training, tuning, evaluation, ONNX export, Docker Compose\n\nThe report includes methods, metrics, figures, and limitations.', 26000)
} finally {
  await page.close()
  await context.close()
  await browser.close()
  const files = readdirSync(out).filter(name => name.endsWith('.webm'))
  const newest = files.map(name => ({ name, time: statSync(path.join(out, name)).mtimeMs })).sort((a, b) => b.time - a.time)[0]
  if (newest) renameSync(path.join(out, newest.name), path.join(out, 'GenAI_Assignment_1_Demo_complete.webm'))
}

import { useEffect, useRef, useState } from 'react'

const workspaces = [
  { id: 'universal', number: '01', label: 'Universal Restoration', subtitle: 'One model for every condition' },
  { id: 'hard', number: '02', label: 'Hard-Routed Restoration', subtitle: 'Classifier and specialists' },
  { id: 'soft', number: '03', label: 'Soft Mixture of Experts', subtitle: 'Weighted expert contributions' },
  { id: 'sketch', number: '04', label: 'Face-to-Sketch Generator', subtitle: 'Three artistic styles' },
]
const corruptionOptions = [
  { value: 'none', label: 'Use upload as-is' },
  { value: 'clean', label: 'Clean identity' },
  { value: 'noise', label: 'Salt-and-pepper noise' },
  { value: 'blur', label: 'Gaussian blur' },
  { value: 'occlusion', label: 'Black rectangle occlusion' },
]
const routeLabels = ['Clean', 'Noise', 'Blur', 'Occlusion']
const weightLabels = ['Identity', 'Noise expert', 'Blur expert', 'Occlusion expert']

function MeasureBars({ values, labels, title }) {
  return <section className="card p-5" aria-label={title}>
    <p className="tag mb-4">{title}</p>
    <div className="space-y-4">{labels.map((label, i) => <div key={label}>
      <div className="mb-1.5 flex items-center justify-between text-sm"><span className="font-medium">{label}</span><span className="font-bold tabular-nums">{(values[i] * 100).toFixed(1)}%</span></div>
      <div className="h-2.5 overflow-hidden rounded-full bg-slate-100" role="progressbar" aria-label={label} aria-valuenow={Math.round(values[i] * 100)} aria-valuemin="0" aria-valuemax="100"><div className="h-full rounded-full bg-indigo-600 transition-all" style={{ width: `${values[i] * 100}%` }} /></div>
    </div>)}</div>
  </section>
}

function ImageCard({ label, src, emptyText }) {
  return <div className="card overflow-hidden"><div className="border-b border-slate-100 px-5 py-3"><p className="tag">{label}</p></div>
    <div className="flex aspect-square items-center justify-center bg-slate-50 p-4">
      {src ? <img className="max-h-full max-w-full rounded-lg object-contain shadow-sm" src={src} alt={label} /> : <p className="max-w-52 text-center text-sm text-slate-400">{emptyText}</p>}
    </div>
  </div>
}

export default function App() {
  const [task, setTask] = useState('universal')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [corruption, setCorruption] = useState('none')
  const [severity, setSeverity] = useState('medium')
  const [style, setStyle] = useState(1)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [health, setHealth] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [cameraOpen, setCameraOpen] = useState(false)
  const fileInput = useRef(null)
  const video = useRef(null)
  const stream = useRef(null)
  const active = workspaces.find(item => item.id === task)
  const taskReady = health?.workspaces?.[task]

  useEffect(() => {
    let mounted = true
    const refresh = () => fetch('/api/health').then(response => response.json())
      .then(data => { if (mounted) setHealth(data) })
      .catch(() => { if (mounted) setHealth({ status: 'offline' }) })
    refresh()
    const timer = setInterval(refresh, 20000)
    return () => { mounted = false; clearInterval(timer) }
  }, [])
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])
  useEffect(() => () => { stream.current?.getTracks().forEach(track => track.stop()) }, [])

  function chooseFile(candidate) {
    if (!candidate) return
    if (!candidate.type.startsWith('image/')) { setError('Please choose an image file.'); return }
    if (candidate.size > 10 * 1024 * 1024) { setError('Choose an image under 10 MB.'); return }
    setFile(candidate); setPreview(URL.createObjectURL(candidate)); setResult(null); setError('')
  }

  function changeTask(id) {
    setTask(id); setResult(null); setError(''); closeCamera()
  }

  async function openCamera() {
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires a supported browser on localhost or HTTPS.')
      const capture = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      stream.current = capture; setCameraOpen(true); setError('')
      setTimeout(() => { if (video.current) video.current.srcObject = capture }, 0)
    } catch (reason) { setError(reason?.message || 'Camera unavailable. You can upload a photo instead.') }
  }
  function closeCamera() {
    stream.current?.getTracks().forEach(track => track.stop())
    stream.current = null; setCameraOpen(false)
  }
  function captureFrame() {
    if (!video.current?.videoWidth) return
    const canvas = document.createElement('canvas')
    canvas.width = video.current.videoWidth; canvas.height = video.current.videoHeight
    canvas.getContext('2d').drawImage(video.current, 0, 0)
    canvas.toBlob(blob => { if (blob) chooseFile(new File([blob], 'webcam-capture.png', { type: 'image/png' })) }, 'image/png')
    closeCamera()
  }

  async function run() {
    if (!file) { setError('Upload or capture an image first.'); return }
    if (taskReady === false) { setError('This workspace model is still being prepared.'); return }
    setBusy(true); setError(''); setResult(null)
    try {
      const form = new FormData(); form.append('file', file)
      const params = new URLSearchParams(task === 'sketch' ? { style } : { corruption, severity, seed: '42' })
      const response = await fetch(`/api/infer/${task}?${params}`, { method: 'POST', body: form })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || `Inference failed (${response.status})`)
      setResult(data)
    } catch (reason) { setError(reason.message || 'The backend could not process this image.') }
    finally { setBusy(false) }
  }

  function download() {
    if (!result?.output) return
    const link = document.createElement('a')
    link.href = result.output; link.download = `genai-a1-${task}${task === 'sketch' ? `-style-${style}` : ''}.png`; link.click()
  }

  return <div className="min-h-screen lg:flex">
    <aside className="border-b border-slate-200 bg-white lg:min-h-screen lg:w-72 lg:shrink-0 lg:border-b-0 lg:border-r">
      <div className="border-b border-slate-100 px-6 py-6"><div className="flex items-center gap-3"><div className="flex size-11 items-center justify-center rounded-xl bg-indigo-600 text-xl font-extrabold text-white">R</div><div><h1 className="text-base font-extrabold tracking-tight">Restore & Sketch Lab</h1><p className="text-xs text-slate-500">Generative vision workspace</p></div></div></div>
      <nav className="p-4" aria-label="Workspaces"><p className="tag px-3 pb-3 pt-2">Workspaces</p><div className="grid gap-1 sm:grid-cols-2 lg:grid-cols-1">{workspaces.map(item => <button key={item.id} type="button" onClick={() => changeTask(item.id)} className={`flex w-full items-start gap-3 rounded-xl px-3 py-3 text-left transition ${task === item.id ? 'bg-indigo-50 text-indigo-800' : 'hover:bg-slate-50'}`} aria-current={task === item.id ? 'page' : undefined}><span className={`mt-0.5 text-xs font-bold ${task === item.id ? 'text-indigo-600' : 'text-slate-400'}`}>{item.number}</span><span><span className="block text-sm font-semibold">{item.label}</span><span className="mt-0.5 block text-xs text-slate-500">{item.subtitle}</span></span></button>)}</div></nav>
      <div className="mx-4 mb-5 rounded-xl border border-slate-200 bg-slate-50 p-4"><p className="tag mb-2">Workspace status</p><div className="flex items-center gap-2 text-sm"><span className={`size-2 rounded-full ${taskReady ? 'bg-teal-500' : health?.status === 'offline' ? 'bg-rose-500' : 'bg-amber-500'}`} /><span>{taskReady ? 'Ready for inference' : health?.status === 'offline' ? 'Backend unavailable' : taskReady === false ? 'Model being prepared' : 'Checking models…'}</span></div></div>
    </aside>

    <main className="min-w-0 flex-1 px-4 py-6 sm:px-8 lg:px-10 lg:py-9"><div className="mx-auto max-w-6xl">
      <div className="mb-7 flex flex-wrap items-start justify-between gap-4"><div><p className="tag mb-2 text-indigo-600">Workspace {active.number} / 04</p><h2 className="text-2xl font-extrabold tracking-tight sm:text-3xl">{active.label}</h2><p className="mt-2 text-sm text-slate-500">{task === 'sketch' ? 'Turn a portrait into one of three paired sketch styles.' : 'Inspect the processed input and restore it at 128 × 128 pixels.'}</p></div><span className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600">128 × 128 model input</span></div>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_310px]">
        <div className="card p-5 sm:p-6"><div className="mb-4 flex items-center justify-between"><p className="tag">01 / Input image</p><span className="text-xs text-slate-400">PNG, JPEG, WebP · max 10 MB</span></div>
          <input ref={fileInput} type="file" accept="image/*" className="hidden" onChange={event => chooseFile(event.target.files?.[0])} />
          <div className={`flex min-h-64 flex-col items-center justify-center rounded-xl border-2 border-dashed p-5 text-center transition ${dragging ? 'border-indigo-500 bg-indigo-50' : 'border-slate-200 bg-slate-50'}`} onDragOver={event => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); chooseFile(event.dataTransfer.files?.[0]) }}>
            {preview ? <img src={preview} alt="Selected upload" className="max-h-52 max-w-full rounded-lg object-contain shadow-sm" /> : <><div className="mb-3 flex size-14 items-center justify-center rounded-2xl bg-indigo-100 text-2xl text-indigo-700">↑</div><p className="font-semibold">Drop an image here</p><p className="mt-1 text-sm text-slate-500">or choose one from your device</p></>}
            <button type="button" onClick={() => fileInput.current?.click()} className="mt-4 rounded-lg border border-indigo-200 bg-white px-4 py-2 text-sm font-semibold text-indigo-700 hover:bg-indigo-50">{preview ? 'Change image' : 'Browse images'}</button>
          </div>
          {task === 'sketch' && <div className="mt-4"><button type="button" onClick={cameraOpen ? closeCamera : openCamera} className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold hover:bg-slate-50">{cameraOpen ? 'Close camera' : 'Use webcam'}</button>{cameraOpen && <div className="mt-4 max-w-xl"><video ref={video} autoPlay playsInline muted className="w-full rounded-xl bg-slate-900" /><button type="button" onClick={captureFrame} className="mt-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white">Capture photo</button></div>}</div>}
        </div>

        <div className="card flex flex-col p-5 sm:p-6"><p className="tag mb-5">02 / {task === 'sketch' ? 'Style controls' : 'Input controls'}</p>
          {task === 'sketch' ? <><label className="mb-2 text-sm font-semibold" htmlFor="style">Sketch style</label><select id="style" className="field" value={style} onChange={event => setStyle(Number(event.target.value))}><option value={1}>Style 1</option><option value={2}>Style 2</option><option value={3}>Style 3</option></select><p className="mt-3 text-xs leading-relaxed text-slate-500">Styles correspond to the three annotation labels in FS2K. The same photograph can be rendered in each style.</p></> : <><label className="mb-2 text-sm font-semibold" htmlFor="corruption">Input treatment</label><select id="corruption" className="field" value={corruption} onChange={event => setCorruption(event.target.value)}>{corruptionOptions.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}</select>{corruption !== 'none' && corruption !== 'clean' && <><label className="mb-2 mt-5 text-sm font-semibold" htmlFor="severity">Severity</label><select id="severity" className="field" value={severity} onChange={event => setSeverity(event.target.value)}><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></>}<p className="mt-4 text-xs leading-relaxed text-slate-500">Use an already damaged image as-is, or choose a synthetic corruption to apply before inference.</p></>}
          <div className="mt-auto pt-7"><button type="button" onClick={run} disabled={busy || !file || taskReady === false} className="w-full rounded-xl bg-indigo-600 px-5 py-3.5 text-sm font-bold text-white shadow-sm hover:bg-indigo-700">{busy ? 'Processing…' : task === 'sketch' ? 'Generate sketch' : 'Run restoration'} <span aria-hidden="true">→</span></button></div>
        </div>
      </div>

      {error && <div role="alert" className="mt-5 rounded-xl border border-rose-200 bg-rose-50 px-5 py-4 text-sm font-medium text-rose-800">{error}</div>}
      <section className="mt-7"><div className="mb-4 flex flex-wrap items-center justify-between gap-3"><div><p className="tag mb-1">03 / Results</p><h3 className="text-lg font-bold">{task === 'sketch' ? 'Photo and generated sketch' : 'Before and after'}</h3></div>{result && <div className="flex items-center gap-3"><span className="text-xs text-slate-500">{result.inference_ms} ms</span><button type="button" onClick={download} className="rounded-lg border border-indigo-200 bg-white px-4 py-2 text-sm font-semibold text-indigo-700 hover:bg-indigo-50">Download PNG ↓</button></div>}</div>
        <div className="grid gap-5 sm:grid-cols-2"><ImageCard label={task === 'sketch' ? 'Original photo' : 'Processed input'} src={result?.input} emptyText="Your processed input will appear here." /><ImageCard label={task === 'sketch' ? 'Generated sketch' : 'Restored output'} src={result?.output} emptyText="The model output will appear after processing." /></div>
        {result?.probabilities && <div className="mt-5 grid gap-5 md:grid-cols-2"><MeasureBars values={result.probabilities} labels={routeLabels} title="Classifier probabilities" /><div className="card p-5"><p className="tag mb-4">Routing decision</p><p className="text-xl font-bold capitalize">{result.predicted_label}</p><p className="mt-2 text-sm text-slate-500">Selected branch: <strong>{result.selected_branch}</strong></p></div></div>}
        {result?.weights && <div className="mt-5"><MeasureBars values={result.weights} labels={weightLabels} title="Soft mixture contributions" /></div>}
      </section>
      <p className="mt-10 border-t border-slate-200 pt-5 text-xs text-slate-400">Generative AI Assignment 1 · Syed Ashher Majid · FAST NUCES Islamabad</p>
    </div></main>
  </div>
}

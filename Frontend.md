# HypeGuard — FRONTEND.md
> Owner: Frontend Teammate
> Stack: Next.js 14 + Tailwind CSS + Shadcn UI
> Status: Scope Frozen — Do not add anything not listed here

---

## YOUR SINGLE JOB

Consume the `/analyze` API response and display it beautifully.
You do NOT touch Python, ML, or data logic. You only build UI.

---

## 0. THE API CONTRACT (Read-Only — Backend Owns This)

Your entire UI is built around this one JSON shape.
**Never assume a field exists — always use optional chaining (`?.`).**

```typescript
// types/hypeguard.ts  ← create this file first
export type HypeLabel = "ORGANIC" | "HYPE" | "INSTITUTIONAL" | "NEUTRAL" | "PUMP_ALERT"
export type ActionLabel = "BUY" | "WAIT" | "AVOID"
export type Currency = "INR" | "USD"

export interface AnalyzeRequest {
  ticker: string         // e.g. "GME"
  amount: number         // e.g. 5000
  currency: Currency     // "INR" or "USD"
}

export interface TopHeadline {
  title:      string     // headline text
  source:     string     // e.g. "Reuters"
  hype_score: number     // 0–100, per-headline score
}

export interface AnalyzeResponse {
  ticker:          string
  snapshot_time:   string       // ISO datetime string
  hype_score:      number       // 0–100  ← drives the gauge
  label:           HypeLabel    // drives badge color
  anomaly_score:   number       // 0–1
  sentiment_score: number       // 0–1

  reasoning: string[]           // ["Volume is 4.2x...", "RSI at 81..."]

  volume_data: {
    rvol:              number   // e.g. 4.2
    volume_zscore:     number   // e.g. 3.1
    latest_volume:     number   // raw int
    avg_20d_volume:    number   // raw int
    is_volume_anomaly: boolean
  }

  price_data: {
    current_price:  number      // e.g. 147.32
    rsi_14:         number      // 0–100
    price_vs_sma20: number      // e.g. +12.4 (percent)
    is_overbought:  boolean
  }

  news_data: {
    total_headlines:        number
    extreme_language_ratio: number   // 0–1
    source_diversity:       number   // 0–1
    headline_similarity:    number   // 0–1
    top_headlines:          TopHeadline[]  // max 5 items
  }

  investment_advice: {
    action:           ActionLabel
    deploy_now_pct:   number     // e.g. 20
    deploy_now_inr:   number     // e.g. 1000
    deploy_now_usd:   number     // e.g. 12
    wait_days:        number     // e.g. 4
    reason:           string
  }
}

export interface ErrorResponse {
  error:   string
  detail:  string
}
```

---

## 1. SETUP (Run Once)

```bash
# 1. Create Next.js project
npx create-next-app@latest hypeguard-ui --typescript --tailwind --eslint --app --src-dir
cd hypeguard-ui

# 2. Init Shadcn
npx shadcn-ui@latest init
# When prompted: Style = Default, Base color = Zinc, CSS variables = Yes

# 3. Install ONLY these Shadcn components (nothing else)
npx shadcn-ui@latest add card
npx shadcn-ui@latest add button
npx shadcn-ui@latest add input
npx shadcn-ui@latest add badge
npx shadcn-ui@latest add alert
npx shadcn-ui@latest add progress
npx shadcn-ui@latest add table
npx shadcn-ui@latest add separator
npx shadcn-ui@latest add skeleton

# 4. Install one additional package for the gauge
npm install react-gauge-component

# 5. Create env file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
```

---

## 2. FOLDER STRUCTURE (Create Exactly This)

```
src/
├── app/
│   ├── page.tsx              ← Main page (assembles all components)
│   ├── layout.tsx            ← Root layout (dark theme)
│   └── globals.css           ← Tailwind base styles
├── components/
│   ├── SearchBar.tsx         ← TASK 1
│   ├── HypeMeter.tsx         ← TASK 2
│   ├── SignalGrid.tsx         ← TASK 3
│   ├── ReasoningBox.tsx      ← TASK 4
│   ├── NewsFeed.tsx          ← TASK 5
│   ├── InvestmentAdvice.tsx  ← TASK 6
│   └── DemoButtons.tsx       ← TASK 7
├── types/
│   └── hypeguard.ts          ← API types (copy from Section 0)
├── lib/
│   └── api.ts                ← API fetch functions (Section 3)
└── hooks/
    └── useAnalyze.ts         ← Data fetching hook (Section 3)
```

---

## 3. API LAYER (lib/api.ts + hooks/useAnalyze.ts)

### lib/api.ts
```typescript
import { AnalyzeRequest, AnalyzeResponse } from "@/types/hypeguard"

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

export async function analyzeStock(req: AnalyzeRequest): Promise<AnalyzeResponse> {
  const res = await fetch(`${BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail ?? "Analysis failed")
  }
  return res.json()
}

export async function fetchDemo(ticker: string): Promise<AnalyzeResponse> {
  const res = await fetch(`${BASE}/demo/${ticker.toUpperCase()}`)
  if (!res.ok) throw new Error("Demo data unavailable")
  return res.json()
}
```

### hooks/useAnalyze.ts
```typescript
"use client"
import { useState } from "react"
import { AnalyzeResponse, AnalyzeRequest } from "@/types/hypeguard"
import { analyzeStock, fetchDemo } from "@/lib/api"

export function useAnalyze() {
  const [data, setData]       = useState<AnalyzeResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)

  async function analyze(req: AnalyzeRequest) {
    setLoading(true)
    setError(null)
    try {
      const result = await analyzeStock(req)
      setData(result)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function loadDemo(ticker: string) {
    setLoading(true)
    setError(null)
    try {
      const result = await fetchDemo(ticker)
      setData(result)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return { data, loading, error, analyze, loadDemo }
}
```

---

## 4. COLOR & LABEL SYSTEM (Use Everywhere — No Exceptions)

```typescript
// lib/theme.ts  ← create this helper file

import { HypeLabel, ActionLabel } from "@/types/hypeguard"

export const LABEL_CONFIG: Record<HypeLabel, {
  color: string        // Tailwind text color
  bg: string           // Tailwind bg color
  border: string       // Tailwind border
  text: string         // Display text
  icon: string         // Emoji
}> = {
  ORGANIC:      { color: "text-green-400",  bg: "bg-green-950",  border: "border-green-700", text: "ORGANIC MOVE",  icon: "✅" },
  NEUTRAL:      { color: "text-gray-400",   bg: "bg-gray-900",   border: "border-gray-700",  text: "NEUTRAL",       icon: "➖" },
  HYPE:         { color: "text-orange-400", bg: "bg-orange-950", border: "border-orange-700",text: "INFLATED",      icon: "🔶" },
  INSTITUTIONAL:{ color: "text-blue-400",   bg: "bg-blue-950",   border: "border-blue-700",  text: "INSTITUTIONAL", icon: "🏦" },
  PUMP_ALERT:   { color: "text-red-400",    bg: "bg-red-950",    border: "border-red-700",   text: "PUMP ALERT",    icon: "🚨" },
}

export const ACTION_CONFIG: Record<ActionLabel, {
  color: string
  bg: string
  text: string
  icon: string
}> = {
  BUY:  { color: "text-green-400",  bg: "bg-green-950",  text: "SAFE TO INVEST", icon: "✅" },
  WAIT: { color: "text-yellow-400", bg: "bg-yellow-950", text: "WAIT",           icon: "⏳" },
  AVOID:{ color: "text-red-400",    bg: "bg-red-950",    text: "AVOID",          icon: "🚫" },
}

export function getHypeColor(score: number): string {
  if (score < 30)  return "#22c55e"   // green
  if (score < 60)  return "#eab308"   // yellow
  if (score < 85)  return "#f97316"   // orange
  return "#ef4444"                    // red
}
```

---

## 5. TASKS (Build In This Order)

---

### TASK 1 — SearchBar.tsx
**Input:** User types ticker + amount + picks currency
**Output:** Calls `analyze()` or `loadDemo()` from `useAnalyze`

**Props:**
```typescript
interface SearchBarProps {
  onAnalyze: (ticker: string, amount: number, currency: "INR" | "USD") => void
  onDemo:    (ticker: string) => void
  loading:   boolean
}
```

**Elements to build:**
- Text input for ticker (uppercase, max 5 chars, placeholder "GME")
- Number input for investment amount (placeholder "5000")
- Toggle button: `INR ↔ USD` — toggles currency state locally
- Primary button: "Analyze" → calls `onAnalyze`
- Below: three demo buttons side by side

**Demo buttons (hardcoded):**
```typescript
const DEMO_TICKERS = [
  { ticker: "GME",  label: "🎮 GME — Classic Pump" },
  { ticker: "NVDA", label: "🤖 NVDA — Organic Growth" },
  { ticker: "AMC",  label: "🎬 AMC — Meme Hype" },
]
```

**Shadcn used:** `Input`, `Button`
**Outcome:** User can trigger analysis or load demo instantly.

---

### TASK 2 — HypeMeter.tsx
**Input:** `hype_score` (0–100), `label` (HypeLabel), `loading` (boolean)
**Output:** Big animated gauge — the hero of the entire page

**Props:**
```typescript
interface HypeMeterProps {
  hype_score: number
  label:      HypeLabel
  loading:    boolean
}
```

**Elements to build:**
- `react-gauge-component` with these exact settings:
```typescript
<GaugeComponent
  value={hype_score}
  type="radial"
  arc={{
    colorArray: ["#22c55e", "#eab308", "#f97316", "#ef4444"],
    padding: 0.02,
    subArcs: [
      { limit: 30 },
      { limit: 60 },
      { limit: 85 },
      { limit: 100 },
    ]
  }}
  pointer={{ elastic: true, animationDelay: 0 }}
  labels={{
    valueLabel: { formatTextValue: (v) => `${v.toFixed(1)}%` }
  }}
/>
```
- Below gauge: large label badge using `LABEL_CONFIG[label]`
- Loading state: use `Skeleton` from Shadcn (circular placeholder)

**Shadcn used:** `Badge`, `Skeleton`
**Outcome:** Needle animates to score on load. Instantly communicates severity.

---

### TASK 3 — SignalGrid.tsx
**Input:** `volume_data`, `price_data`, `news_data` from `AnalyzeResponse`
**Output:** Three metric cards in a row

**Props:**
```typescript
interface SignalGridProps {
  volume_data: AnalyzeResponse["volume_data"]
  price_data:  AnalyzeResponse["price_data"]
  news_data:   AnalyzeResponse["news_data"]
}
```

**Three cards — build each exactly:**

**Card 1 — Volume**
```
📊 VOLUME
─────────────────
RVOL:       4.2x        ← red if > 2.5, green if <= 1.5
Z-Score:    3.1         ← red if > 2.0
20D Avg:    1,200,000
Today:      5,100,000
Status Badge: [🔴 ANOMALY] or [✅ NORMAL]
```

**Card 2 — Price**
```
📈 PRICE
─────────────────
Current:    $147.32
vs SMA20:   +12.4%      ← green if positive, red if negative
RSI (14):   81          ← red if > 75
BB Width:   8.2%
Status Badge: [🔥 OVERBOUGHT] or [✅ NORMAL]
```

**Card 3 — Sentiment**
```
📰 NEWS
─────────────────
Headlines:  30
Hype Lang:  42%         ← red bar if > 30%
Source Mix: 65%         ← green bar if > 50%
Similarity: 38%         ← red bar if > 50%
Status Badge: [⚠️ SUSPICIOUS] or [✅ DIVERSE]
```

For the mini progress bars inside cards, use Shadcn `Progress`.

**Shadcn used:** `Card`, `CardHeader`, `CardContent`, `Badge`, `Progress`
**Outcome:** Panel/jury sees the three signals at a glance.

---

### TASK 4 — ReasoningBox.tsx
**Input:** `reasoning: string[]` from `AnalyzeResponse`
**Output:** Bulleted list of why the stock was flagged

**Props:**
```typescript
interface ReasoningBoxProps {
  reasoning: string[]
  label:     HypeLabel
}
```

**Layout:**
```
┌─────────────────────────────────────────┐
│  WHY WAS THIS FLAGGED?                  │
│  ─────────────────────────────          │
│  • Volume is 4.2x the 20-day average   │
│  • 34 near-identical headlines / 6h    │
│  • No earnings catalyst found           │
│  • RSI at 81 — overbought territory    │
└─────────────────────────────────────────┘
```

- Use the border color from `LABEL_CONFIG[label].border`
- Each bullet is a `<li>` with the label's icon prefix
- If `reasoning` is empty, show: *"Insufficient data to generate explanation."*

**Shadcn used:** `Card`, `CardHeader`, `CardContent`
**Outcome:** Explains the model's decision in plain English.

---

### TASK 5 — NewsFeed.tsx
**Input:** `news_data.top_headlines` (max 5 items)
**Output:** Table of headlines with per-headline hype score

**Props:**
```typescript
interface NewsFeedProps {
  top_headlines: TopHeadline[]
}
```

**Table columns:**
| # | Headline (truncate at 60 chars) | Source | Hype Score |
|---|---|---|---|

**Hype score cell:**
- 0–30 → green badge
- 31–65 → yellow badge
- 66–100 → red badge

If `top_headlines` is empty, show: *"No recent headlines available."*

**Shadcn used:** `Table`, `TableHeader`, `TableRow`, `TableCell`, `Badge`
**Outcome:** User sees which specific headlines are driving the hype score.

---

### TASK 6 — InvestmentAdvice.tsx
**Input:** `investment_advice`, `currency` (from user's search), `hype_score`
**Output:** The final "money" panel — what to actually do

**Props:**
```typescript
interface InvestmentAdviceProps {
  advice:     AnalyzeResponse["investment_advice"]
  currency:   "INR" | "USD"
  hype_score: number
}
```

**Layout:**
```
┌─────────────────────────────────────────────────┐
│  💰 INVESTMENT ADVICE          [⏳ WAIT]         │
│  ─────────────────────────────────────────────  │
│  Your Input:     ₹5,000 / $60                   │
│                                                 │
│  Deploy NOW:  ████░░░░░░  ₹1,000  (20%)         │
│  Hold:                    ₹4,000  (wait 4 days) │
│                                                 │
│  Risk Level:  🔴 HIGH                           │
│  Reason: "High manipulation risk..."            │
└─────────────────────────────────────────────────┘
```

**Rules:**
- Show `deploy_now_inr` if `currency === "INR"`, else `deploy_now_usd`
- The deploy bar uses `Progress` component: value = `deploy_now_pct`
- Risk Level derived from `hype_score`:
  - 0–30 → 🟢 LOW
  - 31–60 → 🟡 MEDIUM
  - 61–85 → 🟠 HIGH
  - 86–100 → 🔴 EXTREME

**Shadcn used:** `Card`, `Progress`, `Badge`, `Alert`
**Outcome:** The clearest deliverable — tells user exactly what to do with their money.

---

### TASK 7 — DemoButtons.tsx
**Input:** `onDemo` callback, `loading` state
**Output:** Three pre-set demo buttons + a note about demo mode

**Props:**
```typescript
interface DemoButtonsProps {
  onDemo:  (ticker: string) => void
  loading: boolean
}
```

```typescript
const DEMOS = [
  { ticker: "GME",  label: "🎮 GME",  subtitle: "Classic Pump" },
  { ticker: "NVDA", label: "🤖 NVDA", subtitle: "Organic Growth" },
  { ticker: "AMC",  label: "🎬 AMC",  subtitle: "Meme Hype" },
]
```

- Each is a `Button` variant="outline" with ticker label + subtitle below
- Small note below: *"Demo mode uses pre-cached data for instant results"*

**Shadcn used:** `Button`
**Outcome:** Live demo safety net — one click, instant result, no API wait.

---

### TASK 8 — page.tsx (Final Assembly)
**Assemble all components into the main page.**

```typescript
"use client"
import { useState } from "react"
import { useAnalyze } from "@/hooks/useAnalyze"
import SearchBar        from "@/components/SearchBar"
import DemoButtons      from "@/components/DemoButtons"
import HypeMeter        from "@/components/HypeMeter"
import SignalGrid        from "@/components/SignalGrid"
import ReasoningBox     from "@/components/ReasoningBox"
import NewsFeed         from "@/components/NewsFeed"
import InvestmentAdvice from "@/components/InvestmentAdvice"

export default function Home() {
  const { data, loading, error, analyze, loadDemo } = useAnalyze()
  const [currency, setCurrency] = useState<"INR" | "USD">("INR")

  return (
    <main className="min-h-screen bg-black text-white p-6 max-w-5xl mx-auto">
      {/* Header */}
      <h1 className="text-3xl font-bold mb-1">🛡️ HypeGuard</h1>
      <p className="text-gray-400 mb-6 text-sm">
        Detect artificial stock volatility before it costs you money.
      </p>

      {/* Search */}
      <SearchBar
        onAnalyze={(t, a, c) => { setCurrency(c); analyze({ ticker: t, amount: a, currency: c }) }}
        onDemo={loadDemo}
        loading={loading}
      />

      {/* Demo buttons */}
      <DemoButtons onDemo={loadDemo} loading={loading} />

      {/* Error */}
      {error && (
        <div className="mt-4 p-3 border border-red-700 bg-red-950 rounded text-red-400 text-sm">
          ⚠️ {error}
        </div>
      )}

      {/* Results — only show after data or during loading */}
      {(data || loading) && (
        <div className="mt-8 space-y-6">
          <HypeMeter
            hype_score={data?.hype_score ?? 0}
            label={data?.label ?? "NEUTRAL"}
            loading={loading}
          />
          {data && <>
            <SignalGrid
              volume_data={data.volume_data}
              price_data={data.price_data}
              news_data={data.news_data}
            />
            <ReasoningBox reasoning={data.reasoning} label={data.label} />
            <NewsFeed top_headlines={data.news_data.top_headlines} />
            <InvestmentAdvice
              advice={data.investment_advice}
              currency={currency}
              hype_score={data.hype_score}
            />
          </>}
        </div>
      )}
    </main>
  )
}
```

---

## 6. GLOBAL STYLES (app/layout.tsx)

```typescript
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "HypeGuard — Stock Hype Detector",
  description: "Detect artificial stock volatility with ML",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-black`}>{children}</body>
    </html>
  )
}
```

---

## 7. WHAT YOU DO NOT BUILD

❌ No charts or graphs (backend notebook handles those)
❌ No authentication or user accounts
❌ No real-time updates (no websockets, no polling)
❌ No settings page
❌ No mobile-specific layout (responsive is fine, mobile-first is not required)
❌ No extra Shadcn components beyond the list in Section 1
❌ No direct calls to yfinance, Python, or any external data API
❌ No Redux or Zustand — `useState` + `useAnalyze` hook only

---

## 8. HANDOFF CHECKLIST

Before saying "done", verify:
- [ ] `npm run build` completes with zero errors
- [ ] Demo buttons return results without errors (use mock JSON if backend not ready)
- [ ] All 7 components render without undefined errors when `data` is null
- [ ] HypeMeter gauge animates on data load
- [ ] INR/USD toggle correctly switches the InvestmentAdvice display
- [ ] Dark theme is consistent across all components

---

## 9. IF BACKEND IS NOT READY YET

Use this mock response to develop UI independently:

```typescript
// lib/mock.ts
export const MOCK_RESPONSE = {
  ticker: "GME",
  snapshot_time: new Date().toISOString(),
  hype_score: 87.3,
  label: "PUMP_ALERT",
  anomaly_score: 0.81,
  sentiment_score: 0.74,
  reasoning: [
    "Volume is 4.2x the 20-day average",
    "34 near-identical headlines detected in 6 hours",
    "No earnings catalyst found in last 5 days",
    "RSI at 81 — overbought territory"
  ],
  volume_data: { rvol: 4.2, volume_zscore: 3.1, latest_volume: 5100000, avg_20d_volume: 1200000, is_volume_anomaly: true },
  price_data: { current_price: 147.32, rsi_14: 81, price_vs_sma20: 12.4, is_overbought: true },
  news_data: {
    total_headlines: 30,
    extreme_language_ratio: 0.42,
    source_diversity: 0.28,
    headline_similarity: 0.61,
    top_headlines: [
      { title: "GME soars 200% as retail traders pile in", source: "Reddit", hype_score: 91 },
      { title: "GameStop short squeeze incoming say analysts", source: "MarketWatch", hype_score: 78 },
      { title: "Why GameStop is rallying again", source: "Bloomberg", hype_score: 34 },
      { title: "GME options volume hits record high", source: "Reuters", hype_score: 55 },
      { title: "GameStop reports quarterly earnings next week", source: "SEC", hype_score: 12 },
    ]
  },
  investment_advice: {
    action: "WAIT",
    deploy_now_pct: 20,
    deploy_now_inr: 1000,
    deploy_now_usd: 12,
    wait_days: 4,
    reason: "High manipulation risk. Deploy only 20% now, wait for volume normalization."
  }
}
```

Use `useAnalyze()` data even while backend is not ready.
No Stubb work just write pure logic and wire up things as required so that once everything is ready its work perfectly 
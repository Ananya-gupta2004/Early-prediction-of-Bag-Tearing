# 🏭 Baghouse Filter Bag Tear Detection System
> AI-powered early warning system that predicts industrial filter bag failure **before it happens** — using multi-sensor fusion and machine learning.

---

## 📌 Overview

Industrial baghouse filters in steel and cement plants use hundreds of fabric bags to trap dust. When a bag tears, unfiltered air bypasses the system — causing pollution violations, equipment damage, and unplanned shutdowns.

Traditional systems detect a tear **after** it occurs. This system detects it **before** — providing 10 minutes to several weeks of advance warning depending on fault type.

---

## 🔍 How It Works

Two sensors are monitored continuously:

### 1️⃣ Differential Pressure (ΔP)
Measures the resistance to airflow across the filter bags.
- **Normal** → Stable sawtooth pattern from periodic cleaning pulses
- **Clogging** → Steadily rising trend as dust accumulates on bags
- **Pre-Tear** → Highly volatile and unstable — bag is under mechanical stress
- **Tear** → **Sudden sharp DROP** — air escapes through the hole, reducing resistance

> ΔP dropping is counter-intuitive but physically correct. A torn bag creates a bypass path — like a tyre puncture releasing pressure instantly.

### 2️⃣ Particulate Matter (PM)
Measures dust concentration on the clean air side of the filter.
- **Normal** → Near zero — intact bags block almost all particles
- **Clogging** → Still near zero — bags are blocked, not torn
- **Tear** → **Sudden sharp SPIKE** — unfiltered dust rushes through the hole
- **Recovery** → Gradually decays back to baseline after maintenance

> PM stays flat during clogging, making it the clearest indicator that a **tear specifically** has occurred — not just pressure buildup.

### 🎯 The Joint Detection Signature
The system targets the simultaneous occurrence of both events:

```
ΔP drops sharply  +  PM spikes sharply  =  Bag has torn
```

Engineered features — `dp_slope`, `dp_variance`, and `pm_spike_flag` — detect the **pre-conditions** of this signature minutes to weeks before it appears, enabling truly predictive alerts rather than reactive detection.

---

## ⚡ Why Two Sensors — Not One

| | ΔP Alone | PM Alone | ΔP + PM Together |
|--|---------|---------|-----------------|
| Detects clogging | ✅ Yes | ❌ No | ✅ Yes |
| Detects tear | ✅ Yes (late) | ✅ Yes (late) | ✅ Yes (early) |
| Distinguishes tear from clogging | ❌ No | ❌ No | ✅ Yes |
| Advance warning | ❌ Minimal | ❌ Minimal | ✅ 10 min – weeks |
| False positive rate | High | High | Low (<2%) |

Neither sensor alone is sufficient. ΔP cannot distinguish a tear from normal pressure fluctuation. PM cannot detect gradual deterioration. Together, their **decorrelation point** — where ΔP falls while PM rises — is unambiguous.

---

## 🏷️ Detection Phases

| Phase | ΔP | PM | Label | Action |
|-------|----|----|-------|--------|
| Normal | 130–150 mmWC, stable | ~0.03 mg/m³ | `Normal` | Monitor |
| Clogging | 150–165 mmWC, rising | ~0.03 mg/m³ | `Clogging` | Schedule inspection |
| Pre-Tear | >160 mmWC, volatile | Micro-spikes | `Pre_Tear` | **Alert — act now** |
| Tear | Drops to ~107 mmWC | Spikes to ~2.8 mg/m³ | `Tear` | Emergency shutdown |
| Recovery | Returns to 130–145 mmWC | Decays to baseline | `Recovery` | Maintenance underway |

---

## 👥 Team
Capstone Project — Thapar Institute of Engineering and Technology (TIET)

---


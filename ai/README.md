# AI Pipeline Architecture

This directory contains the Python-based AI pipeline for the Weather Intelligence project.

## Current Prototype Implementation
* **Baseline Normalization**: 2016–2020 ERA5 statistics are used *only* for the `STEA-Net` reproduction.
* **Forecast Input**: Deterministic (mocked ensemble where necessary for pipeline validation).
* **EFI**: `EFI_PROXY` based on threshold exceedance probabilities, rather than the full cumulative distribution function.

## Required for Final Problem Statement (PS)
* **Climatology**: ~30-year ERA5/IMDAA historical baseline to compute true distributions.
* **Forecast Input**: Live NEPS-G / EPS ensemble distribution.
* **EFI**: True EFI calculation `(2/pi) * integral_0_1 (F_f(p) - p) / sqrt(p * (1-p)) dp` (implemented as a fail-loud stub until data is available).

## Architecture Flow
```text
                 NWP forecast
                     │
                     ▼
             ┌───────────────┐
             │ Preprocessing │
             └───────┬───────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     STEA-Net              Climatology
     baseline                  │
          │                     ▼
          │              Anomaly / EFI
          │                     │
          └──────────┬──────────┘
                     ▼
             Extreme-event
               information
```

**Note:** The 2016-2020 normalization files required to reproduce the STEA-Net baseline remain completely segregated from the future 30-year climatology pipeline.

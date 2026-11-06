# Gravitational Wave Detections Dashboard (O1–O3a)

An interactive Power BI dashboard analyzing 40 confirmed gravitational wave detections from LIGO/Virgo's first three observing runs, built as a hands-on project alongside Microsoft's PL-300 (Power BI Data Analyst) certification.

## Data source

Event parameters pulled directly from the [GWOSC Event Portal](https://gwosc.org/eventapi/), the official public data release from the LIGO/Virgo/KAGRA collaboration. This is the same underlying catalog used in my BSc dissertation, which tested modified gravity models via gravitational wave–galaxy cross-correlations using MCMC and Bayesian inference.

- `gw_events.csv` — 40 confirmed detections (GWTC-1 through GWTC-3), with masses, distance, redshift, network SNR, spin, and false alarm rate
- `observing_runs.csv` — reference table of LIGO/Virgo observing run windows (O1–O3b) with active detector configurations

## What the dashboard covers

- **Mass Distribution by Observing Run** — scatter plot of both merging black hole masses, bubble-sized by distance, colored by observing run
- **Detections per Observing Run** — bar chart showing detection rate climbing as instrument sensitivity improved across O1 → O2 → O3a
- **Key Statistics** — total detections, average total mass, and farthest confirmed detection
- **Network SNR Over Time** — signal-to-noise ratio trend across all 40 events

## Skills demonstrated

- **Data preparation**: cleaning real-world scientific data with genuine nulls and inconsistent formats (Power Query)
- **Data modeling**: DAX-based date-range classification (`SWITCH` logic) to assign each event to its observing run, since the two tables share no direct key
- **DAX measures**: `COUNTROWS`, `AVERAGE`, `MAX`, conditional bucketing
- **Data visualization**: scatter, bar, KPI card, and time-series visuals designed to surface a clear narrative rather than just display numbers

## View the dashboard

See `GW_Dashboard.pdf` in this repo for a static export of the full dashboard.

---
*Built by Vyshailendra Azryl Pathma Raja — BSc (Hons) Physics, Astrophysics and Cosmology, University of Portsmouth*

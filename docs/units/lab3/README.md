# Lab 3 — Data Acquisition (Sentinel-2)

**Notebook:** [lab3_1_data_acquisition.ipynb](../../../notebooks/iceland-ml/lab3_1_data_acquisition.ipynb)

## Scope
Acquire and inspect Sentinel-2 L2A data for your chosen tile via the Copernicus
Data Space Ecosystem.

## What changed for 2027
The 2025/26 notebook read credentials with `os.getenv('COPERNICUS_CLIENT_ID',
<a literal id>)`, so a student with nothing configured silently ran against the
instructor's account instead of failing. It also printed the first 15 characters
of the client secret into a committed cell output. Credentials are now read with
`os.environ[...]` so a missing one raises, and nothing about a credential is ever
printed — only `'set'` or `'MISSING'`.

## Learning outcomes
- Authenticate against the Copernicus Data Space Ecosystem with OAuth2
  client-credentials, using environment variables.
- Define an AOI and temporal window, and filter candidate scenes by cloud cover.
- Stage scenes with metadata so lab 4 can tell which file is which.

## Assignment 1 constraints
Four acquisitions of **one Sentinel-2 tile in Europe**, one per month across
**March–October 2018**, each with **≤ 30 % cloud cover**, overlaid with CORINE
Land Cover.

## Suggested flow (2 h)
1. Credential check + AOI setup
2. Scene filtering and quick visualization
3. Download, verify, and write a request manifest

## Expected outputs
- Four staged Sentinel-2 L2A scenes with metadata
- A request manifest recording what was asked for and what was returned
- Inputs ready for Lab 4 preprocessing

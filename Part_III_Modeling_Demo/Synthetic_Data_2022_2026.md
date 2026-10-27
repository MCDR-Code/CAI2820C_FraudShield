# New Sparkov dataset, January 2022–December 2026

The requested full generation is separate from the one-month pilot and the
original Kaggle dataset. It uses **999 completely new customers**, matching the
999 unique card numbers actually present across the original train/test files.
All 12 configured spending profiles and their paired fraud rules are used.
Three profiles have 84 customers and the other nine have 83, totaling 999.

The generation period is **January 1, 2022 through December 31, 2026 inclusive**.
These are simulated dates, including the remaining months of 2026; no actual
future transactions are involved. Files and history are ordered forward in time.

## Completed generation — October 27, 2026

**4,989,171 transactions**, **999 unique cards**, and **10,336 fraud transactions
(0.2072%)** across all five years. Total CSV size is approximately 1.33 GB decimal
(1.24 GiB). All rows passed the quality checks, transaction IDs are unique across
all years, and card overlap with either original CSV or the pilot is zero.

| File | Transactions | Active cards | Fraud |
|---|---:|---:|---:|
| transactions_2022.csv | 964,197 | 931 | 2,178 |
| transactions_2023.csv | 995,892 | 927 | 2,052 |
| transactions_2024.csv | 1,019,948 | 928 | 2,001 |
| transactions_2025.csv | 1,048,555 | 935 | 2,010 |
| transactions_2026.csv | 960,579 | 926 | 2,095 |

There are 999 unique cards in the union of these files. The smaller active-card
counts per year reflect customers with no transactions that year. The machine-
readable [generation report](results/synthetic_2022_2026.json) contains the exact
settings, retry records, counts, file sizes and hashes. The generator and export
changes passed 38 automated checks, plus the seeded parallel/sequential check.

## Local files

Folder: `Part_III_Modeling_Demo/data/synthetic_2022_2026/`

- `transactions_2022.csv` through `transactions_2026.csv`: five primary labeled
  transaction files, each sorted chronologically and sharing the same customer
  population. They are yearly partitions, not automatic training/test splits.
  A card can have no transactions in a particular year; 999 is the total unique
  customer count across the complete dataset, not a promised count in every file.
- `customers.psv`: reusable new customer details and exact spending-profile
  assignments, separated with pipes. It is not an uploadable transaction file.
- `raw/`: untouched upstream outputs, one pipe-delimited file per profile.
- `generation.json`: generator commit, seed, profile allocation, dates, package
  versions, source hashes and any zero-transaction customer retry seeds.
- `dataset_report.json`: final row/fraud/customer counts by year/profile, previous
  card-overlap checks, quality outcome, file sizes and SHA-256 hashes.

Generated data stays local and is excluded from GitHub. The repository contains
the generation code, this explanation and a summary report with counts/hashes.
No original datasets, existing models or Phase II deliverables are changed.
Generation does not train or evaluate any model.

Two of the initial 999 generated customers had no transactions under Sparkov's
random rules. Only those two customers were regenerated with recorded secondary
seeds, preserving their identities and profiles. Initial raw files are unchanged;
the supplemental outputs are under `raw/coverage_*.psv` and their customer subsets
under `coverage/`. This conditions the finished dataset on every customer having
at least one transaction, and is explicitly recorded rather than silently
omitting customers.

## Reproduce

Use Python 3.12, working in `Part_III_Modeling_Demo`. The setup shared with the
earlier [pilot](Synthetic_Pilot.md) is needed only once:

```powershell
git clone https://github.com/namebrandon/Sparkov_Data_Generation.git data/Sparkov_Data_Generation
git -C data/Sparkov_Data_Generation checkout b5eb45c89d36f2aa4ef16044a42945bed8b96d93
python -m venv data/sparkov_env
python -m pip --python data/sparkov_env/Scripts/python.exe install -r requirements-sparkov.txt
```

Then generate and prepare:

```powershell
& data/sparkov_env/Scripts/python.exe generate_sparkov.py --worker --customers 999 --workers 4 --seed 20221027 --start 2022-01-01 --end 2026-12-31 --output data/synthetic_2022_2026
python generate_sparkov.py --output data/synthetic_2022_2026
```

Skip the first setup block if it already exists. Use a NEW output folder when
rerunning; the program refuses to overwrite prepared data. For a smaller or
larger new population, change `--customers`. At least 12 customers are needed
to cover all profiles. Change the seed to generate another set.

The generation worker uses four independent profile processes with separate
seeded Python/NumPy/Faker streams. A 60-customer check with four processes matched
the original sequential customer file and all 12 transaction outputs exactly.
Preparation reads raw data in 100,000-row chunks and sorts one year at a time;
it does not load the complete five-year dataset into a single dataframe. It
also checks transaction IDs for duplicates across all years. No GPU is used.

## Compatibility and meaning

The CSVs retain the 22 named Kaggle transaction fields, including `cc_num`,
`merchant`, dates, coordinates, amounts and `is_fraud`. There is no unused CSV
index column. Card numbers and ZIP codes are preserved as strings. Raw generator
date/time columns are combined, merchants get the Kaggle-style `fraud_` prefix,
and `unix_time` is exported as UTC epoch seconds derived from the ISO timestamp.
The model uses `trans_date_trans_time`, not the redundant epoch field.

Customers are assigned profiles using Sparkov's demographic rules, with adult
birthdates anchored to January 1, 2022 rather than the computer's current date.
Assigned spending profiles stay fixed during this generation; this is not a
simulation of customers transitioning between age groups. Each year contains
transactions for the same customers; earlier years can supply their history.
Never use a later transaction as context for an earlier one.

The pinned upstream transaction and fraud logic is unchanged. In that version,
nearly every customer gets one short fraud window over the ENTIRE requested
generation period. A five-year run therefore has a different fraud proportion
from the earlier one-month run. Equal profile coverage also differs from
population weighting. These files are additional synthetic scenarios, not an
exact reproduction of the Kaggle author's unrecorded settings.

The demo accepts one review CSV and supports sequential or multi-file history
additions, with updated counts and duplicate-ID checks. A follow-up removed fixed file-size and row
caps, so full-year files can be loaded, subject to available memory and processing
time. This generation does not add multi-file uploads, retrain v3 or change the
training program's original-dataset requirements.

The upload update passed 39 automated tests. A complete `transactions_2022.csv`
upload (964,197 rows / 258,971,874 bytes) was also scored through the demo endpoint
using the saved Random Forest: HTTP 200, all 964,197 results returned, no row
truncation. This is a compatibility check, not a reported performance evaluation
against the new fraud labels. Large uploads still need time and memory.

Source: [Pinned Sparkov generator](https://github.com/namebrandon/Sparkov_Data_Generation/tree/b5eb45c89d36f2aa4ef16044a42945bed8b96d93).

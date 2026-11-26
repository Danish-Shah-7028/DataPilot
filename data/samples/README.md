# Sample datasets

Small demo files built from two public datasets by `datapilot-samples`
(`src/datapilot/data/samples.py`). Rebuilding them gives the same files because the
sampling uses a fixed seed.

## online_retail.csv and retail.sqlite

Source: Daqing Chen, *Online Retail II*, UCI Machine Learning Repository, 2012.
<https://doi.org/10.24432/C5CG6D>. Licence: CC BY 4.0.

Transactions of a UK online gift retailer between December 2009 and December 2011.
What I changed:

- kept only completed product sales (dropped cancelled invoices, returns, zero prices,
  and bookkeeping lines such as postage, bank charges and bad-debt adjustments);
- kept a random 4% of invoices (seed 42), always whole invoices;
- renamed the columns to snake_case.

`online_retail.csv` is the flat line-level file. `retail.sqlite` holds the same rows
split into `customers`, `products`, `invoices` and `invoice_lines`, plus a small
`country_regions` lookup (region and continent per country) that I added so questions
like "by region" or "only for Europe" have something to work with.

## customer_churn.xlsx

Source: *Iranian Churn Dataset*, UCI Machine Learning Repository, 2020.
<https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset>. Licence: CC BY 4.0.

3,150 customers of an Iranian telecom company over 12 months with a churn label. I
converted it to Excel (so the demo covers all three upload formats), renamed the columns
to snake_case and added a `customer_id` column numbered in the original row order.

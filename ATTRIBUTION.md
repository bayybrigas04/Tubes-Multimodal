# Attribution — SEA-VL Indonesian subset

This dataset is a filtered and re-packaged derivative of:

- **Dataset**: `KORIKA-AI/sea-vl_crowdsourcing_id`
  (Indonesian subset of `SEACrowd/sea-vl_crowdsourcing`)
- **Licence**: Creative Commons Attribution-ShareAlike 4.0 International
  (CC BY-SA 4.0) — https://creativecommons.org/licenses/by-sa/4.0/
- **Paper**: Cahyawijaya et al., *Crowdsource, Crawl, or Generate? Creating
  SEA-VL, a Multicultural Vision-Language Dataset for Southeast Asia*,
  ACL 2025.

## Obligations when you use this data

1. **Attribution** — credit SEA-VL / SEACrowd / KORIKA-AI and cite the paper
   above wherever results are published or presented.
2. **ShareAlike** — if you redistribute this derivative dataset (or a modified
   version of it), you must release it under the same CC BY-SA 4.0 licence.
3. **No additional restrictions** — you may not apply legal terms that
   prevent others from doing what the licence permits.

## Local modifications applied

- retained only rows whose `culture_relevant_loc` contains "Indonesia"
- removed duplicate images (identical MD5)
- converted all images to RGB and capped the long side
- added a coarse keyword-based `category` column and a `split` column

Number of retained samples: **2763**

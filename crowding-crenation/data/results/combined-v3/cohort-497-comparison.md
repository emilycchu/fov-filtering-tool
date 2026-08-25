# Tanzania cohort at full size: 497 slides, 271 positive + 226 negative

Slide-level `density_mean` (gated FOVs contribute 0.0, the headline convention).

| group | n | min | p25 | median | p75 | max | mean |
|---|---|---|---|---|---|---|---|
| positive | 271 | 0.015 | 0.161 | 0.247 | 0.370 | 0.707 | 0.268 |
| negative | 226 | 0.012 | 0.207 | 0.320 | 0.435 | 0.715 | 0.328 |
| all 497 | 497 | 0.012 | 0.177 | 0.286 | 0.399 | 0.715 | 0.295 |

## Do the negatives sit elsewhere?

- Mann-Whitney z = -4.56, p = 5.16e-06
- rank-biserial correlation = -0.237 (0 = a random positive is as likely to be denser as sparser than a random negative)
- two-sample KS distance = 0.214
- median difference = -0.073 (0.247 positive vs 0.320 negative)

- positive buckets: {'slightly dense': 24, 'monolayer': 76, 'sparser': 165, 'dense': 6}
- negative buckets: {'slightly dense': 42, 'monolayer': 81, 'sparser': 91, 'dense': 11, 'very dense': 1}

## Site is confounded with truth

| site | n pos | n neg | median dens pos | median dens neg |
|---|---|---|---|---|
| KIT | 104 | 43 | 0.263 | 0.318 |
| KTR | 29 | 45 | 0.291 | 0.315 |
| NKR | 46 | 79 | 0.282 | 0.379 |
| RUB | 92 | 58 | 0.187 | 0.215 |
| RUK | 0 | 1 | - | 0.559 |

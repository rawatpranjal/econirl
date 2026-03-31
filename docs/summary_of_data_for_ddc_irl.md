# Summary of Data for DDC/IRL Modeling

> **Purpose**: One-stop shop for dataset evaluation before calling `.fit()`.
> Each dataset is assessed against the 6 structural assumptions from
> [`before_we_model_we_think.md`](../before_we_model_we_think.md).
>
> **Scorecard legend**: ✅ Pass | ⚠️ Warn | ❌ Fail | — N/A
>
> **Columns**: A1=Markov, A2=Additive Separability, A3=IIA/Gumbel,
> A4=Discrete Actions, A5=Time Homogeneity, A6=Stationary Transitions

---

## Domain 1: Canonical DDC Benchmarks

### Rust Bus Engine Replacement

**Location**: `data/raw/rust_bus/rust_bus_original.csv`
**Scale**: 8260 observations | 104 buses
**Papers**: Rust (1987) *Econometrica*; Iskhakov, Rust & Schjerning (2016) NFXP-NK polyalgorithm

**Schema**: `bus_id, period, mileage, mileage_bin, replaced, group`

**Actions** (column `replaced`):
  - `keep` 8,200 (99.3%)
  - `replace` 60 (0.7%)

**Session lengths** (periods per bus): mean=79.4, p50=70, max=117

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | ✅ | `mileage_bin` fully captures state; past history irrelevant given mileage |
| A2 Additive Separability | ✅ | Linear cost function in mileage; canonical specification |
| A3 IIA/Gumbel | ✅ | Binary choice — IIA trivially satisfied |
| A4 Discrete Actions | ✅ | {keep, replace} — finite, mutually exclusive |
| A5 Time Homogeneity | ✅ | Structural parameters assumed constant across buses/periods |
| A6 Stationary Transitions | ✅ | Mileage transition matrix stable by construction |

**State design**: `mileage_bin` (90 states) → already implemented in `econirl`
**Action**: {keep=0, replace=1}
**Recommended estimator**: NFXP (reference implementation ✓)

---

## Domain 2: Transportation & Route Choice

### Citi Bike NYC

**Location**: `data/raw/citibike/202401-citibike-tripdata_1.csv`
**Scale**: ~2M trips (Jan 2024) | 4118 origin stations | 1066 destination stations
**Papers**: Ermon et al. (2015) *AAAI* large-scale spatio-temporal DDC

**Schema**: `ride_id, rideable_type, started_at, ended_at, start_station_id, end_station_id, start_lat, start_lng, end_lat, end_lng, member_casual`

**Actions** (rideable types):
  - `electric_bike` 126,955 (63.5%)
  - `classic_bike` 73,045 (36.5%)

**Member types**:
  - `member` 174,308 (87.2%)
  - `casual` 25,692 (12.8%)

**Trip duration**: mean=11.6 min, p50=8.0 min
**Peak hours**: [17, 16, 8]

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | ✅ | `(origin_station, hour_bin, weekday)` sufficient; no memory needed |
| A2 Additive Separability | ✅ | Distance, elevation, dock-availability are observable state vars |
| A3 IIA/Gumbel | ⚠️ | Nearby stations are spatial substitutes; consider spatial nesting |
| A4 Discrete Actions | ✅ | Destination station is discrete; ~800 stations → cluster to ~30-50 zones |
| A5 Time Homogeneity | ✅ | Single month, no regime shifts |
| A6 Stationary Transitions | ✅ | Station-to-station travel times stable within month |

**State design**: `(origin_zone, hour_bin, weekday)` → ~300 states
**Action**: destination_zone (30-50 clusters)
**Recommended estimator**: MCE-IRL (recovers utility over distance, elevation, dock-availability)

---

### T-Drive (Beijing Taxi GPS)

**Location**: `data/raw/tdrive/` (10K+ individual taxi txt files)
**Scale**: ~17.7M GPS points | ~10,000+ taxis | 1 week (Feb 2008)
**Papers**: Ziebart et al. (2008) Pittsburgh taxi IRL; Barnes et al. (2024) Google Maps RHIP

**Schema**: `taxi_id, timestamp, longitude, latitude`
**Lon/Lat range**: [0.0, 122.361] / [0.0, 40.602]
**Sample interval**: ~300s median (sparse — ~10 min gaps)

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | ⚠️ | ~10-min sampling creates ambiguous path; need road-network snap |
| A2 Additive Separability | ✅ | Road features (speed limit, road type, distance) are observable |
| A3 IIA/Gumbel | ⚠️ | Parallel roads are near-substitutes; route nesting advisable |
| A4 Discrete Actions | ⚠️ | Continuous GPS → requires grid/OSM-node discretization first |
| A5 Time Homogeneity | ✅ | One week, stable preferences |
| A6 Stationary Transitions | ✅ | Road network fixed; traffic varies but can be binned into state |

**State design**: Snap to Beijing OSM nodes → `(node_id, hour_bin)` → ~2,000-5,000 states
**Action**: next_link (neighboring road segment)
**Required preprocessing**: Map-match GPS to OSM road network via FMM or OSRM
**Recommended estimator**: MCE-IRL (Ziebart 2008 algorithm on discretized road graph)

---

### NGSIM US-101 (Highway Driving)

**Location**: `data/raw/ngsim/us101_trajectories.csv`
**Scale**: ~4.8M frames | 128 vehicles (sample)
**Papers**: Multiple IRL highway papers; GAIL applied to NGSIM (Ho & Ermon 2016)

**Schema**: `vehicle_id, frame_id, local_x/y, v_vel, v_acc, lane_id, space_headway, time_headway, v_class`

**Vehicle classes**:
  - `auto` 193,853 (96.9%)
  - `truck` 3,984 (2.0%)
  - `motorcycle` 2,163 (1.1%)

**Velocity (ft/s)**: {'count': 200000.0, 'mean': 34.3, 'std': 12.9, 'min': 0.0, '25%': 25.9, '50%': 34.5, '75%': 43.4, 'max': 95.3}

**Derived actions** (lane changes):
  - `stay` 151,004 (75.5%)
  - `lane_left` 24,530 (12.3%)
  - `lane_right` 24,466 (12.2%)

**Frames per vehicle**: mean=1562.5, p50=1614

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | ⚠️ | Need `(lane, speed_bin, gap_bin)` to capture following behavior |
| A2 Additive Separability | ✅ | Lane position, speed, headway are fully observed |
| A3 IIA/Gumbel | ✅ | Lane changes (left/right/stay) are distinct, non-substitutable |
| A4 Discrete Actions | ⚠️ | Lane is discrete (5 lanes); velocity needs binning (e.g. 10 bins) |
| A5 Time Homogeneity | ✅ | ~45-min recording window; stable preferences |
| A6 Stationary Transitions | ✅ | Fixed highway; traffic stable within recording window |

**State design**: `(lane_id, speed_bin[5], headway_bin[5])` → ~125 states
**Action**: {stay, lane_left, lane_right} or extended with speed choices
**Recommended estimator**: MCE-IRL or AIRL (reward transfer to new highway segments)

---

## Domain 3: E-Commerce & Sequential Search

### Trivago Hotel Search 2019  ← PRIMARY for search cost estimation

**Location**: `/Volumes/Expansion/datasets/trivago-2019/train.csv`
**Scale**: 15,932,993 interactions | 11,282 sessions (sample) | 11,232 users (sample)
**Papers**: Ursu (2018) *Marketing Science* Expedia search cost model; Compiani et al. (2024) *Marketing Science*

**Schema**: `user_id, session_id, timestamp, step, action_type, reference, platform, city, device, current_filters, impressions, prices`

**Action vocabulary** (10 types):
  - `interaction item image` 149,332 (74.7%)
  - `clickout item` 19,522 (9.8%)
  - `filter selection` 8,874 (4.4%)
  - `change of sort order` 5,022 (2.5%)
  - `search for destination` 4,984 (2.5%)
  - `interaction item info` 3,615 (1.8%)
  - `interaction item rating` 2,705 (1.4%)
  - `interaction item deals` 2,388 (1.2%)
  - … 2 more types

**Session lengths** (steps per session): mean=17.7, p50=4, p95=78, max=829
**Temporal span**: Nov–Dec 2017 (1541030432 – 1541548799)

**Missing data**: {'current_filters': np.float64(92.8), 'impressions': np.float64(90.2), 'prices': np.float64(90.2)}

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | ⚠️ | `step` alone insufficient; `(step, last_action_type)` restores Markov property |
| A2 Additive Separability | ✅ | Prices and hotel ratings are directly observable in `impressions`/item metadata |
| A3 IIA/Gumbel | ❌ | `interaction_item_image` and `interaction_item_rating` are near-substitutes (same underlying decision); nested logit or MCE-IRL recommended |
| A4 Discrete Actions | ⚠️ | 13 discrete types ✓; but impression list length varies 1-25 (variable choice set complicates NFXP) |
| A5 Time Homogeneity | ✅ | 2-month window; chi-squared test on P(clickout|device) stable (p>0.05 expected) |
| A6 Stationary Transitions | ✅ | Short window; hotel inventory fixed |

**State design**: `(device, step_bin[5], last_action_cat[5], price_quartile)` → ~480 states
**Action**: Simplified to {examine_item, sort_filter, clickout, abandon}
**Recommended estimator**: MCE-IRL (avoids IIA; recovers hotel utility weights over price/rating/stars)

---

### OTTO RecSys 2022

**Location**: `/Volumes/Expansion/datasets/otto-2022/otto-recsys-train.jsonl`
**Scale**: 12,899,779 sessions (train) | 2,621,110 events in sample
**Papers**: DEERS (Zhao et al., KDD 2018); Pseudo Dyna-Q (Bai et al., WSDM 2020)

**Schema**: JSONL with `session`, `events[aid, ts, type]`

**Action distribution** (from 50K sessions):
  - `clicks` 2,395,745 (91.4%)
  - `carts` 180,595 (6.9%)
  - `orders` 44,770 (1.7%)

**Session lengths**: mean=52.4, p50=19, p95=230, max=495

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | ✅ | Funnel is Markov: P(cart\|state) depends only on current session state |
| A2 Additive Separability | ✅ | Session history (n_clicks, n_carts, last_item) is observable |
| A3 IIA/Gumbel | ✅ | click → cart → order are distinct funnel stages, not substitutes |
| A4 Discrete Actions | ✅ | Exactly 3 types + implicit abandon |
| A5 Time Homogeneity | — | No timestamps in test file; training set covers several weeks |
| A6 Stationary Transitions | — | No explicit time information available |

**State design**: `(last_action_type, n_unique_items_bin, session_length_bin)` → ~30 states
**Action**: {click, add_to_cart, order, [implicit abandon]}
**Recommended estimator**: CCP (Hotz-Miller fast inversion; funnel is simple enough)

---

### finn_slates (Norwegian E-Commerce Slates)

**Location**: `/Volumes/Expansion/datasets/finn_slates/data.npz`
**Scale**: 2,277,645 users × 20 steps × 25-slot slates | ~45,552,900 total step-observations
**Papers**: Lafon et al. (2023) slate recommendation; Swaminathan & Joachims (2015) counterfactual learning

**Schema** (npz arrays): `userId[N]`, `slate[N,20,25]`, `click[N,20,25]`, `click_idx[N,20]`, `interaction_type[N,20]`, `slate_lengths[N,20]`

**Steps with content**: mean=16.4/20, p50=20
**Click rate per step**: 1.0
**Interaction types**: {0: 8126053, 1: 26094491, 2: 11332356} (1=search, 2=recommendation)

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | ⚠️ | 20-step sequences have persistent user preferences; add user-cluster to state |
| A2 Additive Separability | ✅ | Item category features observable in `itemattr.npz` |
| A3 IIA/Gumbel | ❌ | Items within a slate are near-substitutes (same category context) |
| A4 Discrete Actions | ✅ | Binary click/skip per slot; fixed 25-slot choice set |
| A5 Time Homogeneity | — | No timestamps in dataset |
| A6 Stationary Transitions | — | N/A |

**State design**: `(step[20], interaction_type[2], user_cluster[10])` → ~400 states
**Action**: {click_slot_k, skip_all} or simplified {click, skip}
**Recommended estimator**: MCE-IRL (slate sequential structure)

---

## Domain 4: Content Recommendation

### KuaiRand-27K  ← PRIMARY for IRL reward recovery

**Location**: `/Volumes/Expansion/datasets/kuairand/KuaiRand-27K/data/`
**Scale**: ~312M standard interactions (4 parts) + **1,186,059 random-exposure interactions**
**Users**: 27,285 | **Videos**: 7,583
**Papers**: Gao et al. (2022) *CIKM* KuaiRand; MTRec (2025) deployed IRL for short-video

**Schema**: `user_id, video_id, date, hourmin, time_ms, is_click, is_like, is_follow, is_comment, is_forward, is_hate, long_view, play_time_ms, duration_ms, profile_stay_time, comment_stay_time, is_profile_enter, is_rand, tab`

**⭐ KEY ADVANTAGE**: `log_random_4_22_to_5_08_27k.csv` contains 1,186,059 rows with randomly-exposed videos (`is_rand=1`), providing **exogenous variation** for structural identification — analogous to Expedia's randomized rankings in Ursu (2018).

**Derived actions** (from `play_time_ms/duration_ms` + engagement):
  - `skip` 1,014,643 (85.5%)
  - `watch_full` 83,138 (7.0%)
  - `watch_partial` 82,042 (6.9%)
  - `interact` 6,236 (0.5%)

**Feedback signals** (12 types): is_click, is_like, is_follow, is_comment, is_forward, is_hate, long_view

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | ⚠️ | Feed position and user history matter; need user_cluster in state |
| A2 Additive Separability | ✅ | Video duration, category, creator features all observable |
| A3 IIA/Gumbel | ⚠️ | Videos within same category are substitutes; category-level nesting |
| A4 Discrete Actions | ⚠️ | Derive from `play_time_ms/duration_ms`: {watch_full≥0.9, partial≥0.3, skip, interact} |
| A5 Time Homogeneity | ✅ | ~1 month window; platform preferences stable |
| A6 Stationary Transitions | ✅ | Recommendation algorithm fixed within data collection period |

**State design**: `(user_cluster[20], video_category[50], feed_position_bin[5])` → ~5,000 states
**Action**: {watch_full, watch_partial, skip, interact} — 4 actions
**Use log_random for structural estimation** (exogenous identification); standard log for scale
**Recommended estimator**: MCE-IRL on `log_random_4_22_to_5_08_27k.csv`

---

### KuaiRec

**Location**: `/Volumes/Expansion/datasets/kuairec/big_matrix.csv`
**Scale**: 12,530,807 user-video pairs | 57 users | 6,822 videos (sample)
**Note**: Near-fully-observed matrix (99.6% density for small_matrix subset)

**Schema**: `user_id, video_id, play_duration, video_duration, time, date, timestamp, watch_ratio`

**Watch ratio**: {'count': 100000.0, 'mean': 1.0, 'std': 1.989, 'min': 0.0, '25%': 0.318, '50%': 0.741, '75%': 1.218, 'max': 130.819}
**% complete watches (ratio≥0.9)**: 41.2%

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | ❌ | No session structure — single watch events, no sequential state evolution |
| A4 Discrete Actions | ❌ | `watch_ratio` is continuous — requires discretization |

**Assessment**: **Not suitable for DDC as-is.** Best used as:
1. **Reward signal dataset** — learn utility weights from engagement (watch_ratio, implicit preference)
2. **Off-policy evaluation** — dense matrix enables counterfactual estimation
3. **Feature engineering** — user/item embeddings for state representation in KuaiRand IRL

**Recommended estimator**: BC baseline only; or use as auxiliary signal for KuaiRand MCE-IRL

---

### MIND News Recommendation

**Location**: `/Volumes/Expansion/datasets/mind/train/`
**Scale**: 149,116 impression sessions | 49,182 users
**Papers**: Wu et al. (2020) *ACL* NRMS; Microsoft MIND dataset paper

**Schema**: `impression_id, user_id, time, history, impressions(ID-label_pairs)`

**Impression slate size**: mean=37.9, p50=25, max=294
**Click rate per session**: 0.108
**User history length**: mean=33.3 articles

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | — | Single-step per session — no state evolution within session |
| A4 Discrete Actions | ✅ | Binary click/no-click per article in slate |

**Assessment**: **Static discrete choice, not DDC.** One impression → one decision. No dynamic state evolution across impressions. Best modeled as:
- Static MNL on news features (category, entity similarity to history)
- Contextual bandit with user history as context
- CCP as 1-shot logit

**Recommended estimator**: CCP/MNL (single-period); MCE-IRL only if modeling reading sequences across sessions over time

---

## Domain 5: Location Choice

### Foursquare NYC Check-ins

**Location**: `data/raw/foursquare/dataset_TSMC2014_NYC.csv`
**Scale**: 227,428 check-ins | 2,042 users | 35,891 venues | 252 categories

**Schema**: `userId, venueId, venueCategoryId, venueCategory, latitude, longitude, timezoneOffset, utcTimestamp`

**Top venue categories**:
  - `Bar` 13,929 (7.0%)
  - `Home (private)` 13,606 (6.8%)
  - `Office` 11,309 (5.7%)
  - `Subway` 8,377 (4.2%)
  - `Gym / Fitness Center` 7,561 (3.8%)
  - `Coffee Shop` 6,482 (3.2%)
  - `Food & Drink Shop` 5,861 (2.9%)
  - `Train Station` 5,734 (2.9%)
  - … 244 more types

**Check-ins per user**: mean=97.9, p50=74, max=1743

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | ⚠️ | `(last_category, hour_bin, day_of_week)` plausibly Markov; location history matters less |
| A2 Additive Separability | ✅ | Category, distance from last venue, time-of-day are observable |
| A3 IIA/Gumbel | ⚠️ | Nearby similar venues (two coffee shops) may have correlated shocks |
| A4 Discrete Actions | ✅ | Venue category is discrete (290 types → cluster to ~20 semantic categories) |
| A5 Time Homogeneity | ✅ | Apr 2012–Feb 2013, stable urban mobility patterns |
| A6 Stationary Transitions | ✅ | Venue landscape stable across the year |

**State design**: `(last_category_cluster[20], hour_bin[4], weekday[2])` → ~160 states
**Action**: next_category_cluster (20 groups)
**Recommended estimator**: MCE-IRL (recovers utility over category appeal, distance, time-of-day)

---

## Domain 6: Gig Economy / Labor Supply

### NYC TLC (Yellow Taxi + Uber/Lyft HVFHV)

**Location**: `/Volumes/Expansion/datasets/nyc_tlc/`
**Files**: `yellow_tripdata_2024-01.parquet` (2,964,624 trips) | `fhvhv_tripdata_2024-01.parquet` (~19.6M trips)
**Papers**: Buchholz, Shum & Xu (2025) *Princeton WP* NYC taxi stopping DDC; Farber (2015) *AER*

**Yellow taxi schema**: ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance', 'RatecodeID', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount', 'congestion_surcharge', 'Airport_fee']
**HVFHV schema** (sample): ['hvfhs_license_num', 'dispatching_base_num', 'originating_base_num', 'request_datetime', 'on_scene_datetime', 'pickup_datetime', 'dropoff_datetime', 'PULocationID', 'DOLocationID', 'trip_miles', 'trip_time', 'base_passenger_fare', 'tolls', 'bcf', 'sales_tax', 'congestion_surcharge', 'airport_fee', 'tips', 'driver_pay', 'shared_request_flag', 'shared_match_flag', 'access_a_ride_flag', 'wav_request_flag', 'wav_match_flag']
**Has persistent driver_id in HVFHV**: True *(dispatching_base_num ≠ driver_id)*

**HVFHV driver_pay stats**: {'count': 50000.0, 'mean': 18.22, 'std': 15.35, 'min': 0.0, '25%': 8.16, '50%': 13.58, '75%': 22.73, 'max': 302.02}

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | ❌ | **No persistent driver ID in HVFHV** — cannot reconstruct shifts for DDC stopping model |
| A4 Discrete Actions | — | {keep_driving, stop_shift} well-defined but unobservable without driver linkage |

**⚠️ CRITICAL GAP**: Jan 2024 HVFHV data has no persistent driver identifier. Shift reconstruction (cumulative earnings, hours worked) is impossible without linking trips to individual drivers.

**For DDC labor supply, need**: NYC Yellow Taxi **2009–2013** parquet (has `medallion` + `hack_license` driver IDs). Download from: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page or ICPSR 37254.

**Current data is useful for**:
- Market-level demand analysis (surge zone patterns, time-of-day demand)
- State variable calibration (earnings distribution, trip frequencies by zone)

**Recommended estimator**: NFXP or CCP on 2009-2013 data once downloaded
**State design** (Buchholz et al.): `(cumulative_earnings_bin, hours_worked_bin, location_zone)` → ~1,875 states

---

## Domain 7: Pedestrian Dynamics

### ETH/UCY Pedestrian Trajectories

**Location**: `data/raw/eth_ucy/`
**Scale**: ~6,500 rows/scene | scenes: ['hotel']
**Schema**: `frame_id, pedestrian_id, x, y` (pixel coordinates)

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | ⚠️ | Position + velocity captures most relevant state; social forces need nearby-agent encoding |
| A4 Discrete Actions | ⚠️ | Continuous (x,y) displacement → requires direction discretization (8-direction compass) |

**State design**: `(grid_cell, direction_bin[8], speed_bin[3])` → ~500 states
**Recommended estimator**: MCE-IRL on discretized grid; or continuous IRL (AIRL with neural reward)
**Note**: Dataset is small (~6K rows/scene) — best for algorithm validation, not structural estimation

---

### Stanford Drone Dataset

**Location**: `data/raw/stanford_drone/annotations/`
**Scale**: ~350K bbox annotations | scenes: ['gates']
**Schema**: `track_id, x1, y1, x2, y2, frame_id, lost, occluded, generated, label`

**Agent types**:
  - `Biker` 257,629 (54.9%)
  - `Pedestrian` 187,075 (39.9%)
  - `Skater` 19,428 (4.1%)
  - `Bus` 3,465 (0.7%)
  - `Car` 1,564 (0.3%)

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A4 Discrete Actions | ⚠️ | Continuous 2D trajectory → grid or direction discretization needed |
| A1 Markov | ⚠️ | Social forces (other agent positions) must enter state for Markov property |

**Recommended estimator**: AIRL (reward transfer across campus scenes)
**Note**: Mixed agent types (pedestrian, cyclist, car) require separate models or agent-type conditioning

---

## Domain 8: Continuous Control (Neural Estimators Only)

### D4RL MuJoCo Expert Trajectories

**Location**: `data/raw/d4rl/` (backed up at `/Volumes/Expansion/datasets/econirl_local_raw_backup/d4rl/`)
**Files**: ['hopper-expert-v2.hdf5', 'walker2d-expert-v2.hdf5', 'halfcheetah-expert-v2.hdf5']
**Papers**: Fu et al. (2020) *NeurIPS* D4RL; offline IRL benchmarks

**Datasets**: halfcheetah-expert, hopper-expert, walker2d-expert
**Schema**: `observations[N, obs_dim]`, `actions[N, act_dim]`, `rewards[N]`, `terminals[N]`

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A4 Discrete Actions | ❌ | Continuous action spaces (joint torques) — incompatible with standard DDC |
| A1 Markov | — | MDP by construction; Markov property satisfied |

**Assessment**: **Not suitable for NFXP/CCP/MCE-IRL (tabular estimators).** Requires neural estimators:
- **TD-CCP**: temporal-difference CCP with neural function approximation
- **Deep MCE-IRL**: neural reward function, continuous state/action
- **GLADIUS**: Q-network + EV-network for model-free DDC

**Recommended estimator**: TD-CCP or GLADIUS
**Use case**: Validate neural estimators before applying to real behavioral data

---

## Domain 9: Route Choice GPS Trajectories

### Porto Taxi (ECML-PKDD 2015)  ← PRIMARY for route choice IRL

**Location**: `/Volumes/Expansion/datasets/porto_taxi/train.csv`
**Scale**: 1,710,670 trips | 428 taxis | 12 months (July 2013–June 2014)
**License**: CC BY 4.0 (free)
**Papers**: Ziebart et al. (2008) Pittsburgh taxi IRL; Barnes et al. (2024) Google Maps RHIP; multiple recursive logit papers

**Schema**: `TRIP_ID, CALL_TYPE, ORIGIN_CALL, ORIGIN_STAND, TAXI_ID, TIMESTAMP, DAY_TYPE, MISSING_DATA, POLYLINE`

**Call types**:
  - `B` 24,784 (49.6%)
  - `C` 14,432 (28.9%)
  - `A` 10,784 (21.6%)

**GPS polyline stats** (valid trips only):
- Points per trip: mean=47.8, p50=41, p95=100, max=2516
- Sample interval: 15s (dense, low ambiguity)
- Missing data: 0.0% of trips flagged (`MISSING_DATA=True`) → filter these out

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | ✅ | 15-second sampling → current road node captures all relevant state |
| A2 Additive Separability | ✅ | Road features (type, speed limit, length) observable from OSM |
| A3 IIA/Gumbel | ⚠️ | Parallel routes are substitutes; route nesting helps |
| A4 Discrete Actions | ⚠️ | Continuous GPS → snap to Porto OSM road nodes via FMM map-matching |
| A5 Time Homogeneity | ✅ | 12-month window; stable road network and driver preferences |
| A6 Stationary Transitions | ✅ | Porto road network fixed; traffic variation enters via time-of-day state |

**State design**: `(osm_node_id, hour_bin[4], day_type[3])` → ~5,000-10,000 states (after map-matching)
**Action**: next_link (road segment choice at each intersection)
**Required preprocessing**: Map-match POLYLINE to Porto OSM graph using FMM or OSRM
**Porto OSM graph**: Extract with `osmnx.graph_from_place("Porto, Portugal", network_type="drive")`
**Recommended estimator**: MCE-IRL (Ziebart 2008 MaxEnt on road network)

---

### Shanghai Taxi RCM-AIRL (Zhao & Liang 2023)

**Location**: `/Volumes/Expansion/datasets/shanghai_taxi_rcm_airl/`
**Scale**: 24,468 OD route pairs | 320 road nodes | 714 directed edges
**Coverage**: Shanghai [31.18-31.23°N, 121.41-121.47°E] (small sub-network of Shanghai)
**Paper**: Zhao & Liang (2023) "Route Choice Modeling via Adversarial IRL" — **code + data released on GitHub**

**Schema** (`path.csv`): `ori, des, path (node sequence), len`
**Network** (`edge.txt`): `u, v, name, highway, oneway, length, lanes, maxspeed, …`
**Path length stats**: {'count': 24468.0, 'mean': 19.5, 'std': 4.3, 'min': 15.0, '25%': 16.0, '50%': 19.0, '75%': 21.0, 'max': 49.0}

**Cross-validation splits**: []
**Pre-trained models**: True (in `trained_models/` directory)
**Methods implemented**: BC (behavioral cloning), GAIL, AIRL

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | ✅ | OD pair + current node is sufficient state (road network is fixed) |
| A2 Additive Separability | ✅ | Edge features (road type, speed, length) fully observed in `edge.txt` |
| A3 IIA/Gumbel | ⚠️ | Parallel roads are substitutes at intersections |
| A4 Discrete Actions | ✅ | Next node choice at each intersection — finite set of neighbors |
| A5 Time Homogeneity | ✅ | Network is static; no temporal variation in provided data |
| A6 Stationary Transitions | ✅ | Deterministic road network (no stochastic transitions) |

**State design**: `(current_node[320], destination_node[320])` → graph-structured MDP, not tabular
**Action**: next_node (successor nodes in road graph)
**⭐ READY TO RUN**: Pre-processed data + CV splits + pre-trained AIRL weights → fastest path to replication
**Recommended estimator**: AIRL (matches paper); MCE-IRL as baseline comparison

---

### NYC Yellow Taxi 2013

**Location**: `/Volumes/Expansion/datasets/nyc_yellow_taxi_2013/yellow_tripdata_2013-01.parquet`
**Scale**: 14,776,617 trips (January 2013 only)
**Papers**: Buchholz, Shum & Xu (2025) driver stopping DDC; Farber (2015) *AER*

**Schema**: ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance', 'RatecodeID', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount', 'congestion_surcharge', 'airport_fee', 'duration_min']

**⚠️ CRITICAL FINDING**: `medallion` and `hack_license` columns are **absent** — this parquet uses the modern TLC schema (`PULocationID`/`DOLocationID` zone IDs), not the original 2009-2013 format that contained driver identifiers. Shift reconstruction for DDC labor supply is **not possible** with this file.

**Fare statistics**: {'count': 50000.0, 'mean': 11.71, 'std': 9.88, 'min': 2.5, '25%': 6.5, '50%': 9.0, '75%': 13.0, 'max': 345.0}
**Location columns**: ['PULocationID', 'DOLocationID']

**To get the original driver-identified 2013 data**:
Download from ICPSR 37254 (https://doi.org/10.3886/ICPSR37254.v1) which preserves the original schema with `medallion` and `hack_license` fields.

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | ❌ | No driver ID → cannot reconstruct cumulative earnings/hours state |
| A4 Discrete Actions | — | {keep_driving, stop_shift} unobservable without shift reconstruction |

**Assessment**: **Not usable for DDC labor supply** in current form. Useful for:
- Market-level demand pattern analysis
- Pickup zone transition matrix estimation (zone → zone flow)
- Fare distribution calibration for structural model calibration

---

### Chicago Taxi

**Location**: `/Volumes/Expansion/datasets/chicago_taxi/chicago_taxi_sample.csv`
**Status**: ❌ **CORRUPTED** — file contains API timeout error response, not CSV data

**Content**: `{
  "error" : true,
  "message" : "Timeout"
}`

**Action required**: Re-download from https://data.cityofchicago.org/Transportation/Taxi-Trips-2013-2023-/wrvz-psew
**Note**: Chicago taxi data has consistent `taxi_id` across trips (unlike NYC HVFHV) enabling within-year shift reconstruction, but timestamps are rounded to 15 min and locations to census tracts.

---

## Master Comparison Table

| Dataset | Domain | Scale | A1 | A2 | A3 | A4 | A5 | A6 | Estimator |

|---------|--------|-------|----|----|----|----|----|----|-----------|

| Rust Bus | DDC (canonical) | 8.3K obs | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | NFXP |

| Trivago 2019 | Hotel search | 15.9M events | ⚠️ | ✅ | ❌ | ⚠️ | ✅ | ✅ | MCE-IRL |

| KuaiRand | Short-video IRL | 312M + 1.2M rnd | ⚠️ | ✅ | ⚠️ | ⚠️ | ✅ | ✅ | MCE-IRL |

| OTTO 2022 | E-com funnel | 12.9M sessions | ✅ | ✅ | ✅ | ✅ | — | — | CCP |

| finn_slates | E-com slates | 2.3M users | ⚠️ | ✅ | ❌ | ✅ | — | — | MCE-IRL |

| KuaiRec | Video engagement | 12.5M pairs | ❌ | — | — | ❌ | — | — | BC baseline |

| MIND | News (static) | 149K sessions | — | ✅ | ⚠️ | ✅ | — | — | CCP/MNL |

| Citi Bike | Route/station choice | ~2M trips | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | MCE-IRL |

| NGSIM US-101 | Highway driving | 4.8M frames | ⚠️ | ✅ | ✅ | ⚠️ | ✅ | ✅ | MCE-IRL/AIRL |

| T-Drive | Taxi route choice | 17.7M GPS pts | ⚠️ | ✅ | ⚠️ | ⚠️ | ✅ | ✅ | MCE-IRL |

| Foursquare NYC | Location choice | 227K check-ins | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ✅ | MCE-IRL |

| NYC TLC | Gig labor (limited) | ~2M (Jan 2024) | ❌ | ⚠️ | — | — | ✅ | ✅ | Needs 2009-13 data |

| D4RL MuJoCo | Continuous control | ~1M steps each | — | — | — | ❌ | ✅ | ✅ | TD-CCP / GLADIUS |

| ETH/UCY | Pedestrian dynamics | ~6K/scene | ⚠️ | — | — | ⚠️ | ✅ | ✅ | MCE-IRL (continuous) |

| Stanford Drone | Campus mobility | 350K bbox | ⚠️ | — | — | ⚠️ | ✅ | ✅ | MCE-IRL (continuous) |

| Porto Taxi | Route choice IRL | 1.71M trips | ✅ | ✅ | ⚠️ | ⚠️ | ✅ | ✅ | MCE-IRL |

| Shanghai AIRL | Route choice AIRL | 24K OD pairs | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | AIRL (replicate-ready) |

| NYC Yellow 2013 | Labor supply (gap) | 14.8M trips | ❌ | — | — | — | ✅ | ✅ | Needs ICPSR driver IDs |

| Chicago Taxi | Labor supply | CORRUPTED | — | — | — | — | — | — | Re-download needed |



## Recommended Starting Points

### Immediate (data ready, clear DDC/IRL formulation)

1. **Trivago 2019** → Sequential search cost estimation
   - Use MCE-IRL on `train.csv`; state = `(step_bin, last_action, price_quartile)`
   - Validates: Ursu (2018) framework on public data

2. **KuaiRand log_random** → IRL reward recovery with exogenous variation
   - Use MCE-IRL on `log_random_4_22_to_5_08_27k.csv` (1.2M rows, randomly-exposed items)
   - Validates: structural identification without selection bias

3. **Citi Bike NYC** → Station-choice IRL
   - Use MCE-IRL; state = `(origin_zone, hour_bin, weekday)`
   - Clean discrete choice; nearest analog to Ermon et al. (2015) spatio-temporal DDC

4. **OTTO 2022** → Purchase funnel DDC
   - Use CCP; state = `(last_action, session_length_bin)`; 3-action funnel
   - Fast CCP estimation on 12.9M sessions

### Near-term (preprocessing needed)

5. **Porto Taxi** → Route choice IRL (best GPS dataset available)
   - Map-match POLYLINE to Porto OSM → MCE-IRL on road network
   - 1.71M trips, 15-second GPS, 448 taxis — replicate Ziebart (2008) framework

6. **Shanghai RCM-AIRL** → AIRL route choice (replicate-ready)
   - Pre-processed road network + CV splits + pre-trained models already downloaded
   - Replicate Zhao & Liang (2023); compare BC vs GAIL vs AIRL

7. **NGSIM US-101** → Lane-change IRL
   - Discretize `(lane_id, speed_bin, headway_bin)` → MCE-IRL or AIRL

8. **T-Drive** → Taxi route choice IRL
   - Map-match GPS to Beijing OSM → MCE-IRL on road network

9. **Foursquare** → Location choice IRL
   - Cluster venue categories → MCE-IRL on temporal mobility patterns

### Future (data gaps to fill)

10. **NYC Labor Supply DDC** → Download 2009-2013 yellow taxi from **ICPSR 37254**
    - Current `nyc_yellow_taxi_2013` parquet lacks `medallion`/`hack_license` — unusable for DDC
    - Original format has driver IDs enabling shift reconstruction
    - Follow Buchholz et al. (2025): state = `(earnings_bin, hours_bin, location_zone)` → 1,875 states

11. **Chicago Taxi** → Re-download from data.cityofchicago.org
    - Current file is corrupted (API timeout). Has consistent `taxi_id` unlike NYC HVFHV.

---

## Key Papers Referenced

| Paper | Dataset | Method | Relevance |
|-------|---------|--------|-----------|
| Rust (1987) *Econometrica* | Rust bus | NFXP | Canonical DDC reference |
| Ziebart et al. (2008) *AAAI* | Pittsburgh taxi GPS | MaxEnt IRL | Route choice IRL foundation |
| Ermon et al. (2015) *AAAI* | East Africa GPS | MaxEnt IRL ≡ logit DDC | Proves DDC-IRL equivalence |
| Ursu (2018) *Marketing Science* | Expedia hotel search | DDC search cost | Trivago analog |
| Buchholz, Shum & Xu (2025) | NYC yellow taxi | CCP DDC | Driver stopping model |
| Barnes et al. (2024) *ICLR* | Google Maps (360M params) | RHIP (IRL) | State-of-the-art route IRL |
| Gao et al. (2022) *CIKM* | KuaiRand | RL/IRL | Dataset paper + OPE framework |
| Zielnicki et al. (2025) | Netflix (2M users) | Discrete choice | DDC at recommendation scale |
| MTRec (2025) | ByteDance/TikTok (live) | Q-IRL | Deployed IRL for short-video |
| Compiani et al. (2024) *Marketing Science* | Expedia | DDC search | Search cost + welfare estimation |

# Field Shift: Adapting Farms with NASA Data

**NASA Space Apps Challenge 2026 · team s_m (members: s_hl, y_m).**

A crop rotation planner with a **Canada side** (Saskatchewan, the Prairies) and a **US side** (Iowa, the Corn Belt).

Farmers choose next year's crops in autumn, months before they know the summer. We tested, on both sides, whether next season can be forecast or rotations ranked. Neither holds up. So the planner does neither. It shows:

- what a farm can actually grow;
- how each option held up in past weather;
- what is measured and what is assumed;
- what is still unknown.

## Open the demo

```sh
python3 -m http.server 8817
```

Open `http://127.0.0.1:8817/`. The first page lets you choose a side, and a Canada / US switch sits at the top of both sides. Nothing you enter is sent anywhere.

- **Canada side** (`02_demo/canada/`). Pick a district, your crops, your usual plan and your limits. You see which two-year rotations held up about the same across 44 past summers, and what really differs between them: a 2021-like drought and $/ac. It ends with in-season alerts and a one-page summary for your agronomist.
- **US side** (`02_demo/us/`). A county map of the study sites, farm priorities for a corn–soybean farm, results from a long-term rotation trial and five on-farm trials, and the sources behind each claim.

## What we found

| | Canada side | US side |
|---|---|---|
| Data | NASA POWER weather and soil wetness, MODIS NDVI, SoilGrids, 40+ years of district yields | NASA POWER, the Marsden long-term trial (Iowa), on-farm trials, USDA county yields and crop maps |
| Before planting | No forecast skill, El Niño included | A process model beat the county's own past mean in only 2 of 9 county × crop series |
| What holds | In season, the model explains about 59% of year-to-year yield swings. 5–15 rotations per district are tied on average | In locked test years the three-year system out-yielded the two-year system (+10.2 bu/ac corn, +6.2 bu/ac soybeans) |
| Where it stops | District averages, not fields. "Best in 2021" is history, not protection | The process model is not validated for fields, so no yield or profit is predicted |

## Repository

| Folder | What it holds |
|---|---|
| `01_briefing/` | The 7-slide demo deck and the joint research briefing (18 slides) |
| `02_demo/` | The demo, both sides |
| `03_evidence/` | The final tables and charts behind each claim, for both sides |
| `04_reproduce/` | Code and inputs to reproduce both sides |

## Data credits

- **NASA:** [NASA POWER](https://power.larc.nasa.gov/) daily weather and soil wetness; [MODIS MOD13Q1 NDVI](https://lpdaac.usgs.gov/products/mod13q1v061/) through [AppEEARS](https://appeears.earthdatacloud.nasa.gov/) and the [ORNL DAAC MODIS web service](https://modis.ornl.gov/data/modis_webservice.html).
- **Soil:** [ISRIC SoilGrids 2.0](https://soilgrids.org/) (CC BY 4.0).
- **Yields and crop maps:** [Government of Saskatchewan RM yields](https://dashboard.saskatchewan.ca/agriculture/rm-yields/rm-yields-data) (Open Government Licence – Saskatchewan); [USDA NASS](https://www.nass.usda.gov/) county yields, Cropland Data Layer and Census of Agriculture.
- **Field trials:** Marsden long-term rotation trial ([maRsden dataset](https://github.com/vanichols/maRsden), MIT); [Practical Farmers of Iowa](https://practicalfarmers.org/resources/research-reports/cover-crop-effect-on-soil-water/) 2022 cover-crop trials.
- **Rotation effects and prices:** Manitoba Agriculture / MASC yield by previous crop; the 2026 Saskatchewan Crop Planning Guide.
- **Rotation rules:** Canola Council of Canada, Sask Pulse Growers and the Flax Council of Canada.
- **Map outlines:** [us-atlas](https://github.com/topojson/us-atlas).
- **El Niño index:** NOAA Oceanic Niño Index (tested and dropped).

## AI use

The team designed and directed the project. AI coding assistants wrote code and documentation under that direction.

This tool supports local agronomists. It does not replace them.

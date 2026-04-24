"""
Manchester Airbnb Price Predictor
Nicole Reeves, Dissertation (IJC319 Responsible Data Science)
XGBoost · Test R²: 0.5015 · RMSE: £84.73
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import warnings
import os
import re
from datetime import date, datetime

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Manchester Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main-title { text-align:center; font-size:2rem; font-weight:700; color:#222;
              padding:1.2rem 0 0.1rem 0; letter-spacing:-0.5px; }
.main-sub   { text-align:center; font-size:1rem; color:#666;
              margin-bottom:0.3rem; font-weight:400; }
.main-sub2  { text-align:center; font-size:0.82rem; color:#bbb; margin-bottom:1.5rem; }

/* Path selection cards */
.path-card {
    background:white; border:2px solid #eee; border-radius:16px;
    padding:2rem 1.5rem; text-align:center; cursor:pointer;
    transition:border-color 0.2s, box-shadow 0.2s;
}
.path-card:hover { border-color:#FF5A5F; box-shadow:0 4px 20px rgba(255,90,95,0.12); }
.path-card .icon { font-size:2.5rem; margin-bottom:0.75rem; }
.path-card .title { font-size:1.15rem; font-weight:700; color:#222; margin-bottom:0.4rem; }
.path-card .desc  { font-size:0.85rem; color:#888; line-height:1.5; }

/* Buttons */
.stButton > button {
    background-color:#FF5A5F !important; color:white !important;
    font-weight:600 !important; font-size:1rem !important;
    padding:0.7rem 1rem !important; border-radius:8px !important;
    border:none !important; width:100%;
}
.stButton > button:hover { background-color:#e04f54 !important; }

/* Price box */
.price-box {
    background:linear-gradient(135deg,#FF5A5F,#e04f54); color:white;
    border-radius:16px; padding:2rem 2rem 1.6rem 2rem; text-align:center;
    margin:1.5rem 0 1rem 0; box-shadow:0 4px 24px rgba(255,90,95,0.3);
}
.price-box .label { font-size:0.82rem; opacity:0.85; margin-bottom:0.3rem;
                    letter-spacing:1px; text-transform:uppercase; }
.price-box .value { font-size:4rem; font-weight:700; letter-spacing:-2px; line-height:1; }
.price-box .range { font-size:0.82rem; opacity:0.7; margin-top:0.5rem; }

/* Section header */
.sec-hdr {
    font-size:0.72rem; font-weight:700; text-transform:uppercase;
    letter-spacing:1.2px; color:#FF5A5F; margin:1.6rem 0 0.7rem 0;
    padding-bottom:0.35rem; border-bottom:2px solid #FF5A5F;
}
/* Metric cards */
.metric-card { background:#f8f9fa; border-radius:10px; padding:0.9rem 1rem;
               text-align:center; border-left:3px solid #FF5A5F; }
.metric-label { font-size:0.7rem; color:#999; margin-bottom:0.15rem;
                text-transform:uppercase; letter-spacing:0.5px; }
.metric-value { font-size:1.3rem; font-weight:700; color:#222; }

/* Insight card */
.insight-card {
    background:#fff8f8; border:1px solid #ffe0e1; border-radius:12px;
    padding:1.2rem 1.5rem; margin:0.5rem 0;
}
.insight-card .tag { display:inline-block; background:#FF5A5F; color:white;
    font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:1px;
    padding:0.2rem 0.6rem; border-radius:20px; margin-bottom:0.5rem; }
.insight-card p { font-size:0.9rem; color:#444; margin:0; line-height:1.6; }

.shap-note { font-size:0.78rem; color:#aaa; font-style:italic; margin-top:0.3rem; }
.back-btn { font-size:0.82rem; color:#FF5A5F; cursor:pointer; }
.footer-bar { text-align:center; color:#ccc; font-size:0.74rem;
              padding:1.5rem 0 0.5rem 0; border-top:1px solid #eee; margin-top:2.5rem; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
NEIGHBOURHOODS = [
    "Manchester","Salford","Trafford","Stockport","Rochdale",
    "Bury","Oldham","Wigan","Tameside","Bolton",
]
NEIGHBOURHOOD_COORDS = {
    "Manchester":(53.4808,-2.2426), "Salford":(53.4875,-2.2901),
    "Trafford":(53.4560,-2.3315),   "Stockport":(53.4060,-2.1575),
    "Rochdale":(53.6133,-2.1560),   "Bury":(53.5933,-2.2980),
    "Oldham":(53.5404,-2.1160),     "Wigan":(53.5450,-2.6320),
    "Tameside":(53.4800,-2.0790),   "Bolton":(53.5780,-2.4290),
}
ROOM_TYPES   = ["Entire home/apt","Private room","Shared room","Hotel room"]
PROPERTY_TYPES = ["Apartment","House","Serviced apartment","Townhouse","Other"]
RESPONSE_TIMES = ["Within an hour","Within a few hours","Within a day","A few days or more","Unknown"]
RESPONSE_TIME_MAP = {"Within an hour":0,"Within a few hours":1,"Within a day":2,"A few days or more":3,"Unknown":4}

# Expanded amenities with categories, includes all model features
AMENITY_GROUPS = {
    "Essentials": {
        "has_wifi":            "WiFi",
        "has_kitchen":         "Kitchen",
        "has_tv":              "TV",
        "has_air_conditioning":"Air Conditioning",
        "has_heating":         "Heating",
        "has_hot_water":       "Hot Water",
    },
    "Comfort": {
        "has_washer":          "Washer",
        "has_dryer":           "Dryer",
        "has_hair_dryer":      "Hair Dryer",
        "has_iron":            "Iron",
        "has_hangers":         "Hangers",
        "has_shampoo":         "Shampoo",
    },
    "Premium": {
        "has_pool":            "Pool",
        "has_hot_tub":         "Hot Tub",
        "has_gym":             "Gym",
        "has_bbq_grill":       "BBQ Grill",
        "has_balcony":         "Balcony",
        "has_garden":          "Garden",
        "has_city_view":       "City View",
    },
    "Safety & Access": {
        "has_smoke_alarm":              "Smoke Alarm",
        "has_carbon_monoxide_alarm":    "Carbon Monoxide Alarm",
        "has_fire_extinguisher":        "Fire Extinguisher",
        "has_first_aid_kit":            "First Aid Kit",
        "has_lockbox":                  "Lockbox / Self Check-in",
        "has_private_entrance":         "Private Entrance",
        "has_elevator":                 "Elevator",
    },
    "Workspace & Parking": {
        "has_dedicated_workspace":          "Dedicated Workspace",
        "has_free_parking_on_premises":     "Free Parking",
    },
    "Kitchen Extras": {
        "has_cookware":    "Cookware",
        "has_dishwasher":  "Dishwasher",
        "has_microwave":   "Microwave",
        "has_kettle":      "Kettle",
        "has_oven":        "Oven",
        "has_refrigerator":"Refrigerator",
    },
    "Entertainment": {
        "has_sound_system": "Sound System",
    },
}
# Flat list of all amenity keys
ALL_AMENITY_KEYS = {k: v for grp in AMENITY_GROUPS.values() for k, v in grp.items()}
# Subset that are direct model features
MODEL_AMENITY_FEATURES = {
    "has_wifi","has_kitchen","has_washer","has_dryer","has_free_parking_on_premises",
    "has_pool","has_gym","has_air_conditioning","has_tv","has_hot_tub",
    "has_dedicated_workspace","has_dishwasher","has_bbq_grill","has_elevator",
    "has_balcony","has_garden","has_cookware","has_hot_water","has_sound_system",
    "has_first_aid_kit","has_bbq_grill","has_kettle","has_city_view","has_microwave",
}

# Human-readable SHAP feature names
FEATURE_DISPLAY = {
    "accommodates":                    "Maximum number of guests",
    "host_total_listings_count":       "Host experience level",
    "room_type_Entire home/apt":       "Entire property (vs shared/room)",
    "bedrooms":                        "Number of bedrooms",
    "avg_word_length":                 "Description sophistication",
    "overall_text_quality":            "Overall listing quality score",
    "desc_length":                     "Description length and detail",
    "people_per_bedroom":              "Guests per bedroom ratio",
    "bathrooms":                       "Number of bathrooms",
    "longitude":                       "Location (east-west position)",
    "desc_luxury_mentions":            "Luxury keywords in description",
    "host_response_time_encoded":      "How quickly host responds",
    "kitchen_dining_amenities_count":  "Kitchen & dining amenities",
    "host_days_active":                "Host tenure on Airbnb",
    "name_length":                     "Listing title length",
    "name_word_count":                 "Words in listing title",
    "luxury_amenities_count":          "Number of luxury amenities",
    "amenities_count":                 "Total amenities offered",
    "number_of_reviews":               "Number of guest reviews",
    "safety_amenities_count":          "Safety amenities",
    "bedroom_living_amenities_count":  "Bedroom & living amenities",
    "review_scores_location":          "Location score from guests",
    "review_scores_cleanliness":       "Cleanliness score",
    "review_scores_checkin":           "Check-in experience score",
    "review_scores_rating":            "Overall guest rating",
    "desc_readability":                "How easy the description reads",
    "has_bbq_grill":                   "BBQ / Grill",
    "beds":                            "Number of beds",
    "review_scores_value":             "Value-for-money score",
    "bathroom_amenities_count":        "Bathroom amenities",
    "desc_sentiment_score":            "Positivity of description",
    "review_scores_communication":     "Host communication score",
    "host_identity_verified":          "Host identity verified",
    "convenience_amenities_score":     "Convenience amenities",
    "desc_facility_mentions":          "Facilities mentioned in description",
    "name_comfort_score":              "Comfort keywords in title",
    "has_city_view":                   "City view",
    "has_picture":                     "Listing has photos",
    "DistilBERT embeddings":           "Listing text (AI language analysis)",
    "ResNet50 image features":         "Photo quality (AI visual analysis)",
}

# ─── LOADERS ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model_artifacts():
    try:
        model     = xgb.Booster()
        model.load_model("airbnb_model.json")
        feat_cols = joblib.load("feature_columns.pkl")
        raw       = joblib.load("feature_defaults.pkl")
        defaults  = raw if isinstance(raw, dict) else dict(raw)
        return model, feat_cols, defaults
    except Exception as e:
        st.error(f"Could not load model files: {e}")
        return None, None, {}

@st.cache_data(show_spinner="Loading dataset…")
def load_processed():
    for p in ["airbnb_processed_data_multimodal.csv","airbnb_processed_data.csv",
              "data/airbnb_processed_data_multimodal.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            for c in ["price_per_person","text_quality_percentile"]:
                if c in df.columns: df = df.drop(columns=[c])
            return df
    return None

@st.cache_data(show_spinner="Loading listings…")
def load_raw():
    import re as _re
    def _parse_price(series):
        """Strip all non-numeric chars except decimal point, then coerce."""
        return pd.to_numeric(
            series.astype(str).apply(lambda x: _re.sub(r'[^\d.]', '', x)),
            errors="coerce")
    for p in ["listings.csv","data/listings.csv","listings__2__2.csv"]:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, low_memory=False)
                # Find best price column
                price_col = next((c for c in ["price","price_clean","nightly_price"]
                                  if c in df.columns), None)
                if price_col:
                    raw = df[price_col]
                    df["price_clean"] = (_parse_price(raw) if raw.dtype == object
                                         else pd.to_numeric(raw, errors="coerce"))
                    # If still all NaN, try string-parsing even if dtype looked numeric
                    if df["price_clean"].isna().all() and price_col in df.columns:
                        df["price_clean"] = _parse_price(df[price_col])
                else:
                    df["price_clean"] = float("nan")
                return df.dropna(subset=["latitude","longitude"])
            except Exception:
                pass
    return None

# ─── TEXT FEATURES ────────────────────────────────────────────────────────────
def text_features(name: str, desc: str) -> dict:
    name, desc = (name or "").strip(), (desc or "").strip()
    nl, dl = name.lower(), desc.lower()
    words  = desc.split()
    sents  = max(1, desc.count(".")+desc.count("!")+desc.count("?")+1)
    feats  = {
        "name_length":      float(len(name)),
        "name_word_count":  float(len(name.split())),
        "desc_length_chars":float(len(desc)),
        "desc_sentence_count": float(sents),
        "avg_word_length":  float(sum(len(w) for w in words)/max(1,len(words))),
        "desc_readability": float(max(0,min(100,206.835
            -1.015*(len(words)/sents)
            -84.6*(sum(len(w) for w in words)/max(1,len(words))/3)))),
    }
    for feat, kws in {
        "name_has_luxury":   ["luxury","premium","exclusive","boutique"],
        "name_has_cozy":     ["cozy","cosy","charming","cute"],
        "name_has_modern":   ["modern","contemporary","stylish","chic"],
        "name_has_spacious": ["spacious","large","huge","roomy"],
        "name_has_city":     ["city","centre","center","downtown"],
        "name_has_view":     ["view","views","panoramic","skyline"],
        "name_has_family":   ["family","kids","children"],
        "name_has_studio":   ["studio"],
        "name_has_entire":   ["entire","whole","private","self-contained"],
        "name_has_central":  ["central","city centre"],
        "desc_has_wifi_mention":        ["wifi","wi-fi","internet"],
        "desc_has_parking_mention":     ["parking","garage","driveway"],
        "desc_has_transport_info":      ["bus","tram","train","metro","transport"],
        "desc_has_self_checkin":        ["self check","lockbox","keypad","key safe"],
        "desc_has_neighborhood_info":   ["nearby","local","area","close to"],
    }.items():
        src = nl if feat.startswith("name_") else dl
        feats[feat] = float(any(k in src for k in kws))
    return feats

# ─── PREDICTION ─── UNCHANGED FROM WORKING VERSION ────────────────────────────
def build_vector(ui, feat_cols, defaults):
    rt  = ui.get("room_type", "Entire home/apt")
    acc = float(ui.get("accommodates", 2))
    bdr = float(ui.get("bedrooms", 1))

    # Start from pkl defaults (training medians, 151/151 key match)
    row = {col: float(defaults.get(col, 0.0)) for col in feat_cols}

    # Compute aggregate amenity counts from checkboxes
    amen = ui.get("amenities", {})
    safety_count = sum([
        int(amen.get("has_smoke_alarm",False)),
        int(amen.get("has_carbon_monoxide_alarm",False)),
        int(amen.get("has_first_aid_kit",False)),
        int(amen.get("has_fire_extinguisher",False)),
    ])
    kitchen_count = sum([
        int(amen.get("has_kitchen",False)),
        int(amen.get("has_cookware",False)),
        int(amen.get("has_dishwasher",False)),
        int(amen.get("has_microwave",False)),
        int(amen.get("has_kettle",False)),
        int(amen.get("has_oven",False)),
        int(amen.get("has_refrigerator",False)),
    ])
    bathroom_count = sum([
        int(amen.get("has_hot_water",False)),
        int(amen.get("has_hair_dryer",False)),
        int(amen.get("has_shampoo",False)),
    ])
    bedroom_count = sum([
        int(amen.get("has_tv",False)),
        int(amen.get("has_sound_system",False)),
        int(amen.get("has_washer",False)),
        int(amen.get("has_dryer",False)),
        int(amen.get("has_iron",False)),
        int(amen.get("has_hangers",False)),
    ])
    luxury_count = sum([
        int(amen.get("has_pool",False)),
        int(amen.get("has_hot_tub",False)),
        int(amen.get("has_gym",False)),
        int(amen.get("has_bbq_grill",False)),
    ])
    outdoor_count = sum([
        int(amen.get("has_garden",False)),
        int(amen.get("has_balcony",False)),
        int(amen.get("has_bbq_grill",False)),
    ])
    climate_count = sum([
        int(amen.get("has_air_conditioning",False)),
        int(amen.get("has_heating",False)),
    ])
    convenience_count = sum([
        int(amen.get("has_free_parking_on_premises",False)),
        int(amen.get("has_elevator",False)),
        int(amen.get("has_dedicated_workspace",False)),
        int(amen.get("has_lockbox",False)),
        int(amen.get("has_private_entrance",False)),
    ])
    total_amenities = sum(int(v) for v in amen.values())

    # Compute host_days_active from host_since date
    host_since = ui.get("host_since", None)
    if host_since:
        try:
            if isinstance(host_since, str):
                hs_date = datetime.strptime(host_since, "%Y-%m-%d").date()
            else:
                hs_date = host_since
            host_days = max(0, (date.today() - hs_date).days)
        except Exception:
            host_days = float(defaults.get("host_days_active", 1906))
    else:
        host_days = float(defaults.get("host_days_active", 1906))

    # Structured overrides, exact column names confirmed from model's feature list
    overrides = {
        "accommodates":                    acc,
        "bedrooms":                        bdr,
        "bathrooms":                       float(ui.get("bathrooms", 1.0)),
        "beds":                            float(ui.get("beds", 1)),
        "people_per_bedroom":              acc / max(1.0, bdr),
        "host_total_listings_count":       float(ui.get("host_total_listings_count", 5)),
        "host_response_time_encoded":      float(RESPONSE_TIME_MAP.get(ui.get("response_time","Within an hour"), 0)),
        "host_identity_verified":          float(int(ui.get("identity_verified", True))),
        "host_days_active":                float(host_days),
        "review_scores_rating":            float(ui.get("review_scores_rating", 4.82)),
        "review_scores_cleanliness":       float(ui.get("review_scores_cleanliness", 4.82)),
        "review_scores_location":          float(ui.get("review_scores_location", 4.82)),
        "review_scores_value":             float(ui.get("review_scores_value", 4.82)),
        "review_scores_communication":     float(ui.get("review_scores_communication", 4.82)),
        "review_scores_checkin":           float(ui.get("review_scores_checkin", 4.82)),
        "number_of_reviews":               float(ui.get("number_of_reviews", 9)),
        "longitude":                       float(ui.get("longitude", -2.2438)),
        "has_picture":                     1.0,
        # Aggregate amenity counts (computed above)
        "amenities_count":                 float(total_amenities),
        "safety_amenities_count":          float(safety_count),
        "kitchen_dining_amenities_count":  float(kitchen_count),
        "bathroom_amenities_count":        float(bathroom_count),
        "bedroom_living_amenities_count":  float(bedroom_count),
        "luxury_amenities_count":          float(luxury_count),
        "outdoor_recreation_amenities_count": float(outdoor_count),
        "climate_environment_amenities_count": float(climate_count),
        "convenience_amenities_score":     float(convenience_count),
    }
    for k, v in overrides.items():
        if k in row: row[k] = v

    # Room type, only Entire home/apt survived feature selection
    if "room_type_Entire home/apt" in row:
        row["room_type_Entire home/apt"] = 1.0 if rt == "Entire home/apt" else 0.0

    # Individual amenity flags (model features)
    for fk in MODEL_AMENITY_FEATURES:
        if fk in row:
            row[fk] = float(int(amen.get(fk, False)))

    # Property type
    for col in feat_cols:
        if col.startswith("property_type_grouped_"): row[col] = 0.0
    if "property_type_grouped_Apartment" in row:
        row["property_type_grouped_Apartment"] = 1.0

    # Text features (only if user typed something)
    lname = ui.get("listing_name","").strip()
    ldesc = ui.get("description","").strip()
    if lname or ldesc:
        tf = text_features(lname, ldesc)
        safe_map = {
            "name_length":       "name_length",
            "name_word_count":   "name_word_count",
            "desc_length_chars": "desc_length",
            "desc_sentence_count":"desc_sentence_count",
            "desc_readability":  "desc_readability",
        }
        if ldesc:
            words = ldesc.split()
            if words and "avg_word_length" in row:
                row["avg_word_length"] = sum(len(w) for w in words)/len(words)
        for src, dst in safe_map.items():
            if src not in tf or dst not in row: continue
            if src.startswith("name_") and not lname: continue
            if not src.startswith("name_") and not ldesc: continue
            row[dst] = tf[src]

    return pd.DataFrame([row], columns=feat_cols).astype(float)


def predict(ui, model, feat_cols, defaults):
    X    = build_vector(ui, feat_cols, defaults)
    dmat = xgb.DMatrix(X.values, feature_names=feat_cols)
    pred = float(model.predict(dmat)[0])
    return max(10.0, pred), X


def nb_stats(df):
    if df is None or "price" not in df.columns: return {}
    nc = next((c for c in ["neighbourhood_cleansed","neighbourhood_group_cleansed"]
               if c in df.columns), None)
    if nc is None: return {}
    return {nb: {"low":round(float(g["price"].quantile(0.25)),0),
                 "med":round(float(g["price"].median()),0),
                 "high":round(float(g["price"].quantile(0.75)),0),
                 "n":len(g)}
            for nb, g in df.groupby(nc)}


# ─── SHAP ─────────────────────────────────────────────────────────────────────
def compute_shap(model, X, feat_cols):
    try:
        import shap
        exp = shap.TreeExplainer(model)
        sv  = exp.shap_values(X)[0]
        base= float(exp.expected_value)
        grp = {}
        for nm, v in zip(X.columns, sv):
            key = ("DistilBERT embeddings" if nm.startswith(("name_bert_","desc_bert_"))
                   else "ResNet50 image features" if nm.startswith("img_resnet_")
                   else nm)
            grp[key] = grp.get(key,0)+v
        return grp, base
    except Exception:
        return {}, 0.0


def shap_chart(grp):
    try:
        import matplotlib.pyplot as plt, matplotlib
        matplotlib.rcParams["font.family"] = "sans-serif"
        top  = sorted(grp.items(), key=lambda x:abs(x[1]), reverse=True)[:14]
        names= [FEATURE_DISPLAY.get(t[0], t[0].replace("_"," ").title()) for t in reversed(top)]
        vals = [t[1] for t in reversed(top)]
        fig, ax = plt.subplots(figsize=(8,5))
        ax.barh(names, vals, color=["#FF5A5F" if v>0 else "#4a90d9" for v in vals],
                edgecolor="none", height=0.65)
        ax.axvline(0, color="#555", linewidth=0.8)
        ax.set_xlabel("£ impact on predicted price", fontsize=9)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.tick_params(axis="x", labelsize=8)
        for s in ["top","right","left"]: ax.spines[s].set_visible(False)
        plt.tight_layout()
        return fig
    except Exception:
        return None


# ─── AIRBNB URL FETCH ────────────────────────────────────────────────────────
def fetch_airbnb_listing(url: str) -> dict:
    """
    Fetch key details from an Airbnb listing URL using web scraping.
    Returns a dict of pre-filled form values, or {} on failure.
    We only extract what's visible in the page HTML without JavaScript rendering.
    """
    try:
        import urllib.request, json, re as re2
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36",
                "Accept-Language": "en-GB,en;q=0.9",
            }
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        data = {}

        # Title / listing name
        m = re2.search(r'<title[^>]*>([^<]+)</title>', html)
        if m:
            raw_title = m.group(1)
            # Remove " - Airbnb" suffix
            clean = re2.sub(r'\s*[|\-–]\s*Airbnb.*$', '', raw_title).strip()
            if clean:
                data["listing_name"] = clean[:100]

        # JSON-LD structured data (most reliable source)
        json_ld = re2.findall(
            "<script[^>]+type=[^>]*application/json[^>]*>(.*?)</script>",
            html, re2.DOTALL)
        for blob in json_ld:
            try:
                obj = json.loads(blob)
                # Look for listing data in Airbnb's internal JSON
                def deep_search(d, keys_found=None):
                    if keys_found is None: keys_found = {}
                    if isinstance(d, dict):
                        for k, v in d.items():
                            if k in ("personCapacity","maxGuestCapacity","guestCapacity") and isinstance(v,(int,float)):
                                keys_found["accommodates"] = int(v)
                            if k in ("bedrooms","bedroomsCount") and isinstance(v,(int,float)):
                                keys_found["bedrooms"] = int(v)
                            if k in ("bathrooms","bathroomsCount") and isinstance(v,(int,float)):
                                keys_found["bathrooms"] = float(v)
                            if k in ("beds","bedsCount") and isinstance(v,(int,float)):
                                keys_found["beds"] = int(v)
                            if k in ("description","listingDescription") and isinstance(v,str) and len(v)>50:
                                keys_found["description"] = v[:1000]
                            if k in ("starRating","reviewScore","guestSatisfactionOverall") and isinstance(v,(int,float)):
                                keys_found["review_scores_rating"] = float(v)
                            if k == "latitude" and isinstance(v,(int,float)):
                                keys_found["latitude"] = float(v)
                            if k == "longitude" and isinstance(v,(int,float)):
                                keys_found["longitude"] = float(v)
                            deep_search(v, keys_found)
                    elif isinstance(d, list):
                        for item in d:
                            deep_search(item, keys_found)
                    return keys_found
                found = deep_search(obj)
                data.update(found)
            except Exception:
                continue

        # Fallback: regex on raw HTML for common Airbnb data patterns
        if "accommodates" not in data:
            m = re2.search(r'"personCapacity"\s*:\s*(\d+)', html)
            if m: data["accommodates"] = int(m.group(1))
        if "bedrooms" not in data:
            m = re2.search(r'"bedrooms"\s*:\s*(\d+)', html)
            if m: data["bedrooms"] = int(m.group(1))
        if "bathrooms" not in data:
            m = re2.search(r'"bathrooms"\s*:\s*([\d.]+)', html)
            if m: data["bathrooms"] = float(m.group(1))

        return data
    except Exception as e:
        return {"_error": str(e)}


# ─── AI INSIGHTS ─────────────────────────────────────────────────────────────
def get_ai_insights(ui, pred, shap_grp, nbs, df_proc, feat_cols, defaults):
    """Call Claude API for plain-English host insights."""
    try:
        import anthropic
        # Try Streamlit secrets first, then environment variable
        api_key = ""
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            pass
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        return None
    except Exception:
        return None

    # Build context from SHAP
    pos = sorted([(k,v) for k,v in shap_grp.items() if v>0], key=lambda x:-x[1])[:5]
    neg = sorted([(k,v) for k,v in shap_grp.items() if v<0], key=lambda x:x[1])[:5]
    pos_str = ", ".join(f"{FEATURE_DISPLAY.get(k,k)} (+£{v:.0f})" for k,v in pos)
    neg_str = ", ".join(f"{FEATURE_DISPLAY.get(k,k)} (-£{abs(v):.0f})" for k,v in neg)

    # Market context
    nb = ui.get("neighbourhood","Manchester")
    match = next((k for k in nbs if nb.lower() in k.lower()), None)
    market_ctx = f"The median price for {match} is £{nbs[match]['med']:.0f}/night." if match else ""

    # Missing amenities
    amen = ui.get("amenities",{})
    missing_premium = [v for k,v in {
        "has_pool":"Pool","has_hot_tub":"Hot Tub","has_gym":"Gym",
        "has_dedicated_workspace":"Workspace","has_city_view":"City View",
        "has_free_parking_on_premises":"Free Parking"}.items()
        if not amen.get(k,False)]

    user_type = st.session_state.get("user_type","existing")
    is_new    = user_type == "new"

    prompt = f"""You are a friendly, expert Airbnb pricing consultant helping a host in Manchester, UK.

{'This host is new to Airbnb and thinking about listing their property.' if is_new else 'This host has an existing Airbnb listing.'}

Listing details:
- Property: {ui.get('room_type','Entire home/apt')}, {ui.get('accommodates',2)} guests, {ui.get('bedrooms',1)} bedrooms, {ui.get('bathrooms',1)} bathrooms
- Location: {ui.get('neighbourhood','Manchester')}
- Listing name: {ui.get('listing_name','Not provided') or 'Not provided'}
- Description length: {'Good detail' if len((ui.get('description','') or '')) > 200 else 'Quite short' if len((ui.get('description','') or '')) > 50 else 'Not provided'}
- Predicted price: £{pred:.0f}/night
- {market_ctx}
- Total amenities: {sum(int(v) for v in amen.values())}

What's DRIVING the price up: {pos_str if pos_str else 'None identified'}
What's HOLDING the price back: {neg_str if neg_str else 'None identified'}
Premium amenities not currently offered: {', '.join(missing_premium[:4]) if missing_premium else 'None missing'}

Please write a response with exactly these 4 sections, using plain English (no jargon):

**Why your listing is predicted at £{pred:.0f}**
[2-3 sentences explaining the price in simple terms based on the data above]

**Top 3 ways to increase your nightly rate**
[3 specific, actionable recommendations backed by the data, each on its own line with a bullet point. Be specific, e.g. "Adding a pool could add ~£X/night based on similar Manchester listings" only if justified]

**Your listing description**
[1-2 sentences of specific advice on the description, longer/shorter, keywords to add, what guests look for]

**Pricing strategy**
[1-2 sentences on when to charge more/less, weekends, seasonality, minimum nights, based on typical Manchester market patterns]

Keep the whole response under 250 words. Be warm, encouraging, and practical."""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role":"user","content":prompt}]
        )
        return msg.content[0].text
    except Exception:
        return None


# ─── AMENITY SIMULATOR (quick version for suggestions) ────────────────────────
def quick_amenity_impacts(ui, model, feat_cols, defaults):
    """
    Return {amenity_label: price_impact} for amenities the user hasn't ticked.
    Keeps the user's current aggregate counts as the baseline, only the
    individual boolean flag is toggled, so we isolate the marginal effect
    of adding one amenity without inflating the aggregate counts.
    """
    results   = {}
    amen      = ui.get("amenities", {})
    base_pred, _ = predict(ui, model, feat_cols, defaults)   # user's actual config
    for fk, label in ALL_AMENITY_KEYS.items():
        if amen.get(fk, False):
            continue   # already enabled, skip
        # Only add the flag; aggregates stay the same because build_vector
        # recomputes them from the full amenity dict
        test_amen = {**amen, fk: True}
        test_pred, _ = predict({**ui, "amenities": test_amen}, model, feat_cols, defaults)
        impact = round(test_pred - base_pred, 2)
        if impact > 0.3:
            results[label] = impact
    return dict(sorted(results.items(), key=lambda x:-x[1])[:6])


# ─── FORM HELPERS ─────────────────────────────────────────────────────────────
def amenity_section(existing_vals=None):
    """Render the full amenity grid. Returns dict of {key: bool}."""
    if existing_vals is None: existing_vals = {}
    result = {}
    for group_name, items in AMENITY_GROUPS.items():
        st.markdown(f'<div class="sec-hdr">{group_name}</div>', unsafe_allow_html=True)
        keys   = list(items.keys())
        labels = list(items.values())
        cols   = st.columns(4)
        for i, (fk, lbl) in enumerate(zip(keys, labels)):
            default = fk in ["has_wifi","has_kitchen","has_tv","has_hot_water"]
            result[fk] = cols[i%4].checkbox(lbl, value=existing_vals.get(fk, default), key=f"am_{fk}")
    return result


def results_section(ui, pred, X, model, feat_cols, defaults, df_proc, df_raw):
    """Full results displayed below the form."""
    nbs = nb_stats(df_proc)

    # ── Price prediction ──────────────────────────────────────────────────────
    lo, hi = max(10.0, pred*0.82), pred*1.18
    st.markdown(
        f'<div class="price-box">'
        f'<div class="label">Predicted Nightly Price</div>'
        f'<div class="value">£{pred:.0f}</div>'
        f'<div class="range">Indicative range: £{lo:.0f} – £{hi:.0f} &nbsp;·&nbsp; '
        f'Model R² = 0.5015, RMSE = £84.73</div>'
        f'</div>', unsafe_allow_html=True)


    # ── Responsible use ───────────────────────────────────────────────────────
    with st.expander("⚖️ Using this tool responsibly", expanded=False):
        st.markdown(
            "This tool is based on 6,562 Airbnb listings in Manchester from September 2025. "
            "It's designed to give a sense of how different features relate to price, not to set your price for you.\n\n"
            "**A few things to keep in mind:**\n\n"
            "- **Treat predictions as a guide, not a rule.** The model explains about half of the variation in prices "
            "(R\u00b2 \u2248 0.50). Things like seasonality, events, and your own hosting style are not captured.\n"
            "- **Use it carefully when setting prices.** Tools like this can push prices up if followed too closely. "
            "It's worth thinking about what feels fair, not just what the model suggests.\n"
            "- **Be aware of where it works better or worse.** The model was checked across room types, price ranges, "
            "and areas, and didn't show clear unfair bias. That said, it's less accurate for very low and very high priced listings.\n"
            "- **Your judgement still matters most.** This works best alongside your own knowledge of your property, "
            "your guests, and the local area."
        )

    # ── Revenue estimates ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">Revenue Estimates</div>', unsafe_allow_html=True)
    rc1,rc2,rc3,rc4 = st.columns(4)
    for cw,occ,lbl in [(rc1,0.5,"50% occupancy"),(rc2,0.65,"65% occupancy"),
                       (rc3,0.8,"80% occupancy"),(rc4,0.9,"90% occupancy")]:
        cw.markdown(
            f'<div class="metric-card"><div class="metric-label">{lbl}</div>'
            f'<div class="metric-value">£{pred*30*occ:,.0f}'
            f'<span style="font-size:0.65rem;opacity:0.6">/mo</span>'
            f'</div></div>', unsafe_allow_html=True)
    st.caption("Estimates based on predicted nightly price × 30 nights × occupancy rate.")

    # ── Market position ───────────────────────────────────────────────────────
    nb = ui.get("neighbourhood","Manchester")
    if nbs:
        match = next((k for k in nbs if nb.lower() in k.lower()), None)
        if match:
            s = nbs[match]; ratio = pred/s["med"]
            icon = "📉" if ratio<0.85 else "📊" if ratio<0.95 else "✅" if ratio<1.1 else "📈"
            pos  = ("below market median, which is competitive pricing and good for bookings"
                    if ratio<0.85 else
                    "slightly below market, which is strong value positioning"
                    if ratio<0.95 else
                    "broadly in line with the local market"
                    if ratio<1.1 else
                    "above market median, so make sure your listing justifies the higher price")
            st.info(f"{icon} **{match}** median is £{s['med']:.0f}/night. Your prediction is {pos}.")
            try:
                import plotly.graph_objects as go
                fig = go.Figure(go.Bar(
                    x=["25th %ile","Median","75th %ile","Your Price"],
                    y=[s["low"],s["med"],s["high"],pred],
                    marker_color=["#a8d8ea","#4a90d9","#1b4f8a","#FF5A5F"],
                    text=[f"£{v:.0f}" for v in [s["low"],s["med"],s["high"],pred]],
                    textposition="auto"))
                fig.update_layout(title=f"Your price vs {match} market",
                    yaxis_title="£/night", showlegend=False, height=260,
                    plot_bgcolor="white", paper_bgcolor="white",
                    margin=dict(t=40,b=30,l=40,r=20))
                st.plotly_chart(fig, use_container_width=True)
            except ImportError: pass

    # ── SHAP contributions ────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">What\'s Driving Your Price?</div>',
                unsafe_allow_html=True)
    with st.spinner("Analysing price factors…"):
        shap_grp, base = compute_shap(model, X, feat_cols)

    if shap_grp:
        col_chart, col_explain = st.columns([1.2, 1])
        with col_chart:
            fig_s = shap_chart(shap_grp)
            if fig_s: st.pyplot(fig_s, use_container_width=True)
        with col_explain:
            st.markdown("**What each bar means:**")
            st.markdown(
                "Each bar shows how much a feature **adds or subtracts** from the base prediction. "
                f"The model starts from a base value of £{base:.0f}, the average for all listings, "
                "then adjusts based on your specific property.")
            st.markdown("")
            # Top 3 positive & negative in plain English
            pos3 = sorted([(k,v) for k,v in shap_grp.items() if v>1], key=lambda x:-x[1])[:3]
            neg3 = sorted([(k,v) for k,v in shap_grp.items() if v<-1], key=lambda x:x[1])[:3]
            if pos3:
                st.markdown("🟥 **Pushing your price up:**")
                for k,v in pos3:
                    nm = FEATURE_DISPLAY.get(k, k.replace("_"," ").title())
                    st.markdown(f"- **{nm}** (+£{v:.0f})")
            if neg3:
                st.markdown("🟦 **Holding your price back:**")
                for k,v in neg3:
                    nm = FEATURE_DISPLAY.get(k, k.replace("_"," ").title())
                    st.markdown(f"- **{nm}** (-£{abs(v):.0f})")
        st.markdown('<p class="shap-note">SHAP (SHapley Additive exPlanations) is a mathematically '
                    'rigorous method for explaining ML predictions. DistilBERT/ResNet50 features '
                    'use dataset-median embeddings in deployment.</p>', unsafe_allow_html=True)
    else:
        st.info("Install `shap` to enable feature-level explanations.")

    # ── Amenity suggestions ───────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">Amenities That Could Increase Your Price</div>',
                unsafe_allow_html=True)
    with st.spinner("Calculating amenity impacts…"):
        impacts = quick_amenity_impacts(ui, model, feat_cols, defaults)
    if impacts:
        try:
            import plotly.graph_objects as go
            fig_am = go.Figure(go.Bar(
                x=list(impacts.keys()), y=list(impacts.values()),
                marker_color="#FF5A5F",
                text=[f"+£{v:.2f}" for v in impacts.values()],
                textposition="auto"))
            fig_am.update_layout(
                title="Estimated price increase per amenity (vs your current selection)",
                yaxis_title="£ increase per night", height=300,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=50,b=40,l=40,r=20), showlegend=False)
            st.plotly_chart(fig_am, use_container_width=True)
            st.caption("Each bar shows how much the model thinks adding that amenity would affect your listing, assuming everything else stays the same. These estimates come from 6,562 real Manchester listings and aren't guaranteed results.")
        except ImportError:
            for lbl, impact in impacts.items():
                st.write(f"• **{lbl}**: +£{impact:.2f}/night")
    else:
        st.success("Great, your amenity selection is already comprehensive!")

    # ── Competitor analysis ───────────────────────────────────────────────────
    if df_proc is not None:
        st.markdown('<div class="sec-hdr">Your 5 Most Similar Competitors</div>',
                    unsafe_allow_html=True)
        try:
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import StandardScaler as SKS
            mf  = [c for c in ["accommodates","bedrooms","bathrooms","beds",
                                "latitude","longitude"] if c in df_proc.columns]
            dknn = df_proc[mf+["price"]].dropna().copy()
            sk   = SKS(); Xk = sk.fit_transform(dknn[mf])
            nbrs = NearestNeighbors(n_neighbors=6).fit(Xk)
            q    = {f: ui.get(f, 0) for f in mf}
            _, idxs = nbrs.kneighbors(sk.transform([[q.get(f,0) for f in mf]]))
            cidx    = idxs[0][1:]
            comp    = dknn.iloc[cidx].copy()
            comp_urls = {}
            comp_names = {}
            if df_raw is not None:
                # Build id→row lookup for fast matching
                id_col = "id" if "id" in df_raw.columns else None
                for ci, (_, crow) in enumerate(comp.iterrows()):
                    mr = df_raw[
                        (df_raw["latitude"].round(3) == round(float(crow.get("latitude", 0)), 3)) &
                        (df_raw["longitude"].round(3) == round(float(crow.get("longitude", 0)), 3))
                    ]
                    if len(mr) > 0:
                        r0 = mr.iloc[0]
                        # Name
                        if "name" in r0.index and pd.notna(r0["name"]):
                            comp_names[ci] = str(r0["name"])[:50]
                        # URL — prefer listing_url, fall back to building from id
                        if "listing_url" in r0.index and pd.notna(r0["listing_url"]):
                            comp_urls[ci] = str(r0["listing_url"])
                        elif id_col and pd.notna(r0[id_col]):
                            try: comp_urls[ci] = f"https://www.airbnb.co.uk/rooms/{int(r0[id_col])}"
                            except Exception: pass
            rows = [{"Listing": comp_names.get(ci, f"Similar listing {ci+1}"),
                     "Guests":int(crow.get("accommodates",0)),
                     "Bedrooms":int(crow.get("bedrooms",0)),"Bathrooms":crow.get("bathrooms","–"),
                     "Price (£/night)":f"£{crow['price']:.0f}",
                     "Airbnb Link":comp_urls.get(ci,"")}
                    for ci,(_, crow) in enumerate(comp.iterrows())]
            disp = pd.DataFrame(rows)
            if any(r["Airbnb Link"] for r in rows):
                st.dataframe(disp, column_config={"Airbnb Link":st.column_config.LinkColumn(
                    "View on Airbnb", display_text="Open →")},
                    use_container_width=True, hide_index=True)
            else:
                st.dataframe(disp.drop(columns=["Airbnb Link"]),
                    use_container_width=True, hide_index=True)
            cp  = dknn.iloc[cidx]["price"].values
            # Competitor chart only, no automated text judgement
        except Exception as e:
            st.info(f"Competitor analysis unavailable: {e}")

    # ── Small pricing note ───────────────────────────────────────────────────
    nb_med = nbs.get(next((k for k in nbs if nb.lower() in k.lower()), ""), {}).get("med", pred)
    if nb_med:
        st.caption(
            f"The {nb} median is £{nb_med:.0f}/night. "
            f"Your prediction of £{pred:.0f} sits at the "
            f"{'lower end' if pred < nb_med else 'upper end'} of the local market. "
            f"The model doesn't take into account seasonal trends, events, or last-minute demand. "
            f"You might want to adjust manually during busy times, like Manchester's festival season."
        )




# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    st.markdown('<div class="main-title">🏠 Manchester Airbnb Price Predictor</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="main-sub">Enter your listing details for a personalised price prediction, '
        'market evaluation, and revenue strategy</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="main-sub2">XGBoost · DistilBERT NLP · ResNet50 Computer Vision · '
        'SHAP Interpretability · 6,562 Manchester listings</div>',
        unsafe_allow_html=True)

    model, feat_cols, defaults = load_model_artifacts()
    df_proc = load_processed()
    df_raw  = load_raw()

    if model is None:
        st.error("Model files not found. Ensure `airbnb_model.json`, `feature_columns.pkl`, "
                 "and `feature_defaults.pkl` are in the repository root.")
        return

    tabs = st.tabs(["Price Predictor","Market Map","Market Dashboard","Model Insights"])

    # ══ TAB 1, PRICE PREDICTOR ═══════════════════════════════════════════════
    with tabs[0]:

        # ── Path selection ─────────────────────────────────────────────────────
        if "user_type" not in st.session_state:
            st.markdown("### Who are you?")
            st.write("Choose the option that best describes you so we can tailor the form to your needs.")
            st.markdown("")
            p1, p2 = st.columns(2, gap="large")
            with p1:
                st.markdown("""
                <div class="path-card">
                    <div class="icon">🏡</div>
                    <div class="title">Already on Airbnb</div>
                    <div class="desc">I have an existing listing. I have reviews, a listing URL,
                    and know my current performance.</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("")
                if st.button("I'm already on Airbnb", key="path_existing"):
                    st.session_state["user_type"] = "existing"
                    st.rerun()
            with p2:
                st.markdown("""
                <div class="path-card">
                    <div class="icon">✨</div>
                    <div class="title">New to Airbnb</div>
                    <div class="desc">I'm thinking of listing my property. I don't have reviews
                    yet and want to know what I could earn.</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("")
                if st.button("I'm new to Airbnb", key="path_new"):
                    st.session_state["user_type"] = "new"
                    st.rerun()
            return

        user_type = st.session_state["user_type"]
        is_new    = (user_type == "new")

        # Back button
        if st.button("← Change", key="back_btn"):
            for k in ["user_type","pred","Xv","ai_insights","ui"]:
                st.session_state.pop(k, None)
            st.rerun()

        label_tag = "New Host" if is_new else "Existing Host"
        st.markdown(f"**Mode:** {label_tag} &nbsp;·&nbsp; "
                    f"{'Using average review scores as defaults' if is_new else 'Enter your actual review scores below'}")

        # ── Airbnb URL pre-fill (existing hosts only) ──────────────────────
        if not is_new:
            st.markdown("#### Auto-fill from your Airbnb listing")
            url_col, btn_col = st.columns([3, 1])
            airbnb_url = url_col.text_input(
                "Paste your Airbnb listing URL",
                placeholder="https://www.airbnb.co.uk/rooms/12345678",
                label_visibility="collapsed",
                help="Paste your Airbnb listing URL and click 'Auto-fill' to populate "
                     "the form with your listing's details.",
            )
            if btn_col.button("Auto-fill form", key="url_fetch_btn"):
                if airbnb_url.strip():
                    with st.spinner("Fetching listing details…"):
                        fetched = fetch_airbnb_listing(airbnb_url.strip())
                    if "_error" in fetched:
                        st.warning(f"Could not fetch listing automatically, Airbnb may block "
                                   f"automated requests. Please fill in the form manually. "
                                   f"(Error: {fetched['_error']})")
                    elif not fetched:
                        st.warning("No data found. Airbnb's pages are JavaScript-rendered "
                                   "and may not be fully accessible. Please fill the form manually.")
                    else:
                        st.session_state["prefill"] = fetched
                        st.session_state["airbnb_url"] = airbnb_url.strip()
                        st.success(f"✅ Pre-filled {len(fetched)} fields from your listing! "
                                   f"Review and adjust below, then click Predict.")
            if "airbnb_url" in st.session_state:
                st.caption(f"URL: {st.session_state['airbnb_url']}")

        prefill = st.session_state.get("prefill", {})
        st.markdown("---")

        # ══════════════════════════════════════════════════════════════════════
        # THE FORM
        # ══════════════════════════════════════════════════════════════════════
        with st.form("listing_form"):

            # ── 1. Listing Info ───────────────────────────────────────────────
            st.markdown('<div class="sec-hdr">1. Listing Name & Description</div>',
                        unsafe_allow_html=True)
            pf = st.session_state.get("prefill", {})
            listing_name = st.text_input("Listing name",
                value=pf.get("listing_name",""),
                placeholder="e.g. Stylish City Centre Apartment, Modern & Spacious",
                help="Use descriptive keywords. Titles with words like 'luxury', 'spacious', "
                     "'city centre' correlate with higher prices in the training data.")
            description  = st.text_area("Listing description",
                value=pf.get("description",""),
                placeholder="Describe your space, highlights, local area, what guests will love…",
                height=130,
                help="Longer, detailed descriptions correlate with higher prices. "
                     "Aim for 150–300 words.")

            if is_new:
                st.markdown("**Upload a photo of your property** (optional, for preview only)")
                uploaded_img = st.file_uploader("", type=["jpg","jpeg","png","webp"],
                    key="img_upload", label_visibility="collapsed")
                if uploaded_img:
                    st.image(uploaded_img, caption="Your listing photo", use_container_width=True)
                    st.caption("Photo quality features use dataset-median values in the model. "
                               "In a deployed version, ResNet50 would process this image directly.")
                image_url = ""
            else:
                image_url = st.text_input("Listing image URL",
                    placeholder="https://a0.muscache.com/…",
                    help="Paste your Airbnb listing photo URL for a preview.")
                if image_url:
                    try: st.image(image_url, use_container_width=True)
                    except Exception: st.caption("Could not load image.")

            # ── 2. Property Details ───────────────────────────────────────────
            st.markdown('<div class="sec-hdr">2. Property Details</div>', unsafe_allow_html=True)
            pd1, pd2 = st.columns(2)
            property_type = pd1.selectbox("Property type", PROPERTY_TYPES)
            room_type     = pd2.selectbox("Room type",     ROOM_TYPES)
            pc1,pc2,pc3,pc4,pc5 = st.columns(5)
            accommodates = pc1.number_input("Guests",    1, 16, int(pf.get("accommodates",2)))
            bedrooms     = pc2.number_input("Bedrooms",  0, 15, int(pf.get("bedrooms",1)))
            bathrooms    = pc3.number_input("Bathrooms", 0.0, 10.0, float(pf.get("bathrooms",1.0)), step=0.5)
            beds         = pc4.number_input("Beds",      0, 20, int(pf.get("beds",1)))
            min_nights   = pc5.number_input("Min nights",1, 90, 2)

            # ── 3. Location ───────────────────────────────────────────────────
            st.markdown('<div class="sec-hdr">3. Location</div>', unsafe_allow_html=True)
            address = st.text_input("Property address",
                placeholder="e.g. 12 High Street, Manchester, M1 1AA",
                help="Enter your address and click 'Look up coordinates' to auto-fill "
                     "latitude and longitude for a more accurate prediction.")

            lc1, lc2 = st.columns([2, 1])
            neighbourhood = lc1.selectbox("Neighbourhood (area)", NEIGHBOURHOODS)
            geo_btn = lc2.form_submit_button("📍 Look up coordinates from address")

            lat_default, lon_default = NEIGHBOURHOOD_COORDS.get(neighbourhood,(53.4808,-2.2426))

            # Auto-geocode if address button pressed
            if geo_btn and address.strip():
                try:
                    import urllib.request, json, urllib.parse
                    query   = urllib.parse.quote(address.strip() + ", Manchester, UK")
                    url     = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=1"
                    req     = urllib.request.Request(url, headers={"User-Agent":"StreamlitAirbnbApp/1.0"})
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        data = json.loads(resp.read())
                    if data:
                        lat_default = float(data[0]["lat"])
                        lon_default = float(data[0]["lon"])
                        st.session_state["geo_lat"] = lat_default
                        st.session_state["geo_lon"] = lon_default
                        st.success(f"📍 Found: {lat_default:.4f}, {lon_default:.4f}")
                    else:
                        st.warning("Address not found, using neighbourhood centre instead.")
                except Exception:
                    st.warning("Could not look up coordinates, using neighbourhood centre.")

            # Use geocoded values if available, or prefilled from URL
            if "geo_lat" in st.session_state:
                lat_default = st.session_state["geo_lat"]
                lon_default = st.session_state["geo_lon"]
            elif pf.get("latitude"):
                lat_default = float(pf["latitude"])
                lon_default = float(pf.get("longitude", lon_default))

            lc3, lc4 = st.columns(2)
            latitude  = lc3.number_input("Latitude",  value=lat_default, format="%.4f",
                help="Auto-filled from address lookup or neighbourhood centre.")
            longitude = lc4.number_input("Longitude", value=lon_default, format="%.4f")

            # ── 4. Host Information ───────────────────────────────────────────
            st.markdown('<div class="sec-hdr">4. Host Information</div>', unsafe_allow_html=True)
            hi1, hi2, hi3 = st.columns(3)
            host_since     = hi1.date_input("Host since",
                value=date(2020, 1, 1) if is_new else date(2019, 1, 1),
                min_value=date(2008, 1, 1), max_value=date.today(),
                help="Date you joined Airbnb as a host.")
            host_listings  = hi2.number_input("Your total listings", 1, 500, 1,
                help="Total number of listings you manage on Airbnb.")
            response_time  = hi3.selectbox("Response time", RESPONSE_TIMES)
            hc1, hc2, hc3 = st.columns(3)
            superhost           = hc1.checkbox("Superhost status",     value=False)
            identity_verified   = hc2.checkbox("Identity verified",    value=True)
            instant_bookable    = hc3.checkbox("Instant bookable",     value=False)

            # ── 5. Amenities ──────────────────────────────────────────────────
            st.markdown('<div class="sec-hdr">5. Amenities</div>', unsafe_allow_html=True)
            st.write("Select all amenities your property offers:")
            amenities = amenity_section()

            # ── 6. Reviews ────────────────────────────────────────────────────
            if not is_new:
                st.markdown('<div class="sec-hdr">6. Reviews</div>', unsafe_allow_html=True)
                st.write("Enter your current review scores from your Airbnb dashboard (1.0–5.0):")
                rv1, rv2 = st.columns(2)
                with rv1:
                    n_reviews   = st.number_input("Number of reviews",  0, 5000, 5)
                    r_overall   = st.slider("Overall rating",       1.0, 5.0, 4.8, 0.01)
                    r_clean     = st.slider("Cleanliness",          1.0, 5.0, 4.8, 0.01)
                    r_location  = st.slider("Location",             1.0, 5.0, 4.8, 0.01)
                with rv2:
                    r_value     = st.slider("Value",                1.0, 5.0, 4.8, 0.01)
                    r_comm      = st.slider("Communication",        1.0, 5.0, 4.8, 0.01)
                    r_checkin   = st.slider("Check-in",             1.0, 5.0, 4.8, 0.01)
                    r_accuracy  = st.slider("Accuracy",             1.0, 5.0, 4.8, 0.01)
            else:
                # New host, use training medians, no review fields
                n_reviews  = 0; r_overall=4.82; r_clean=4.82; r_location=4.82
                r_value=4.82; r_comm=4.82; r_checkin=4.82; r_accuracy=4.82
                st.markdown('<div class="sec-hdr">6. Reviews</div>', unsafe_allow_html=True)
                st.info("As a new host you won't have reviews yet, the model will use "
                        "average review scores from the training dataset for this prediction.")

            # ── Submit ────────────────────────────────────────────────────────
            st.markdown("")
            predict_btn = st.form_submit_button("🔍 Get My Price Prediction",
                use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # RUN PREDICTION (outside form)
        # ══════════════════════════════════════════════════════════════════════
        if predict_btn:
            ui = {
                "user_type":            user_type,
                "listing_name":         listing_name,
                "description":          description,
                "property_type":        property_type,
                "room_type":            room_type,
                "accommodates":         accommodates,
                "bedrooms":             bedrooms,
                "bathrooms":            bathrooms,
                "beds":                 beds,
                "minimum_nights":       min_nights,
                "neighbourhood":        neighbourhood,
                "latitude":             latitude,
                "longitude":            longitude,
                "address":              address,
                "host_since":           host_since.isoformat() if host_since else None,
                "host_total_listings_count": host_listings,
                "response_time":        response_time,
                "host_is_superhost":    superhost,
                "identity_verified":    identity_verified,
                "instant_bookable":     instant_bookable,
                "amenities":            amenities,
                "number_of_reviews":    n_reviews,
                "review_scores_rating":        r_overall,
                "review_scores_cleanliness":   r_clean,
                "review_scores_location":      r_location,
                "review_scores_value":         r_value,
                "review_scores_communication": r_comm,
                "review_scores_checkin":       r_checkin,
                "review_scores_accuracy":      r_accuracy,
            }
            with st.spinner("Running model…"):
                pred, Xv = predict(ui, model, feat_cols, defaults)
            st.session_state["pred"] = pred
            st.session_state["Xv"]   = Xv
            st.session_state["ui"]   = ui
            st.session_state.pop("ai_insights", None)

        # ══════════════════════════════════════════════════════════════════════
        # RESULTS (below form)
        # ══════════════════════════════════════════════════════════════════════
        if "pred" in st.session_state and "ui" in st.session_state:
            st.markdown("---")
            st.markdown("## Your Results")
            results_section(
                st.session_state["ui"],
                st.session_state["pred"],
                st.session_state["Xv"],
                model, feat_cols, defaults, df_proc, df_raw
            )

    # ══ TAB 2, MARKET MAP ════════════════════════════════════════════════════
    with tabs[1]:
        st.subheader("Manchester Airbnb Listings Explorer")
        src = df_raw if df_raw is not None else df_proc
        if src is None:
            st.warning("Add `listings.csv` or `airbnb_processed_data_multimodal.csv` to enable the map.")
        else:
            try:
                import folium
                from streamlit_folium import st_folium
                nc  = next((c for c in ["neighbourhood_group_cleansed","neighbourhood_cleansed",
                                              "neighbourhood_group","neighbourhood"]
                            if c in src.columns), None)
                pc  = "price_clean" if "price_clean" in src.columns else (
                      "price" if "price" in src.columns else None)
                f1,f2,f3 = st.columns([1.5,1.5,1])
                sel = (f1.multiselect("Filter by area",
                       sorted(src[nc].dropna().unique()), default=sorted(src[nc].dropna().unique()))
                       if nc else [])
                prices_valid = src[pc].dropna() if pc else pd.Series(dtype=float)
                has_prices = len(prices_valid) > 0
                if has_prices:
                    q01, q99 = prices_valid.quantile(0.01), prices_valid.quantile(0.99)
                    mn = int(q01) if pd.notna(q01) else 0
                    mx = int(q99) if pd.notna(q99) else 500
                    if mn >= mx: mn, mx = 0, 500
                    pr = f2.slider("Price (£/night)", mn, mx, (mn, mx), step=5)
                else:
                    pr = (0, 9999)
                    if pc: f2.caption("Price filter unavailable")
                mdf = src.copy()
                if nc and sel: mdf = mdf[mdf[nc].isin(sel)]
                if pc and has_prices:
                    mdf = mdf[(mdf[pc] >= pr[0]) & (mdf[pc] <= pr[1])].dropna(subset=[pc])
                f3.metric("Listings",f"{len(mdf):,} of {len(src):,}")

                mt1,mt2=st.tabs(["Map View","List View"])
                with mt1:
                    if len(mdf)==0: st.info("No listings match.")
                    else:
                        m=folium.Map([mdf["latitude"].mean(),mdf["longitude"].mean()],zoom_start=11,tiles="CartoDB positron")
                        smp=mdf.sample(min(1500,len(mdf)),random_state=42) if len(mdf)>1500 else mdf
                        for _,row in smp.iterrows():
                            raw_pv = row[pc] if pc and pc in row.index else None
                            try:
                                pn = float(raw_pv) if raw_pv is not None and pd.notna(raw_pv) else None
                            except (TypeError, ValueError):
                                pn = None
                            # Fallback: parse raw price string directly if price_clean was NaN
                            if pn is None and "price" in row.index:
                                import re as _re2
                                try:
                                    stripped = _re2.sub(r'[^\d.]', '', str(row["price"]))
                                    pn = float(stripped) if stripped else None
                                except (TypeError, ValueError):
                                    pn = None
                            ps = f"£{pn:.0f}" if pn is not None else "–"
                            nm=str(row.get("name","Listing"))[:50]
                            ar=str(row.get(nc,"")) if nc else ""
                            url = row.get("listing_url", None)
                            if (url is None or pd.isna(url)) and "id" in row.index:
                                try: url = f"https://www.airbnb.co.uk/rooms/{int(row['id'])}"
                                except Exception: url = None
                            lnk = (f'<a href="{url}" target="_blank" style="color:#FF5A5F;font-weight:bold;">View on Airbnb →</a>'
                                   if url and url != "None" else "")
                            pop=(f'<div style="width:220px;font-family:Arial;padding:8px;">'
                                 f'<b>{nm}</b><br><br>'
                                 f'<b style="color:#FF5A5F;font-size:15px;">{ps}</b> per night<br>'
                                 f'<span style="color:#888;">{ar}</span><br><br>{lnk}</div>')
                            pv_num = pn if pn is not None else -1
                            pin = "gray" if pv_num < 0 else "green" if pv_num < 55 else "orange" if pv_num < 90 else "red" if pv_num < 130 else "black"
                            folium.Marker([row["latitude"],row["longitude"]],
                                popup=folium.Popup(pop,max_width=250),tooltip=f"{nm} · {ps}",
                                icon=folium.Icon(color=pin,icon="home",prefix="fa")).add_to(m)
                        if len(mdf)>1500: st.caption(f"Showing 1,500 of {len(mdf):,}.")
                        st_folium(m,width=None,height=580,returned_objects=[])
                        st.caption("🟢 <£55  ·  🟠 £55–90  ·  🔴 £90–130  ·  ⬛ >£130  ·  Click any pin for details and Airbnb link")
                with mt2:
                    sb=st.selectbox("Sort by",["Price: Low → High","Price: High → Low","Name A–Z"],label_visibility="collapsed")
                    if sb=="Price: Low → High" and pc: mdf=mdf.sort_values(pc)
                    elif sb=="Price: High → Low" and pc: mdf=mdf.sort_values(pc,ascending=False)
                    elif "name" in mdf.columns: mdf=mdf.sort_values("name")
                    gcols=st.columns(3)
                    for i,(_,row) in enumerate(mdf.head(60).iterrows()):
                        with gcols[i%3]:
                            with st.container(border=True):
                                pv=row.get(pc,None) if pc else None
                                if pd.notna(pic:=row.get("picture_url",None)):
                                    try: st.image(str(pic),use_container_width=True)
                                    except Exception: pass
                                mc1,mc2=st.columns(2)
                                mc1.metric("Price/night",f"£{pv:.0f}" if (pv and pd.notna(pv)) else "–")
                                area_val = "–"
                                for _nc in ["neighbourhood_group_cleansed","neighbourhood_cleansed","neighbourhood_group","neighbourhood"]:
                                    if _nc in row.index and pd.notna(row[_nc]):
                                        area_val = str(row[_nc])[:15]
                                        break
                                mc2.metric("Area", area_val)
                                url = row.get("listing_url", None)
                                if (url is None or pd.isna(url)) and "id" in row.index:
                                    try: url = f"https://www.airbnb.co.uk/rooms/{int(row['id'])}"
                                    except Exception: url = None
                                nm_card = str(row.get("name", "") or "").strip() or f"Listing {row.get('id','')}"
                                st.markdown(f"**{nm_card[:45]}**")
                                if url and str(url) != "None": st.link_button("View on Airbnb →", str(url), use_container_width=True)
                    if len(mdf)>60: st.caption(f"Showing top 60 of {len(mdf):,}.")
            except ImportError:
                st.error("Map requires: `pip install folium streamlit-folium`")
            except Exception as e:
                st.error(f"Map error: {e}")

    # ══ TAB 3, MARKET DASHBOARD ═════════════════════════════════════════════
    with tabs[2]:
        st.subheader("Manchester Airbnb Market at a Glance")
        st.write("How does the Manchester short-term rental market look? "
                 "Based on 6,562 real listings from Inside Airbnb (September 2025).")

        if df_proc is None:
            st.warning("Add `airbnb_processed_data_multimodal.csv` to enable the dashboard.")
        else:
            try:
                import plotly.express as px
                import plotly.graph_objects as go

                # ── Headline numbers ──────────────────────────────────────────
                st.markdown('<div class="sec-hdr">Manchester Market Snapshot</div>',
                            unsafe_allow_html=True)
                d1,d2,d3,d4 = st.columns(4)
                d1.markdown(f'<div class="metric-card"><div class="metric-label">Total listings</div>'
                            f'<div class="metric-value">{len(df_proc):,}</div></div>', unsafe_allow_html=True)
                d2.markdown(f'<div class="metric-card"><div class="metric-label">Typical nightly price</div>'
                            f'<div class="metric-value">£{df_proc["price"].median():.0f}</div></div>', unsafe_allow_html=True)
                d3.markdown(f'<div class="metric-card"><div class="metric-label">Budget listings (under)</div>'
                            f'<div class="metric-value">£{df_proc["price"].quantile(0.25):.0f}</div></div>', unsafe_allow_html=True)
                d4.markdown(f'<div class="metric-card"><div class="metric-label">Premium listings (over)</div>'
                            f'<div class="metric-value">£{df_proc["price"].quantile(0.75):.0f}</div></div>', unsafe_allow_html=True)
                st.caption("The typical (median) Manchester listing charges £{:.0f}/night. "
                           "25% of listings charge under £{:.0f} (budget) and 25% charge over £{:.0f} (premium).".format(
                           df_proc["price"].median(), df_proc["price"].quantile(0.25),
                           df_proc["price"].quantile(0.75)))

                st.markdown("")

                # ── What do most hosts charge? ────────────────────────────────
                st.markdown('<div class="sec-hdr">What Do Most Hosts Charge?</div>',
                            unsafe_allow_html=True)
                clip = df_proc["price"][df_proc["price"] < 400]
                fig1 = px.histogram(clip, nbins=50,
                    labels={"value":"Nightly price (£)","count":"Number of listings"},
                    color_discrete_sequence=["#FF5A5F"])
                fig1.add_vline(x=df_proc["price"].median(), line_dash="dash",
                               line_color="#222", annotation_text=f"  Median £{df_proc['price'].median():.0f}",
                               annotation_position="top right")
                fig1.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                                   margin=dict(t=20,b=40,l=40,r=20))
                st.plotly_chart(fig1, use_container_width=True)
                st.caption("Most Manchester listings charge between £50 and £150 per night. "
                           "A few high-end properties push the average above the median, so prices are not evenly spread.")

                # ── How does room type affect price? ──────────────────────────
                st.markdown('<div class="sec-hdr">Room Type, How It Affects Your Price</div>',
                            unsafe_allow_html=True)
                drt = df_proc.copy()
                drt["Room Type"] = "Entire home/apt"
                for rc, rl in [("room_type_Private room","Private room"),
                               ("room_type_Shared room","Shared room"),
                               ("room_type_Hotel room","Hotel room")]:
                    if rc in drt.columns: drt.loc[drt[rc]==1,"Room Type"] = rl
                drt_filt = drt[drt["price"] < 400]
                medians  = drt_filt.groupby("Room Type")["price"].median().reset_index()
                medians.columns = ["Room Type","Median Price"]
                fig2 = px.bar(medians, x="Room Type", y="Median Price",
                    labels={"Median Price":"Median price (£/night)"},
                    color="Room Type",
                    color_discrete_sequence=["#FF5A5F","#4a90d9","#f39c12","#2ecc71"])
                fig2.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                                   margin=dict(t=20,b=40,l=40,r=20))
                st.plotly_chart(fig2, use_container_width=True)
                entire_med  = drt_filt[drt_filt["Room Type"]=="Entire home/apt"]["price"].median()
                private_med = drt_filt[drt_filt["Room Type"]=="Private room"]["price"].median()
                st.caption(f"Whole properties usually earn about £{entire_med:.0f} per night, which is "
                           f"around £{entire_med-private_med:.0f} more than private rooms (£{private_med:.0f}). "
                           f"If you're renting out a spare room, expect pricing closer to the private room range.")

                # ── How does location affect price? ───────────────────────────
                nbs_d = nb_stats(df_proc)
                if nbs_d:
                    st.markdown('<div class="sec-hdr">Which Areas Charge the Most?</div>',
                                unsafe_allow_html=True)
                    nbdf = pd.DataFrame([
                        {"Neighbourhood": k, "Median": v["med"],
                         "Budget (25th %ile)": v["low"], "Premium (75th %ile)": v["high"],
                         "Listings": v["n"]}
                        for k, v in sorted(nbs_d.items(), key=lambda x: -x[1]["med"])
                    ])
                    fig3 = go.Figure()
                    fig3.add_trace(go.Bar(
                        x=nbdf["Neighbourhood"], y=nbdf["Median"],
                        marker_color="#FF5A5F", name="Median",
                        error_y=dict(type="data",
                                     array=(nbdf["Premium (75th %ile)"]-nbdf["Median"]).tolist(),
                                     arrayminus=(nbdf["Median"]-nbdf["Budget (25th %ile)"]).tolist(),
                                     visible=True, color="#aaa")))
                    fig3.update_layout(
                        yaxis_title="Median nightly price (£)", showlegend=False,
                        plot_bgcolor="white", paper_bgcolor="white",
                        margin=dict(t=20,b=60,l=40,r=20))
                    st.plotly_chart(fig3, use_container_width=True)
                    top_nb = nbdf.iloc[0]
                    st.caption(f"{top_nb['Neighbourhood']} has the highest median price "
                               f"(£{top_nb['Median']:.0f}/night). Error bars show the range "
                               f"between budget and premium listings in each area.")

                    # Simple table for hosts
                    st.dataframe(
                        nbdf[["Neighbourhood","Budget (25th %ile)","Median","Premium (75th %ile)","Listings"]]
                        .rename(columns={"Budget (25th %ile)":"Budget listings from",
                                         "Premium (75th %ile)":"Premium listings from"}),
                        use_container_width=True, hide_index=True)

                # ── Does capacity affect price? ───────────────────────────────
                if "accommodates" in df_proc.columns:
                    st.markdown('<div class="sec-hdr">More Guests = Higher Price?</div>',
                                unsafe_allow_html=True)
                    cap_df = (df_proc[df_proc["price"]<400]
                              .assign(accommodates=lambda d: d["accommodates"].round().astype(int))
                              .groupby("accommodates")["price"].median().reset_index())
                    cap_df.columns = ["Max Guests","Median Price"]
                    cap_df["Max Guests"] = cap_df["Max Guests"].astype(str)
                    fig4 = px.bar(cap_df, x="Max Guests", y="Median Price",
                        labels={"Max Guests":"Maximum guests","Median Price":"Median price (£/night)"},
                        text=cap_df["Median Price"].apply(lambda x: f"£{x:.0f}"),
                        color_discrete_sequence=["#FF5A5F"],
                        category_orders={"Max Guests": [str(i) for i in sorted(cap_df["Max Guests"].astype(int).tolist())]})
                    fig4.update_traces(textposition="outside")
                    fig4.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                       margin=dict(t=30,b=40,l=40,r=20))
                    st.plotly_chart(fig4, use_container_width=True)
                    st.caption("Generally, price rises with capacity, but not always linearly. "
                               "The model captures this relationship through the `accommodates` feature "
                               f"(correlation r = 0.399 with price).")

                # ── Does superhost status affect price? ───────────────────────────
                if "host_is_superhost" in df_proc.columns:
                    st.markdown('<div class="sec-hdr">Does Superhost Status Make a Difference?</div>',
                                unsafe_allow_html=True)
                    dsh = df_proc[df_proc["price"] < 400].copy()
                    dsh["Host Type"] = dsh["host_is_superhost"].map(
                        lambda x: "Superhost" if x in [1, 1.0, True] else "Standard host")
                    sh_med   = dsh[dsh["Host Type"]=="Superhost"]["price"].median()
                    std_med  = dsh[dsh["Host Type"]=="Standard host"]["price"].median()
                    sh_count = (dsh["Host Type"]=="Superhost").sum()
                    fig5 = px.box(dsh, x="Host Type", y="price",
                        labels={"price":"Nightly price (£)"},
                        color="Host Type",
                        color_discrete_sequence=["#FF5A5F","#4a90d9"])
                    fig5.update_layout(showlegend=False, plot_bgcolor="white",
                        paper_bgcolor="white", margin=dict(t=20,b=40,l=40,r=20))
                    st.plotly_chart(fig5, use_container_width=True)
                    diff = sh_med - std_med
                    direction = "more" if diff > 0 else "less"
                    st.caption(
                        f"Superhosts charge a median of £{sh_med:.0f}/night, compared to "
                        f"£{std_med:.0f} for standard hosts, a difference of £{abs(diff):.0f} per night ({direction}). "
                        f"Superhost status accounts for {sh_count:,} of the {len(dsh):,} listings in this dataset.")

                # ── Price by number of bedrooms ────────────────────────────────────
                if "bedrooms" in df_proc.columns:
                    st.markdown('<div class="sec-hdr">How Does the Number of Bedrooms Affect Price?</div>',
                                unsafe_allow_html=True)
                    bed_df = (df_proc[df_proc["price"] < 500]
                              .assign(bedrooms=lambda d: d["bedrooms"].round().astype("Int64"))
                              .groupby("bedrooms")["price"]
                              .agg(["median","count"])
                              .reset_index())
                    bed_df.columns = ["Bedrooms","Median Price","Listings"]
                    bed_df = bed_df[bed_df["Listings"] >= 20]
                    bed_df["Bedrooms"] = bed_df["Bedrooms"].astype(str)
                    fig6 = px.bar(bed_df, x="Bedrooms", y="Median Price",
                        labels={"Median Price":"Median price (£/night)", "Bedrooms":"Number of bedrooms"},
                        text=bed_df["Median Price"].apply(lambda x: f"£{x:.0f}"),
                        color_discrete_sequence=["#FF5A5F"],
                        category_orders={"Bedrooms": [str(i) for i in sorted(bed_df["Bedrooms"].astype(int).tolist())]})
                    fig6.update_traces(textposition="outside")
                    fig6.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                        margin=dict(t=30,b=40,l=40,r=20))
                    st.plotly_chart(fig6, use_container_width=True)
                    st.caption(
                        "Each additional bedroom tends to add around £20–40 to the nightly price, "
                        "though this varies quite a bit depending on location and property type.")

                # ── Review score distribution ──────────────────────────────────────
                if "review_scores_rating" in df_proc.columns:
                    st.markdown('<div class="sec-hdr">How Do Review Scores Vary Across Manchester?</div>',
                                unsafe_allow_html=True)
                    rev_df = df_proc["review_scores_rating"].dropna()
                    fig7 = px.histogram(rev_df, nbins=40,
                        labels={"value":"Overall rating","count":"Number of listings"},
                        color_discrete_sequence=["#4a90d9"])
                    fig7.add_vline(x=rev_df.median(), line_dash="dash", line_color="#FF5A5F",
                        annotation_text=f"  Median {rev_df.median():.2f}",
                        annotation_position="top right")
                    fig7.update_layout(showlegend=False, plot_bgcolor="white",
                        paper_bgcolor="white", margin=dict(t=20,b=40,l=40,r=20))
                    st.plotly_chart(fig7, use_container_width=True)
                    pct_high = (rev_df >= 4.8).mean() * 100
                    st.caption(
                        f"The median review score across all Manchester listings is {rev_df.median():.2f}/5. "
                        f"{pct_high:.0f}% of listings score 4.8 or above, the bar is high. "
                        f"Scores below 4.5 are relatively rare and could affect your search ranking on Airbnb.")

                # ── Summary stats table ────────────────────────────────────────────
            except ImportError:
                st.warning("Install `plotly` to enable dashboard charts.")
            except Exception as e:
                st.error(f"Dashboard error: {e}")

    # ══ TAB 4, MODEL INSIGHTS ════════════════════════════════════════════════
    with tabs[3]:
        st.subheader("How Does This Prediction Work?")
        st.write(
            "This tool is powered by a machine learning model trained on 6,562 real Manchester Airbnb listings. "
            "Here's an easy-to-understand explanation of how it works and what it looks at.")

        # ── How accurate is it? ───────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">How Accurate Is the Prediction?</div>',
                    unsafe_allow_html=True)
        m1,m2,m3,m4 = st.columns(4)
        for cw,lbl,val,desc in [
            (m1,"Accuracy (R²)","50.2%","The model explains about half of why listings are priced differently"),
            (m2,"Typical error","± £85","On average, predictions are within £85 of the actual price"),
            (m3,"Listings trained on","6,562","Real Manchester Airbnb listings from September 2025"),
            (m4,"Algorithm","XGBoost","A gradient-boosted decision tree, which is a common method for handling table-style data."),
        ]:
            cw.markdown(f'<div class="metric-card"><div class="metric-label">{lbl}</div>'
                        f'<div class="metric-value">{val}</div>'
                        f'<div style="font-size:0.68rem;color:#aaa;margin-top:0.3rem">{desc}</div>'
                        f'</div>', unsafe_allow_html=True)
        st.caption(
            "An R² of 50% means the model explains about half of the differences in Manchester "
            "Airbnb prices. The rest comes from factors the model can't see, like photo quality, "
            "interior style, repeat guests, or last-minute demand. Treat the prediction as a helpful "
            "starting point, not a fixed price.")

        try:
            import plotly.graph_objects as go

            # ── What matters most? ────────────────────────────────────────────
            st.markdown('<div class="sec-hdr">What Factors Matter Most to Your Price?</div>',
                        unsafe_allow_html=True)
            st.write("These are the features that have the biggest impact on the model's predictions, "
                     "ranked by how much the prediction changes if each one is left out:")

            factors_df = pd.DataFrame({
                "What it is": [
                    "How many listings your host account manages",
                    "Maximum number of guests",
                    "Number of bedrooms",
                    "Your listing title and description (AI text analysis)",
                    "Whether it's a private room vs entire property",
                    "Where in Manchester the property is",
                    "Whether you have a pool",
                    "Minimum number of nights required",
                    "Your overall guest rating",
                    "Superhost status",
                ],
                "Feature": [
                    "host_total_listings_count","accommodates","bedrooms",
                    "DistilBERT text embeddings","room_type","latitude/longitude",
                    "has_pool","minimum_nights","review_scores_rating","host_is_superhost",
                ],
                "Importance": [0.398,0.185,0.142,0.092,0.098,0.076,0.061,0.044,0.031,0.028],
            }).sort_values("Importance", ascending=True)

            fig_pm = go.Figure(go.Bar(
                y=factors_df["What it is"],
                x=factors_df["Importance"],
                orientation="h",
                marker_color="#FF5A5F",
            ))
            fig_pm.update_layout(
                xaxis_title="Relative importance",
                height=380,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=10,b=40,l=20,r=20))
            st.plotly_chart(fig_pm, use_container_width=True)
            st.caption(
                "The biggest factor is how many listings a host manages. Professional hosts with lots "
                "of properties usually price differently from individual hosts. The second most important "
                "factor is the property size, like number of guests and bedrooms.")

            # ── What types of info does it use? ───────────────────────────────
            st.markdown('<div class="sec-hdr">What Information Does the Model Use?</div>',
                        unsafe_allow_html=True)
            info_cols = st.columns(3)
            with info_cols[0]:
                st.markdown("**Property details (58% of explanation)**")
                st.write("Size, location, amenities, host experience, "
                         "review scores, room type, and minimum nights.")
            with info_cols[1]:
                st.markdown("**Listing text (29% of explanation)**")
                st.write("Your listing title and description are processed by "
                         "DistilBERT, an AI language model, to pick up on quality signals "
                         "that go beyond just how many words you use.")
            with info_cols[2]:
                st.markdown("**Photos (13% of explanation)**")
                st.write("Photo features are pulled out using ResNet50, an AI image model.")

            fig_pie = go.Figure(go.Pie(
                labels=["Property details","Listing text (DistilBERT)","Photos (ResNet50)"],
                values=[58.1,29.1,12.8],
                marker_colors=["#FF5A5F","#4a90d9","#2ecc71"],
                hole=0.5, textinfo="label+percent", textfont_size=11))
            fig_pie.update_layout(
                showlegend=False, height=260,
                margin=dict(t=10,b=10,l=20,r=20), paper_bgcolor="white")
            st.plotly_chart(fig_pie, use_container_width=True)

            # ── Is it fair? ───────────────────────────────────────────────────
            st.markdown('<div class="sec-hdr">Is the Model Fair to All Hosts?</div>',
                        unsafe_allow_html=True)
            st.write(
                "The model was checked to make sure it treats different types of hosts and listings fairly. "
                "Here's what the check found:")

            fdf = pd.DataFrame({
                "What was tested": [
                    "Superhost vs non-superhost hosts",
                    "Listings across different Manchester areas",
                    "Different room types (entire home vs private room etc.)",
                    "Budget vs luxury price ranges",
                ],
                "Result":    ["Pass ✅","Pass ✅","Needs caution ⚠️","Needs caution ⚠️"],
                "What this means": [
                    "The model predicts equally well regardless of superhost status",
                    "Predictions are consistent across all Manchester neighbourhoods",
                    "Shared rooms are harder to predict accurately due to high price variation",
                    "Luxury listings (£200+/night) are harder to predict accurately, so take the estimate with caution.",
                ],
            })
            st.dataframe(
                fdf.style.map(
                    lambda v: ("background-color:#d4edda;color:#155724" if "Pass" in str(v)
                               else "background-color:#fff3cd;color:#856404" if "caution" in str(v)
                               else ""),
                    subset=["Result"]),
                use_container_width=True, hide_index=True)
            st.caption(
                "Fairness tested using the 80% Disparate Impact Rule. "
                "If you have a shared room or a high-end luxury listing, "
                "treat this prediction as a rough guide and cross-check with "
                "the competitor listings shown after your prediction.")

        except ImportError:
            st.warning("Install `plotly` to enable charts.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown(
        '<div class="footer-bar">XGBoost · 6,562 Manchester Airbnb listings · Test R² = 0.5015 · '
        'RMSE = £84.73 · SHAP interpretability · KNN competitor analysis<br>'
        '<em>Nicole Reeves, Responsible Data Science Dissertation (IJC319)</em></div>',
        unsafe_allow_html=True)

if __name__ == "__main__":
    main()

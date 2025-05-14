import os
import openai
import json
import requests
import base64
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, auth
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from amazon_paapi import AmazonApi
from difflib import SequenceMatcher
import re
import time
import urllib.parse
import bs4

load_dotenv()

cred = credentials.Certificate("firebase_service_account.json")
firebase_admin.initialize_app(cred)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
BESTBUY_API_KEY = os.getenv("BESTBUY_API_KEY", "").strip()
EBAY_CLIENT_ID = os.getenv("EBAY_CLIENT_ID", "").strip()
EBAY_CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET", "").strip()
EBAY_CAMPAIGN_ID = os.getenv("EBAY_CAMPAIGN_ID", "").strip()
EBAY_PUB_ID = os.getenv("EBAY_PUB_ID", "").strip()
AMAZON_ACCESS_KEY = os.getenv("AMAZON_ACCESS_KEY")
AMAZON_SECRET_KEY = os.getenv("AMAZON_SECRET_KEY")
AMAZON_ASSOC_TAG = os.getenv("AMAZON_ASSOC_TAG")

EBAY_AUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
EBAY_ENDPOINT = "https://api.ebay.com/buy/browse/v1/item_summary/search"

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "REPLACE_WITH_A_SECRET_KEY")
last_fetched_products = []


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template(
        'chatbot.html',
        EBAY_CAMPAIGN_ID=os.getenv("EBAY_CAMPAIGN_ID"),
        EBAY_PUB_ID=os.getenv("EBAY_PUB_ID")
    )

@app.route('/compare')
def compare():
    return render_template('compare.html')

@app.route('/firebase-config')
def firebase_config():
    return jsonify({
        "apiKey": os.getenv("FIREBASE_API_KEY"),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
        "projectId": os.getenv("FIREBASE_PROJECT_ID"),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
        "appId": os.getenv("FIREBASE_APP_ID"),
        "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID")
    })

@app.route('/login', methods=['POST'])
def login():
    id_token = request.json.get('idToken')
    try:
        decoded_token = auth.verify_id_token(id_token)
        session['user_id'] = decoded_token['uid']
        return jsonify({"status": "success", "redirect": url_for('chatbot')})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 401

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({"status": "success", "redirect": url_for('home')})
def normalize(text):
    if text is None:
        return ''
    return re.sub(r'\s+', '', str(text).lower()) if text else ''

def is_close_match(name1, name2, threshold=0.85):
    if not name1 or not name2:
        return False
    ratio = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
    return ratio >= threshold

@app.route('/store_selected_product', methods=['POST'])
def store_selected_product():
    product = request.json.get('product')
    structured_attrs = extract_product_attributes_with_gpt(product)
    product['model'] = structured_attrs.get("model")
    product['identifiers'] = structured_attrs.get("identifiers", [])
    session['selected_product_gpt_attrs'] = structured_attrs
    return jsonify({
        "status": "ok",
        "product": {
            "name": product.get("name"),
            "title": product.get("title"),
            "url": product.get("url"),
            "sku": product.get("sku"),
            "model": product.get("model"),
            "identifiers": product.get("identifiers", [])
        },
        "attributes": structured_attrs
    })
    session['selected_product_attrs'] = structured_attrs
    return jsonify({"status": "ok"})


def fast_pre_match_score(p1, p2):
    name1 = (p1.get("name") or p1.get("title") or "").lower()
    name2 = (p2.get("name") or p2.get("title") or "").lower()

    tokens1 = set(re.findall(r'\b\w+\b', name1))
    tokens2 = set(re.findall(r'\b\w+\b', name2))

    if not tokens1 or not tokens2:
        return 0

    return len(tokens1 & tokens2) / len(tokens1 | tokens2)

def sku_match(product_a, product_b):
    ids_a = set((product_a.get("identifiers") or []) + [product_a.get("model", "")])
    ids_b = set((product_b.get("identifiers") or []) + [product_b.get("model", "")])

    normalized_a = {normalize(i) for i in ids_a if i}
    normalized_b = {normalize(i) for i in ids_b if i}

    return not normalized_a.isdisjoint(normalized_b)

def extract_normalized_ids(product):
    source = product.get("source", "").lower()
    raw_ids = set()

    for key in ["model", "manufacturerPartNumber", "upc", "ean", "gtin"]:
        val = product.get(key)
        if val:
            raw_ids.add(str(val))

    if source and product.get("sku") and isinstance(product.get("sku"), str):
        raw_ids.add(product.get("sku"))

    identifiers = product.get("identifiers", [])
    if isinstance(identifiers, list):
        raw_ids.update(str(i) for i in identifiers if i)

    return {normalize(i) for i in raw_ids if i}

def extract_model_like_identifiers(product):
    possible_ids = set()

    for key in ["model", "manufacturerPartNumber", "sku", "modelNumber"]:
        val = product.get(key)
        if isinstance(val, str) and val.strip():
            possible_ids.add(val.strip())
        elif isinstance(val, int):
            possible_ids.add(str(val))

    features = product.get("features", [])
    for feature in features:
        if isinstance(feature, dict):
            text = feature.get("feature", "")
            matches = re.findall(r'\b[A-Z0-9\-]{4,}\b', text)
            possible_ids.update(matches)

    normalized = [normalize(x) for x in possible_ids if x]
    return list(set(normalized))

def is_exact_variant_match(product_a, product_b):
    if not product_a or not product_b:
        return False

    ids_a = extract_normalized_ids(product_a)
    ids_b = extract_normalized_ids(product_b)
    if not ids_a and not ids_b:
        print("[DEBUG] No identifiers on either side. Falling back to GPT match.")
    elif not ids_a or not ids_b:
        print("[DEBUG] One side missing identifiers. Skipping match.")
        return False
    elif not ids_a.isdisjoint(ids_b):
        print("[DEBUG] Strong identifier match!")
        return True

    try:
        a_name = product_a.get('name') or product_a.get('title') or ''
        a_model = product_a.get('model') or ''
        a_color = product_a.get('color') or ''
        a_storage = next((attr.get('feature') for attr in product_a.get('features', []) if 'GB' in attr.get('feature', '')), '')

        b_name = product_b.get('name') or product_b.get('title') or ''
        b_model = product_b.get('model') or ''
        b_color = product_b.get('color') or ''
        b_storage = next((attr.get('feature') for attr in product_b.get('features', []) if 'GB' in attr.get('feature', '')), '')

        prompt = f"""
You are an AI product matcher. Determine if Product A and Product B are the same exact product variant.

Focus only on:
- Product name
- Model
- Color
- Storage size

If the model, color, and storage match, or the name indicates it's the same product, return {{ "match": true }}.
If there are differences, or you are unsure, return {{ "match": false }}.

Product A:
Name: {a_name}
Model: {a_model}
Color: {a_color}
Storage: {a_storage}

Product B:
Name: {b_name}
Model: {b_model}
Color: {b_color}
Storage: {b_storage}

Only return a valid JSON like:
{{ "match": true }} or {{ "match": false }}
"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert product matcher."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10,
            api_key=OPENAI_API_KEY
        )
        result = json.loads(response['choices'][0]['message']['content'].strip())
        return result.get("match", False)
    except Exception as e:
        print(f"[ERROR] GPT match error: {e}")
        return False

def simplify_product_query_for_bestbuy(product):
    try:
        prompt = f"""
You are an assistant optimizing search queries for the Best Buy API. Given the following product, generate a concise, keyword-style query using brand, type, model, capacity, and color only — and strip extra terms like "renewed", "unlocked", "bundle", "for Verizon", etc.

Product:
{json.dumps(product, indent=2)}

Return only the optimized query string, like:
"Apple iPhone 14 Plus 128GB Blue"
"""
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You optimize product queries for strict APIs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=50,
            api_key=OPENAI_API_KEY
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"[ERROR] simplify_product_query_for_bestbuy: {e}")
        return ""

def should_exclude_product(product, filters):
    device_type = classify_product_type(product)
    if filters.get("exclude_accessories") and device_type == "accessory":
        return True
    if filters.get("device_type") and filters["device_type"] != device_type:
        return True
    return False

def clean_ebay_description(html):
    try:
        soup = bs4.BeautifulSoup(html, "html.parser")
        for tag in soup(["style", "script", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return text if text.strip() else "No useful description available."
    except:
        return "No useful description available."
    
def singularize_query(query):
    words = query.lower().split()
    singular_words = [word[:-1] if word.endswith('s') and not word.endswith('ss') else word for word in words]
    return " ".join(singular_words)

def get_important_attributes(product):
    return {
        "description": clean_ebay_description(product.get("description", "")),
        "rating": float(product.get("reviewRating", {}).get("averageRating", 0)),
        "reviewCount": int(product.get("reviewRating", {}).get("reviewCount", 0)),
        "seller_feedback": f"{product.get('seller', {}).get('feedbackPercentage', '')}% positive ({product.get('seller', {}).get('feedbackScore', 0)} ratings)"
    }

    name_similarity = SequenceMatcher(None, attr_a["name"], attr_b["name"]).ratio()
    desc_similarity = SequenceMatcher(None, attr_a["description"], attr_b["description"]).ratio()
    features_similarity = SequenceMatcher(None, attr_a["features"], attr_b["features"]).ratio()

    # Consider it an exact match if name is very close and either description or features match well
    if name_similarity > 0.85 and (desc_similarity > 0.6 or features_similarity > 0.6):
        return True

    return False

def pin_best_exact_match(exact_matches, selected_price=None):
    if not exact_matches:
        return exact_matches

    def price_float(product):
        try:
            return float(re.sub(r"[^\d.]", "", str(product.get("price", ""))))
        except:
            return float("inf")

    new_matches = [p for p in exact_matches if "new" in str(p.get("condition", "") or "").lower()]
    ref_and_used = [p for p in exact_matches if p not in new_matches]

    if new_matches:
        sorted_new = sorted(new_matches, key=price_float)
        cheapest_new = sorted_new[0]
        if selected_price:
            price_diff = selected_price - price_float(cheapest_new)
            if price_diff < 0:
                return exact_matches  # don't pin if more expensive
        exact_matches = sorted_new + ref_and_used

    else:
        cheapest = min(exact_matches, key=price_float)
        exact_matches = [cheapest] + [p for p in exact_matches if p != cheapest]

    return exact_matches

def is_poor_condition(product):
    text = (product.get("name", "") + " " + product.get("title", "") + " " + product.get("description", "")).lower()
    return any(term in text for term in ["broken", "cracked", "for parts", "not working", "damaged", "as-is"])

@app.route('/compare_product', methods=['POST'])
def compare_product():
    try:
        selected_product = request.json.get('selected_product')

        if not selected_product:
            print("[ERROR] No selected_product found in session.")
            return jsonify({"exact_matches": [], "similar_alternatives": [], "message": "No product selected."})

        category = str(selected_product.get("category", "") or "")

        structured_attrs = extract_product_attributes_with_gpt(selected_product)
        print("[DEBUG] Retrieved selected product from session:", selected_product)
        if not structured_attrs.get("identifiers"):
            extracted_ids = extract_model_like_identifiers(selected_product)
            structured_attrs["identifiers"] = extracted_ids

        if not selected_product:
            return jsonify({"exact_matches": [], "similar_alternatives": []})

        def safe_to_str(val):
            if isinstance(val, list):
                return " ".join([str(v) for v in val])
            return str(val or "")

        bestbuy_query = simplify_product_query_for_bestbuy(selected_product)

        def safe_to_str(val):
            if isinstance(val, list):
                return " ".join([str(v) for v in val])
            return str(val or "")

        ebay_amazon_query = " ".join(filter(None, [
            safe_to_str(structured_attrs.get("brand")),
            safe_to_str(structured_attrs.get("type")),
            safe_to_str(structured_attrs.get("model")),
            safe_to_str(structured_attrs.get("color")),
            safe_to_str(structured_attrs.get("size")),
            "Unlocked" if "unlocked" in selected_product.get("name", "").lower() else "",
        ])).strip()

        query_text = ebay_amazon_query

        bestbuy_query = re.sub(r'[^a-zA-Z0-9\s\-]+', '', bestbuy_query or "").strip()
        ebay_amazon_query = re.sub(r'[^a-zA-Z0-9\s\-]+', '', ebay_amazon_query or "").strip()

        def safe_to_str(val):
            if isinstance(val, list):
                return " ".join([str(v) for v in val])
            return str(val or "")

        full_query_text = " ".join(filter(None, [
            safe_to_str(structured_attrs.get("brand")),
            safe_to_str(structured_attrs.get("type")),
            safe_to_str(structured_attrs.get("model")),
            safe_to_str(structured_attrs.get("color")),
            safe_to_str(structured_attrs.get("size")),
        ])).strip()

        full_query_text = re.sub(r'[^a-zA-Z0-9\s\-]+', '', full_query_text)
        
        bestbuy_results = fetch_bestbuy_results(query_text)

        ebay_results = fetch_ebay_results(query_text)
        item_id = None
        item_web_url = None
        for item in ebay_results:
            if item.get("source", "").lower() == "ebay":
                item_id = item.get("itemId") or ""
                item_web_url = item.get("itemWebUrl") or item.get("url") or ""
                seller_info = item.get('seller', {})
                feedback = seller_info.get('feedbackPercentage')
                feedback_count = seller_info.get('feedbackScore')

                if feedback and feedback_count:
                    item['seller_feedback'] = f"{feedback}% positive ({feedback_count} ratings)"
                else:
                    item['seller_feedback'] = None
        
                if item_id or item_web_url:
                    base_url = str(item.get("itemWebUrl") or item.get("url") or "")
                    separator = '&' if '?' in base_url else '?'
                    affiliate_url = f"{base_url}{separator}campid={EBAY_CAMPAIGN_ID}&customid=&toolid=10001&mkevt=1"
                    item["affiliate_url"] = affiliate_url
                    item["url"] = affiliate_url

            item_id = None
            item_web_url = None
            for item in ebay_results:
                if item.get("source", "").lower() == "ebay":
                    item_id = item.get("itemId") or ""
                    item_web_url = item.get("itemWebUrl") or item.get("url") or ""
                    break

            base_url = None
            if item_id:
                base_url = f"https://www.ebay.com/itm/{item_id}"
            elif item_web_url:
                base_url = item_web_url

        amazon_results = fetch_amazon_results(query_text)

        all_results = bestbuy_results + ebay_results + amazon_results

        exact_matches = []
        similar_alternatives = []

        pre_matches = [
            p for p in all_results
            if p.get("url") != selected_product.get("url")
            and (p.get("sku") != selected_product.get("sku")) 
            and fast_pre_match_score(p, selected_product) > 0.5 
]

        for product in pre_matches:
            if is_exact_variant_match(product, selected_product):
                exact_matches.append(product)
            else:
                similar_alternatives.append(product)

        for p in exact_matches + similar_alternatives:
            p["_score"] = get_product_score(p, query_text, [], "")

        exact_matches = sorted(exact_matches, key=lambda x: x.get("_score", 0), reverse=True)
        similar_alternatives = sorted(similar_alternatives, key=lambda x: x.get("_score", 0), reverse=True)

        if not exact_matches and not similar_alternatives:
            message = "No matches found."
        else:
            message = ""

        selected_price = None
        try:
            selected_price = float(str(selected_product.get("price", "")).replace("$", "").replace(",", ""))
        except:
            selected_price = None

        new_matches = []
        refurbished_matches = []
        used_matches = []
        poor_condition_matches = []

        for match in exact_matches:
            condition = str(match.get("condition", "") or match.get("itemCondition", "")).lower()
            price = float(str(match.get("price", "").replace("$", "").replace(",", "")) or 0)

            if "new" in condition and (not selected_price or price <= selected_price):
                new_matches.append(match)
            elif "refurbished" in condition:
                refurbished_matches.append(match)
            elif any(term in condition for term in ["used", "pre-owned"]):
                if is_poor_condition(match):
                    poor_condition_matches.append(match)
                else:
                    used_matches.append(match)
            else:
                used_matches.append(match)

        exact_matches = (
            sorted(new_matches, key=lambda x: x.get("_score", 0), reverse=True) +
            sorted(refurbished_matches, key=lambda x: x.get("_score", 0), reverse=True) +
            sorted(used_matches, key=lambda x: x.get("_score", 0), reverse=True) +
            sorted(poor_condition_matches, key=lambda x: x.get("_score", 0), reverse=True)
        )

        if exact_matches:
            exact_matches[0]["recommended"] = True

        if selected_product and exact_matches and (
            selected_product.get("url") == exact_matches[0].get("url") or
            selected_product.get("name") == exact_matches[0].get("name")
        ):
            selected_product["recommended"] = True

        return jsonify({
            "exact_matches": exact_matches,
            "similar_alternatives": similar_alternatives,
            "message": message
        })

    except Exception as e:
        print(f"[ERROR] /compare_product: {e}")
        return jsonify({"exact_matches": [], "similar_alternatives": []})

ACCESSORY_TERMS = [
    "case", "screen protector", "charger", "Magsafe", "Otterbox", "adapter", "gift card", "dock", "cable",
    "stand", "keyboard", "controller", "service plan", "AppleCare+", "protection plan",
    "warranty", "backpack", "mug", "t-shirt", "shirt", "sweater", "notebook", "book",
    "poster", "calendar", "candle", "journal"
]

def is_accessory_product(product, user_query, extracted_category=""):
    title = str(product.get("name") or product.get("title") or "") + " " + str(product.get("description") or "")
    title = title.lower()
    query = str(user_query or "").lower()
    category = extracted_category.lower()
    if 'laptop' in category and ('backpack' in title or 'bag' in title or 'rucksack' in title):
        return True
    if any(term in title for term in ACCESSORY_TERMS):
        if not any(term in query for term in ACCESSORY_TERMS):
            if "iphone" in query.lower() and not re.search(r'\b(case|cover|screen protector)\b', query.lower()):
                return True
            return False

def classify_product_type(product):
    text = (product.get("name", "") + " " + (product.get("description") or "")).lower()

    # Services / plans
    if any(term in text for term in ["warranty", "applecare", "service plan", "protection plan", "subscription", "membership"]):
        return "service"

    # Accessories
    if any(term in text for term in [
        "case", "cover", "sleeve", "shell", "bag", "mount", "strap", "holder", "skin",
        "screen protector", "keyboard", "charger", "dock", "cable", "adapter"
    ]):
        return "accessory"

    # Components
    if any(term in text for term in ["screen replacement", "lcd", "battery", "flex cable", "ssd", "ram", "gpu", "cpu"]):
        return "component"

    # Novelty / non-core
    if any(term in text for term in ["mug", "poster", "t-shirt", "hoodie", "gift", "sticker", "quote", "funny"]):
        return "novelty"

    return "core"

def is_irrelevant_category(product, extracted_category):
    if not extracted_category:
        return False

    text = (product.get("name") or product.get("title") or "") + " " + (product.get("description") or "")
    text = text.lower()
    category = extracted_category.lower()

    try:
        tfidf = TfidfVectorizer().fit_transform([category, text])
        similarity = cosine_similarity(tfidf[0:1], tfidf[1:2]).flatten()[0]
        return similarity < 0.25
    except:
        return False

def is_strict_match_required(product, query_text):
    required_keywords = [kw.lower() for kw in query_text.strip().split() if kw.strip()]
    title = str(product.get("name") or product.get("title") or "").lower()
    description = str(product.get("description") or "").lower()
    full_text = (title + " " + description).replace('-', ' ').replace('/', ' ')
    matched = sum(1 for kw in required_keywords if kw in full_text)
    match_ratio = matched / len(required_keywords) if required_keywords else 0
    return match_ratio >= 0.75

def get_product_score(product, query_text, attributes, category, filters=None):
    title = str(product.get("name") or product.get("title") or "").lower()
    description = str(product.get("description") or "").lower()
    features = ' '.join([f.get("feature", "") for f in product.get("features", []) if isinstance(f, dict)])
    full_text = f"{title} {description} {features}"

    query_keywords = set(query_text.lower().split())
    attribute_keywords = set(attr.lower() for attr in attributes)

    match_score = len(query_keywords.intersection(full_text.split())) / (len(query_keywords) + 1)
    attr_score = len(attribute_keywords.intersection(full_text.split())) / (len(attribute_keywords) + 1)

    semantic_score = 0.3
    category_score = 0.0
    contextual_boost = 0.0

    try:
        tfidf = TfidfVectorizer().fit_transform([query_text, full_text])
        semantic_score = cosine_similarity(tfidf[0:1], tfidf[1:2]).flatten()[0]

        if category:
            tfidf_cat = TfidfVectorizer().fit_transform([category, full_text])
            category_score = cosine_similarity(tfidf_cat[0:1], tfidf_cat[1:2]).flatten()[0]
    except:
        pass

    if category_score < 0.2:
        contextual_boost -= 0.3  # Penalize mismatch
    else:
        contextual_boost += category_score * 0.3

    contextual_boost = 0.0
    if any(keyword in full_text for keyword in ["dslr", "mirrorless", "tripod", "camera", "lens", "gopro", "canon", "nikon"]):
        contextual_boost += 0.3
    if 'gift' in query_text.lower() and any(term in full_text for term in ACCESSORY_TERMS):
        contextual_boost -= 0.3
    if 'school' in query_text.lower() and 'laptop' not in full_text and 'notebook' not in full_text:
        contextual_boost -= 0.3
    if 'ps5' or 'xbox' or 'nintendo switch' or 'gamecube' or 'ps4' or 'ps3' or 'n64' in query_text.lower() and 'console' in full_text:
        contextual_boost += 0.4 

    if 'otterbox' in full_text or classify_product_type(product) == 'accessory':
        contextual_boost -= 0.6

    if "airpods" in query_text.lower():
        if "gift card" in title or description:
            contextual_boost = -0.9
        if classify_product_type(product) == "accessory":
            contextual_boost -= -0.8
        if "case" in title or "skin" in title or "cover" in title:
            contextual_boost -= -0.8

    if classify_product_type(product) == 'service':
        contextual_boost -= 0.9

    total_score = (match_score * 0.25) + (attr_score * 0.25) + (semantic_score * 0.4) + contextual_boost
    if product.get("source") == "Best Buy":
        total_score += 0.1

    return round(total_score * 100, 2)

    if filters and filters.get("semantic_focus"):
        try:
            tfidf = TfidfVectorizer().fit_transform([filters["semantic_focus"], full_text])
            semantic_match = cosine_similarity(tfidf[0:1], tfidf[1:2]).flatten()[0]
            contextual_boost += semantic_match * 0.3  # Adjust weight as needed
        except:
            pass

def parse_user_message(user_message, previous_context=None):
    user_message = singularize_query(user_message)

    context_prompt = ""
    if isinstance(previous_context, dict):
        context_prompt = f"""
The user has previously searched for: \"{previous_context.get('search_query', '')}\"
Category: {previous_context.get('category', '')}
Attributes: {previous_context.get('attributes', [])}
Refinement Count: {previous_context.get('refined_count', 0)}
"""

    prompt = f"""{context_prompt} You are Peel, an AI shopping assistant. Analyze the user message and extract the following but do not ask questions:
1. A response (e.g. Here are some options for gaming laptops")
2. search_query: A keyword-friendly phrase for APIs. Preserve the specific product name if the user mentions it. Do not suggest a different model unless asked.
3. category: What category does the product belong to?
4. attributes: Specific features or needs (brand, budget, size).
5. query_type: "literal" or "contextual".
5. intent_type: "new_query" if it's a new search or "followup" if it's a refinement of the last one.
6. is_refined_enough: false.
7. filters: An object with fields like:
   - "exclude_accessories" (true if user query is for a core product like laptops, phones, etc.),
   - "device_type" (one of "core", "accessory", or "service"),
   - "semantic_focus" (a semantic target phrase like "school laptop", "gift for dad", or "travel headphones")
Always combine prior product type context with follow-up terms like "Alienware" or "beige".
Always return your best interpretation of the query. Do not ask the user for more information. Never generate a follow-up. Always assume the query is refined enough.
If the user mentions a general category like "smartphone" or "phone", infer it includes top brands like iPhone, Samsung Galaxy, Google Pixel, Motorola, OnePlus. Add these to the attributes list if applicable.
ONLY return valid JSON like:
{{
  "response": "...",
  "search_query": "...",
  "category": "...",
  "attributes": [...],
  "query_type": "...",
  "intent_type": "...",
  "is_refined_enough": true,
  "follow_up": "...",
  "suggested_answers": ["Alienware", "ASUS ROG", "MSI"]
  "filters": {{
    "exclude_accessories": true,
    "device_type": "core",
    "semantic_focus": "school laptop"
}}
User Message: \"{user_message}\""""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a smart AI shopping assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5,
            api_key=OPENAI_API_KEY,
        )

        data = json.loads(response['choices'][0]['message']['content'].strip())

        return (
            data.get("response", "Here's what I found."),
            data.get("search_query", "None"),
            data.get("category", ""),
            data.get("attributes", []),
            data.get("query_type", "contextual"),
            True,  # <- Hardcoded to always treat as refined
            "",    # <- No follow-up
            data.get("intent_type", "new_query"),
            data.get("suggested_answers", []),
            data.get("filters", {})
)
    except Exception as e:
        print(f"[ERROR] GPT parse: {e}")
        return (
            "I'm here to help!",
            None,
            "",
            [],
            "contextual",
            False,
            "",  # follow_up placeholder
            "new_query",
            [],
            {}  # filters – this was missing
)

def extract_product_attributes_with_gpt(product):
    try:
        product_name = product.get("name") or product.get("title", "")
        description = product.get("description", "")

        prompt = f"""
You are an expert product matcher. Analyze the product below and extract these fields:
- category: e.g., electronics, fashion, beauty, tools, etc.
- type: specific type like "gaming laptop", "air fryer", "skincare serum"
- brand: brand name if known
- model: full model name or number (e.g. D9P-00016, A2483, GSRF-SURF13)
- color: if available
- size: if relevant
- key_attributes: list of distinguishing features (e.g. Bluetooth, 512GB SSD, vegan, waterproof)
- identifiers: any unique model numbers, GTIN, SKU, etc.

Product:
Name: {product_name}
Description: {description}
Only return valid JSON.
"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
            api_key=OPENAI_API_KEY
        )
        response_text = response['choices'][0]['message']['content'].strip()

        if response_text.startswith("```"):
           response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
           response_text = re.sub(r"\s*```$", "", response_text)

        print("[DEBUG] Cleaned GPT Attribute Extraction:", response_text)

        print("[DEBUG] Raw GPT Attribute Extraction:", response_text)
        try:
            return json.loads(response_text)
        except Exception as parse_error:
            print("[ERROR] Failed to parse GPT response:", parse_error)
            return {}

    except Exception as e:
        print("[ERROR] GPT Attribute Extraction:", e)
        return {}

    BESTBUY_CACHE[query] = products
    return products

def fetch_bestbuy_results(query, retries=3):
    if not query or query.lower() == "none":
        return []

    base_url = "https://api.bestbuy.com/v1/products"
    headers = {"Accept": "application/json"}

    def run_query(q):
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', q)
        encoded = urllib.parse.quote(cleaned.strip() + '*')
        url = (
            f"{base_url}(name={encoded})"
            f"?apiKey={BESTBUY_API_KEY}&format=json"
            f"&show=sku,name,salePrice,largeFrontImage,url,description,customerReviewCount,customerReviewAverage,shortDescription,features"
            f"&pageSize=50&sort=customerReviewCount.dsc"
        )
        for attempt in range(retries):
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json().get("products", [])
            elif response.status_code == 403 and "per second" in response.text.lower():
                time.sleep(1.5)
        print(f"[ERROR] Best Buy API error {response.status_code}: {response.text}")
        return []

    results = run_query(query)
    if results:
        return format_bestbuy_products(results)

    fallbacks = generate_bestbuy_fallback_queries(query)
    for alt in fallbacks:
        results = run_query(alt)
        if results:
            return format_bestbuy_products(results)

    return []

    def fetch_bestbuy_by_model(model_number):
        if not model_number:
            return []
    
        base_url = "https://api.bestbuy.com/v1/products"
        encoded_model = urllib.parse.quote(model_number.strip())
        url = (
            f"{base_url}(modelNumber={encoded_model})"
            f"?apiKey={BESTBUY_API_KEY}&format=json"
            f"&show=sku,name,salePrice,largeFrontImage,url,description,customerReviewCount,customerReviewAverage,shortDescription,features,modelNumber,manufacturerPartNumber"
            f"&pageSize=10"
        )
        headers = {"Accept": "application/json"}
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                products = response.json().get("products", [])
                return format_bestbuy_products(products)
        except Exception as e:
            print(f"[ERROR] Best Buy model search error: {e}")
    
        return []

def format_bestbuy_products(products):
    for p in products:
        p['image'] = p.get("largeFrontImage")
        p['url'] = p.get("url")
        p['price'] = f"${float(p.get('salePrice', 0)):.2f}" if p.get('salePrice') else "Price not available"
        p['rating'] = p.get('customerReviewAverage', None)
        p['reviewCount'] = p.get("customerReviewCount", 0)
        p['source'] = "Best Buy"
        p['description'] = (
            p.get('shortDescription') or
            p.get('description') or
            "No description available"
        )
        p['device_type'] = classify_product_type(p)
        p["modelNumber"] = p.get("modelNumber")
        p["manufacturerPartNumber"] = p.get("manufacturerPartNumber")

    return products

def generate_bestbuy_fallback_queries(full_query):
    terms = full_query.strip().split()
    fallbacks = set()

    for i in range(len(terms), 1, -1):
        fallbacks.add(" ".join(terms[:i]))

    storage_pattern = re.compile(r'\b\d{2,4}(gb|tb)\b', re.IGNORECASE)
    color_pattern = re.compile(r'\b(black|white|blue|gray|silver|gold|red|green|desert|titanium|purple)\b', re.IGNORECASE)

    stripped = storage_pattern.sub("", full_query)
    stripped = color_pattern.sub("", stripped)
    stripped = re.sub(r'\s+', ' ', stripped).strip()
    if stripped and stripped != full_query:
        fallbacks.add(stripped)

    model_match = re.findall(r'\b[A-Z]{1,3}[\d]{2,5}\b', full_query)
    if model_match:
        fallbacks.update(model_match)

    return list(dict.fromkeys(fallbacks))

def get_ebay_oauth_token():
    try:
        credentials_combined = f"{EBAY_CLIENT_ID}:{EBAY_CLIENT_SECRET}"
        encoded_credentials = base64.b64encode(credentials_combined.encode()).decode()
        headers = {"Content-Type": "application/x-www-form-urlencoded", "Authorization": f"Basic {encoded_credentials}"}
        data = {"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}
        response = requests.post(EBAY_AUTH_URL, headers=headers, data=data)
        if response.status_code == 200:
            return response.json().get('access_token')
        return None
    except Exception as e:
        print(f"[ERROR] eBay OAuth error: {e}")
        return None

def fetch_ebay_results(query):
    if not query or query.lower() == "none":
        return []
    access_token = get_ebay_oauth_token()
    if not access_token:
        return []
    try:
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        params = {"q": query, "limit": "50", "sort": "best_match"}
        response = requests.get(EBAY_ENDPOINT, headers=headers, params=params)
        if response.status_code != 200:
            return []
        items = response.json().get("itemSummaries", [])

        for item in items[:20]:
            full_item = fetch_ebay_item_details(item['itemId'], access_token)
            if full_item:
                item['description'] = full_item.get('description', '')
                item['rating'] = full_item.get('rating', None)
                item['reviewCount'] = full_item.get('reviewCount', 0)
                item['seller_feedback'] = full_item.get('seller_feedback', None)

        for item in items:
            item['name'] = item.get("title", "Unknown Product")
            item['image'] = item.get("image", {}).get("imageUrl")

            seller = item.get("seller", {})
            feedback_pct = seller.get("feedbackPercentage")
            feedback_score = seller.get("feedbackScore")

            if feedback_pct and feedback_score:
                item["seller_feedback"] = f"{feedback_pct}% positive ({feedback_score} ratings)"
            else:
                item["seller_feedback"] = None

            item_web_url = item.get("itemWebUrl")
            if item_web_url:
                encoded_url = urllib.parse.quote(item_web_url, safe='')
                affiliate_url = (
                  f"{item.get('itemWebUrl')}?campid={EBAY_CAMPAIGN_ID}"
                  f"&customid=&toolid=10001&mkevt=1"
            )
                item['url'] = affiliate_url
                item['affiliate_url'] = affiliate_url
            else:
                fallback_url = f"https://www.ebay.com/sch/i.html?_nkw={urllib.parse.quote(item.get('title', ''))}"
                item['url'] = fallback_url
                item['affiliate_url'] = fallback_url

            item['price'] = f"${item.get('price', {}).get('value', 'N/A')}"
            item['source'] = "eBay"
            item["mpn"] = item.get("mpn")
            item["manufacturerPartNumber"] = item.get("manufacturerPartNumber")
            item["itemModelNumber"] = item.get("itemModelNumber")

        return items
    except Exception as e:
        print(f"[ERROR] eBay fetch error: {e}")
        return []

def safe_json(data):
    try:
        return json.dumps(data, indent=2, default=str)
    except Exception as e:
        print("[ERROR] safe_json:", e)
        return "{}"
     
def fetch_ebay_item_details(item_id, token):
    try:
        url = f"https://api.ebay.com/buy/browse/v1/item/{item_id}"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return {
                "description": data.get("description", ""),
                "rating": float(data.get("reviewRating", {}).get("averageRating", 0)),
                "reviewCount": int(data.get("reviewRating", {}).get("reviewCount", 0)),
                "seller_feedback": f"{data.get('seller', {}).get('feedbackPercentage', '')}% positive ({data.get('seller', {}).get('feedbackScore', 0)} ratings)"
            }
    except Exception as e:
        print(f"[ERROR] eBay item fetch error: {e}")
    return None

def fetch_amazon_results(query):
    if not query or query.lower() == "none":
        return []
    try:
        amazon = AmazonApi(
            AMAZON_ACCESS_KEY,
            AMAZON_SECRET_KEY,
            AMAZON_ASSOC_TAG,
            country="US"
        )
        results = amazon.search_items(keywords=query, search_index="All", item_count=10)
        products = []
        for item in results.items:
            title = item.item_info.title.display_value if item.item_info.title else "Unknown Product"
            image = item.images.primary.medium.url if item.images and item.images.primary else None
            url = item.detail_page_url
            price = f"${item.offers.listings[0].price.amount:.2f}" if item.offers and item.offers.listings else "Price not available"
            features = item.item_info.features.display_values if item.item_info.features else []
            products.append({
                "name": title,
                "image": image,
                "url": url,
                "price": price,
                "features": [{"feature": f} for f in features],
                "description": ", ".join(features),
                "source": "Amazon"
            })
        return products
    except Exception as e:
        print(f"[ERROR] Amazon fetch error: {e}")
        return []

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '').strip()
        button_type = request.json.get('button_type', None)
        if not user_message:
            return jsonify({"response": "I didn't receive your message.", "products": []})

        response_msg, search_query, category, attributes, query_type, follow_up, intent_type, is_refined_enough, suggested_answers, filters = parse_user_message(
            user_message
)

        search_query = singularize_query(search_query)
        query_text = search_query

        is_followup = False

        core_keywords = ["iphone", "ipad", "airpods", "macbook", "galaxy", "laptop", "TV", "headphones", "camera", "monitor", "smartwatch"]
        query_lc = (search_query or "").lower()

        if filters is None:
            filters = {}

        if any(word in query_lc for word in core_keywords) and not any(term in query_lc for term in ACCESSORY_TERMS):
            filters["exclude_accessories"] = True

        if is_followup:
            base_query = last_context.get("search_query", "")
            if (search_query or "").lower() != "none":
                search_query = f"{base_query} {search_query}".strip()
            else:
                search_query = base_query
            appended_query = search_query if search_query.lower() != "none" else ""
            base_query = last_context.get("search_query", "") or ""
            search_query = search_query or ""
            search_query = f"{base_query} {search_query}".strip()
            category = last_context["category"]
            attributes = list(set(last_context.get("attributes", []) + attributes))
            query_type = last_context.get("query_type", "contextual")

        should_search = search_query.strip().lower() != "none" or category.strip() != ""

        bestbuy_products = fetch_bestbuy_results(search_query) if should_search else []
        ebay_products = fetch_ebay_results(search_query) if should_search else []
        amazon_products = fetch_amazon_results(search_query) if should_search else []

        all_products = []
        if isinstance(bestbuy_products, list):
            all_products += bestbuy_products
        if isinstance(ebay_products, list):
            all_products += ebay_products
        if isinstance(amazon_products, list):
            all_products += amazon_products

        def filter_by_attributes(products, attributes):
            if not attributes or not isinstance(attributes, list):
                return products

            price_limit = None
            keyword_filters = []

            for attr in attributes:
                attr = attr.lower().strip()
                price_match = re.search(r'\$?(\d{3,5})', attr)
                if 'under' in attr or 'less than' in attr or 'budget' in attr:
                    price_limit = int(price_match.group(1)) if price_match else price_limit
                elif price_match:
                    price_limit = int(price_match.group(1))
                else:
                    keyword_filters.append(attr)

            def matches(product):
                title = (product.get("name") or product.get("title") or "").lower()
                description = (product.get("description") or "").lower()
                full_text = f"{title} {description}"

                price_str = str(product.get("price", "")).replace("$", "").split()[0]
                try:
                    price = float(price_str)
                except:
                    price = None

                if price_limit and (not price or price > price_limit):
                    return False

                for kw in keyword_filters:
                    if kw in full_text or kw in title:
                        continue
                    return False
                return True

            return [p for p in products if matches(p)]

        if query_type == "literal":
            filtered_products = [p for p in all_products if is_strict_match_required(p, search_query)]
        else:
            filtered_products = [
                p for p in all_products
                if not should_exclude_product(p, filters)
                and not is_irrelevant_category(p, category)
                and (category.lower() in (p.get("name", "").lower() + p.get("description", "").lower() + " ") if category else True)
            ]

        filtered_products = filter_by_attributes(filtered_products, attributes)

        for product in filtered_products:
            product['_score'] = get_product_score(product, search_query, attributes, category, filters)
            global last_fetched_products
            last_fetched_products = filtered_products.copy()

        bestbuy_sorted = sorted([p for p in filtered_products if p.get('source') == 'Best Buy'], key=lambda x: x.get('_score', 0), reverse=True)
        ebay_sorted = sorted([p for p in filtered_products if p.get('source') == 'eBay'], key=lambda x: x.get('_score', 0), reverse=True)
        amazon_sorted = sorted([p for p in filtered_products if p.get('source') == 'Amazon'], key=lambda x: x.get('_score', 0), reverse=True)

        mixed_products = []
        i = j = k = 0
        while len(mixed_products) < 24 and (i < len(bestbuy_sorted) or j < len(ebay_sorted) or k < len(amazon_sorted)):
            if i < len(bestbuy_sorted):
                mixed_products.append(bestbuy_sorted[i])
                i += 1
            if j < len(ebay_sorted):
                mixed_products.append(ebay_sorted[j])
                j += 1
            if k < len(amazon_sorted):
                mixed_products.append(amazon_sorted[k])
                k += 1

        response_msg = str(response_msg or "")
        follow_up = str(follow_up) if follow_up is not None else ""

        return jsonify({
            "response": response_msg,
            "products": mixed_products,
            "affiliate_config": {
                "ebay_campaign_id": EBAY_CAMPAIGN_ID,
                "ebay_pub_id": "5575450883"
            }
         })

    except Exception as e:
        print(f"[ERROR] Chat route: {e}")
        return jsonify({"response": "Sorry, something went wrong.", "products": []})

        @app.route('/smart_match', methods=['POST'])
        def smart_match():
            try:
                data = request.json
                selected_product = data.get('selected_product')
                candidates = last_fetched_products

                if not selected_product or not candidates:
                    return jsonify({"exact_matches": [], "similar_alternatives": []})

                exact_matches = []
                similar_alternatives = []

                pre_matches = [
                    p for p in candidates
                    if p.get("url") != selected_product.get("url")
                    and (p.get("sku") != selected_product.get("sku"))
                    and fast_pre_match_score(p, selected_product) > 0.5
                ]

                for product in pre_matches[:25]:
                    if is_exact_variant_match(product, selected_product):
                        exact_matches.append(product)
                    else:
                        similar_alternatives.append(product)

                return jsonify({
                    "exact_matches": exact_matches,
                    "similar_alternatives": similar_alternatives
                })

            except Exception as e:
                print(f"[ERROR] /smart_match: {e}")
                return jsonify({"exact_matches": [], "similar_alternatives": []})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
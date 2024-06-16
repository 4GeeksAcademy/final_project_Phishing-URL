import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
import pickle
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

# Load URL dataset
@st.cache_data
def load_data():
    # Intentar con diferentes codificaciones si utf-8 falla
    try:
        data = pd.read_csv('/workspaces/proyecto-final-Phishing-URL/data/raw/filtered_urls.csv', encoding='utf-8')
    except UnicodeDecodeError:
        data = pd.read_csv('/workspaces/proyecto-final-Phishing-URL/data/raw/filtered_urls.csv', encoding='latin1')
    return data

data = load_data()

# URL list from 'filtered_urls' file
known_urls = data['URL'].tolist()

# Levenshtein distance function
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

# SequenceMatcher for similarity function
def similarity_ratio(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()

# URL Similarity index
def url_similarity_index(url, known_urls):
    min_distance = float('inf')
    for known_url in known_urls:
        distance = levenshtein_distance(url, known_url)
        normalized_distance = distance / max(len(url), len(known_url))
        if normalized_distance < min_distance:
            min_distance = normalized_distance
    return 1 - min_distance

# URL features function
def extraer_caracteristicas(url):
    try:
        response = requests.get(url)
        page_content = response.content
    except Exception as e:
        st.error(f"Error making the request: {e}")
        return None
    
    soup = BeautifulSoup(page_content, 'html.parser')
    title = soup.title.string if soup.title else ''
    domain = urlparse(url).netloc

    #URLSimilarityIndex = url_similarity_index(url, known_urls)
    #CharContinuationRate = len(re.findall(r'[a-zA-Z]{2,}', url)) / len(url)
    #URLCharProb = 0  
    #LetterRatioInURL = sum(c.isalpha() for c in url) / len(url)
    #DegitRatioInURL = sum(c.isdigit() for c in url) / len(url)
    #NoOfOtherSpecialCharsInURL = len(re.findall(r'[^a-zA-Z0-9]', url))
    #SpacialCharRatioInURL = NoOfOtherSpecialCharsInURL / len(url)
    #IsHTTPS = 1 if urlparse(url).scheme == 'https' else 0
    #HasTitle = 1 if soup.title else 0
    #DomainTitleMatchScore = 0
    #URLTitleMatchScore = 0
    #HasFavicon = 1 if soup.find('link', rel='icon') or soup.find('link', rel='shortcut icon') else 0
    #Robots = 1 if requests.get(urlparse(url).scheme + "://" + urlparse(url).netloc + "/robots.txt").status_code == 200 else 0
    #IsResponsive = 0
    #HasDescription = 1 if soup.find('meta', attrs={'name': 'description'}) else 0
    #social_net_keywords = ['facebook', 'twitter', 'instagram', 'linkedin', 'youtube']
    #HasSocialNet = 1 if any(keyword in page_content.decode().lower() for keyword in social_net_keywords) else 0
    #HasSubmitButton = 1 if soup.find('input', type='submit') or soup.find('button', type='submit') else 0
    #HasHiddenFields = 1 if soup.find('input', type='hidden') else 0
    #pay_keywords = ['paypal', 'visa', 'mastercard', 'payment', 'checkout']
    #Pay = 1 if any(keyword in page_content.decode().lower() for keyword in pay_keywords) else 0
    #HasCopyrightInfo = 1 if '©' in page_content.decode() or 'copyright' in page_content.decode().lower() else 0
    #NoOfJS = len(soup.find_all('script'))

    URLSimilarityIndex = url_similarity_index(url, known_urls)
    CharContinuationRate = len(re.findall(r'[a-zA-Z]{2,}', url)) / len(url)
    URLCharProb = 0  
    LetterRatioInURL = sum(c.isalpha() for c in url) / len(url)
    DegitRatioInURL = sum(c.isdigit() for c in url) / len(url)
    NoOfOtherSpecialCharsInURL = len(re.findall(r'[^a-zA-Z0-9]', url))
    SpacialCharRatioInURL = NoOfOtherSpecialCharsInURL / len(url)
    IsHTTPS = 1 if urlparse(url).scheme == 'https' else 0
    HasTitle = 1 if soup.title else 0
    DomainTitleMatchScore = similarity_ratio(domain, title)
    URLTitleMatchScore = similarity_ratio(url, title)
    HasFavicon = 1 if soup.find('link', rel='icon') or soup.find('link', rel='shortcut icon') else 0
    Robots = 1 if requests.get(urlparse(url).scheme + "://" + urlparse(url).netloc + "/robots.txt").status_code == 200 else 0
    IsResponsive = 1 if soup.find('meta', attrs={'name': 'viewport'}) else 0
    HasDescription = 1 if soup.find('meta', attrs={'name': 'description'}) else 0
    social_net_keywords = ['facebook', 'twitter', 'instagram', 'linkedin', 'youtube']
    HasSocialNet = 1 if any(keyword in page_content.decode().lower() for keyword in social_net_keywords) else 0
    HasSubmitButton = 1 if soup.find('input', type='submit') or soup.find('button', type='submit') else 0
    HasHiddenFields = 1 if soup.find('input', type='hidden') else 0
    pay_keywords = ['paypal', 'visa', 'mastercard', 'payment', 'checkout']
    Pay = 1 if any(keyword in page_content.decode().lower() for keyword in pay_keywords) else 0
    HasCopyrightInfo = 1 if '©' in page_content.decode() or 'copyright' in page_content.decode().lower() else 0
    NoOfJS = len(soup.find_all('script'))

    return [URLSimilarityIndex, CharContinuationRate, URLCharProb, LetterRatioInURL, DegitRatioInURL,
            NoOfOtherSpecialCharsInURL, SpacialCharRatioInURL, IsHTTPS, HasTitle, DomainTitleMatchScore,
            URLTitleMatchScore, HasFavicon, Robots, IsResponsive, HasDescription, HasSocialNet,
            HasSubmitButton, HasHiddenFields, Pay, HasCopyrightInfo, NoOfJS]

# Load Model
try:
    with open('/workspaces/proyecto-final-Phishing-URL/models/nbayes_gaussian_opt_var_smoothing-1e-05.sav', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Create Streamlit App
st.title('Phishing URL Detection')

url = st.text_input('Enter a URL:')

if st.button('Predict'):
    if url:
        caracteristicas = extraer_caracteristicas(url)
        if caracteristicas:
            caracteristicas = np.array(caracteristicas).reshape(1, -1)
            y_pred = model.predict(caracteristicas)[0]
            if y_pred: 
                st.success('This is a legitimate URL.')
            else:
                st.error('This is a phishing URL.')
        else:
            st.error('Could not extract features from the URL.')
    else:
        st.error('Please, enter a URL.')
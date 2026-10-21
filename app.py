import math
import re
import unicodedata
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title='AXIOMIS V3.2 | Live Fixtures', page_icon='⚽', layout='wide')
st.markdown('''<style>.stApp{background:#0d1929;color:#e9f2f7}h1,h2,h3{color:#e9f2f7}div[data-testid="stMetric"]{background:#15283b;border:1px solid #294457;padding:12px;border-radius:10px}div[data-testid="stMetricValue"]{color:#61e6c5}</style>''', unsafe_allow_html=True)
st.title('◈ AXIOMIS V3.2')
st.caption('LIVE FIXTURES · Gerçek fikstür + geçmiş sonuçlara dayalı Poisson modeli')
TZ = ZoneInfo('Europe/Istanbul')
# API-Football league IDs; separate from historical CSV league codes.
LEAGUES = {
    'Türkiye Süper Lig': ('tr.1', 203),
    'İngiltere Premier League': ('en.1', 39),
    'Almanya Bundesliga': ('de.1', 78),
    'İspanya La Liga': ('es.1', 140),
    'İtalya Serie A': ('it.1', 135),
    'Fransa Ligue 1': ('fr.1', 61),
}
ALIASES = {
    'manchester united fc': 'manchester united',
    'manchester city fc': 'manchester city',
    'tottenham hotspur fc': 'tottenham hotspur',
    'fc bayern munich': 'bayern munchen',
    'bayern munich': 'bayern munchen',
    'bayern munchen': 'bayern munchen',
    'paris saint germain': 'paris saint germain',
    'psg': 'paris saint germain',
    'inter milan': 'internazionale',
    'inter': 'internazionale',
    'ac milan': 'milan',
    'athletic bilbao': 'athletic club',
    'atletico madrid': 'atletico de madrid',
    'besiktas jk': 'besiktas',
    'fenerbahce sk': 'fenerbahce',
    'galatasaray sk': 'galatasaray',
    'trabzonspor as': 'trabzonspor',
}

def norm(value):
    value = str(value).casefold().replace('ı', 'i')
    value = ''.join(c for c in unicodedata.normalize('NFKD', value) if not unicodedata.combining(c))
    value = re.sub(r'[^a-z0-9 ]+', ' ', value)
    value = re.sub(r'\s+', ' ', value).strip()
    return ALIASES.get(value, value)

@st.cache_data(ttl=3600)
def load_results():
    path = Path(__file__).with_name('matches.csv')
    if not path.exists():
        return None, 'matches.csv bulunamadı.'
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
        required = ['date', 'league', 'home_team', 'away_team', 'home_goals', 'away_goals']
        if any(c not in df.columns for c in required):
            return None, 'matches.csv gerekli sütunları içermiyor.'
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in ['home_goals', 'away_goals']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=required).copy()
        df['hn'] = df.home_team.map(norm)
        df['an'] = df.away_team.map(norm)
        return df.sort_values('date'), None
    except Exception as exc:
        return None, f'Sonuç verisi okunamadı: {exc}'

# Each league/date window is one API request, cached for one hour.
# Errors are cached briefly too, avoiding repeated requests on Streamlit reruns.
@st.cache_data(ttl=900, show_spinner=False)
def fetch_fixtures(league_id, season, start_day, end_day, api_key):
    try:
        response = requests.get(
            'https://v3.football.api-sports.io/fixtures',
            params={'league': league_id, 'season': season, 'from': start_day,
                    'to': end_day, 'timezone': 'Europe/Istanbul'},
            headers={'x-apisports-key': api_key}, timeout=20,
        )
        if response.status_code == 429:
            return [], 'Günlük veya dakikalık API istek sınırına ulaşıldı (429).'
        if response.status_code in (401, 403):
            return [], 'API anahtarı veya abonelik yetkisi reddedildi (401/403).'
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return [], 'API beklenmeyen biçimde yanıt verdi.'
        errors = payload.get('errors')
        if errors:
            return [], 'API uyarısı: ' + str(errors)[:450]
        return payload.get('response') or [], None
    except (requests.RequestException, ValueError) as exc:
        return [], f'API bağlantı hatası: {type(exc).__name__}'


def parse_fixture(event):
    fixture = event.get('fixture') or {}
    teams = event.get('teams') or {}
    home = (teams.get('home') or {}).get('name')
    away = (teams.get('away') or {}).get('name')
    stamp = fixture.get('date')
    if not home or not away or not stamp:
        return None
    try:
        kickoff = datetime.fromisoformat(stamp.replace('Z', '+00:00')).astimezone(TZ)
    except ValueError:
        return None
    return {'home': home, 'away': away, 'kickoff': kickoff,
            'id': fixture.get('id'), 'status': (fixture.get('status') or {}).get('short', '')}

def match_team(name, candidates):
    needle = norm(name)
    exact = [x for x in candidates if norm(x) == needle]
    if len(exact) == 1:
        return exact[0], 'exact'
    # Conservative fuzzy matching; ambiguous names are not matched.
    ranked = sorted(((SequenceMatcher(None, needle, norm(x)).ratio(), x) for x in candidates), reverse=True)
    if ranked and ranked[0][0] >= 0.91 and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 0.08):
        return ranked[0][1], 'approximate'
    return None, 'unmatched'

def model(df, league, home, away, cutoff):
    history = df[(df.league == league) & (df.date < pd.Timestamp(cutoff.date()))].copy()
    if len(history) < 40:
        return None, f'Lig için geçmiş veri yetersiz ({len(history)} maç).'
    H = history[history.hn == norm(home)]
    A = history[history.an == norm(away)]
    if len(H) < 5 or len(A) < 5:
        return None, f'Takım verisi yetersiz (evinde {len(H)}, deplasmanda {len(A)} maç).'
    avg_h = (history.home_goals.sum() + 1.45 * 20) / (len(history) + 20)
    avg_a = (history.away_goals.sum() + 1.15 * 20) / (len(history) + 20)
    hgf = (H.home_goals.sum() + avg_h * 8) / (len(H) + 8)
    hga = (H.away_goals.sum() + avg_a * 8) / (len(H) + 8)
    agf = (A.away_goals.sum() + avg_a * 8) / (len(A) + 8)
    aga = (A.home_goals.sum() + avg_h * 8) / (len(A) + 8)
    lh = max(.1, min(5, hgf * aga / avg_h))
    la = max(.1, min(5, agf * hga / avg_a))
    ph = [math.exp(-lh) * lh**i / math.factorial(i) for i in range(16)]
    pa = [math.exp(-la) * la**i / math.factorial(i) for i in range(16)]
    scores = [(i, j, ph[i]*pa[j]) for i in range(16) for j in range(16)]
    def p(test):
        return sum(v for i, j, v in scores if test(i, j))
    return {'lh': lh, 'la': la, '1': p(lambda i,j:i>j), 'X': p(lambda i,j:i==j),
            '2': p(lambda i,j:i<j), '1.5 Üst': p(lambda i,j:i+j>=2),
            '2.5 Üst': p(lambda i,j:i+j>=3), '3.5 Üst': p(lambda i,j:i+j>=4),
            'KG Var': p(lambda i,j:i>0 and j>0), 'n': len(history),
            'top': sorted(scores, key=lambda x:x[2], reverse=True)[:5]}, None

with st.sidebar:
    st.header('Güncel fikstür')
    league_label = st.selectbox('Lig', list(LEAGUES))
    start = st.date_input('Başlangıç tarihi', datetime.now(TZ).date())
    days = st.slider('Kaç gün listelensin?', 1, 7, 3)
    only_future = st.checkbox('Yalnızca başlamamış maçlar', value=True)
    st.caption('Kaynak: API-Football · Saatler Türkiye saatidir. Fikstür sorguları önbelleğe alınır.')

try:
    api_key = st.secrets.get('API_FOOTBALL_KEY', '')
except Exception:
    api_key = ''
if not api_key or api_key == 'BURAYA_KENDI_API_ANAHTARINI_YAPISTIR':
    st.error('API_FOOTBALL_KEY bulunamadı. Streamlit Cloud → App settings → Secrets bölümüne ekleyin.')
    st.stop()

results, result_error = load_results()
if result_error:
    st.warning(result_error + ' Fikstür gösterilebilir ancak tahmin üretilemez.')

league_code, league_id = LEAGUES[league_label]
end = start + timedelta(days=days - 1)
# European football seasons generally begin in summer. A cross-season query
# would need two API requests; show an explicit message instead of silent loss.
season = start.year if start.month >= 7 else start.year - 1
end_season = end.year if end.month >= 7 else end.year - 1
if season != end_season:
    st.warning('Seçilen tarih aralığı iki futbol sezonuna uzanıyor. Tek sezon içinde bir aralık seçin.')
    st.stop()

with st.spinner('API-Football fikstürü alınıyor...'):
    raw_events, api_error = fetch_fixtures(league_id, season, start.isoformat(), end.isoformat(), api_key)
if api_error:
    st.error(api_error)
    st.info('Bu durum maç olmadığı anlamına gelmez. Ücretsiz planın sezon erişimi ve kotasını API-Sports panelinden kontrol edin.')
    st.stop()

now = datetime.now(TZ)
league_events = []
seen = set()
for raw in raw_events:
    e = parse_fixture(raw)
    if not e or e['id'] in seen:
        continue
    if only_future and (e['kickoff'] <= now or e['status'] not in ('NS', 'TBD')):
        continue
    seen.add(e['id'])
    league_events.append(e)
league_events.sort(key=lambda x: x['kickoff'])
st.subheader(f'{league_label} · {len(league_events)} maç')
if not league_events:
    st.info('Bu tarih aralığında API tarafından maç döndürülmedi. Bu, kesinlikle maç olmadığı anlamına gelmez.')
    st.stop()

labels = [f"{e['kickoff']:%d.%m %H:%M} · {e['home']} — {e['away']}" for e in league_events]
idx = st.selectbox('Analiz edilecek fikstür maçı', range(len(labels)), format_func=lambda i: labels[i])
e = league_events[idx]
st.markdown(f"### {e['home']} — {e['away']}")
st.caption(f"Başlama: {e['kickoff']:%d.%m.%Y %H:%M} (Türkiye saati) · Kaynak: API-Football")
if results is None:
    st.stop()
league_history = results[results.league == league_code]
candidates = sorted(set(league_history.home_team) | set(league_history.away_team))
home_match, home_kind = match_team(e['home'], candidates)
away_match, away_kind = match_team(e['away'], candidates)
if not home_match or not away_match:
    st.warning('Fikstür bulundu fakat takım adları geçmiş veriyle güvenle eşleştirilemedi. Yanıltıcı tahmin göstermiyoruz.')
    st.caption(f"Eşleştirme: ev sahibi {home_match or 'bulunamadı'}, deplasman {away_match or 'bulunamadı'}")
    st.stop()
if home_kind != 'exact' or away_kind != 'exact':
    st.info(f'Yaklaşık isim eşleştirmesi: {e["home"]} → {home_match}; {e["away"]} → {away_match}.')
values, reason = model(results, league_code, home_match, away_match, e['kickoff'])
if reason:
    st.warning('Fikstür bulundu fakat tahmin üretilemedi: ' + reason)
    st.stop()
a, b, c = st.columns(3)
a.metric('Beklenen ev golü', f"{values['lh']:.2f}")
b.metric('Beklenen deplasman golü', f"{values['la']:.2f}")
c.metric('Beklenen toplam gol', f"{values['lh']+values['la']:.2f}")
st.subheader('1-X-2 olasılıkları')
a, b, c = st.columns(3)
for col, market in zip((a, b, c), ('1', 'X', '2')):
    col.metric(market, f"%{100*values[market]:.1f}")
st.subheader('Gol pazarları')
cols = st.columns(4)
for col, market in zip(cols, ('1.5 Üst', '2.5 Üst', '3.5 Üst', 'KG Var')):
    col.metric(market, f"%{100*values[market]:.1f}")
st.progress(float(values['2.5 Üst']), text='2.5 Üst model olasılığı')
st.subheader('En olası skorlar')
st.dataframe(pd.DataFrame([{'Skor': f'{i}-{j}', 'Olasılık': f'%{p*100:.1f}'} for i,j,p in values['top']]), hide_index=True)
st.caption(f"Model geçmişi: {values['n']} lig maçı. Yalnızca maç gününden önceki sonuçlar kullanıldı. CSV son kayıt: {results.date.max():%d.%m.%Y}.")
st.info('V3.2 beta: Poisson modeli henüz geçmişe dönük doğrulanmadı. Güncel kadro, sakatlık ve bahis oranı verisi kullanılmaz. CSV verileri eksik olabilir.')

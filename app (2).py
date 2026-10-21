import math
import re
import unicodedata
from datetime import datetime, timedelta, timezone
import time
from threading import Lock
from difflib import SequenceMatcher
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title='AXIOMIS V3.1 | Live Fixtures', page_icon='⚽', layout='wide')
st.markdown('''<style>.stApp{background:#0d1929;color:#e9f2f7}h1,h2,h3{color:#e9f2f7}div[data-testid="stMetric"]{background:#15283b;border:1px solid #294457;padding:12px;border-radius:10px}div[data-testid="stMetricValue"]{color:#61e6c5}</style>''', unsafe_allow_html=True)
st.title('◈ AXIOMIS V3.1')
st.caption('LIVE FIXTURES · Gerçek fikstür + geçmiş sonuçlara dayalı Poisson modeli')
TZ = ZoneInfo('Europe/Istanbul')
LEAGUES = {
    'Türkiye Süper Lig': ('tr.1', ('turkish super lig', 'super lig', 'süper lig')),
    'İngiltere Premier League': ('en.1', ('english premier league', 'premier league')),
    'Almanya Bundesliga': ('de.1', ('german bundesliga', 'bundesliga')),
    'İspanya La Liga': ('es.1', ('spanish la liga', 'la liga', 'primera division')),
    'İtalya Serie A': ('it.1', ('italian serie a', 'serie a')),
    'Fransa Ligue 1': ('fr.1', ('french ligue 1', 'ligue 1')),
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

# Shared circuit breaker: after a 429, every session stops requesting the API.
@st.cache_resource
def api_guard():
    return {'lock': Lock(), 'blocked_until': 0.0, 'reason': ''}

class RateLimited(Exception):
    pass

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_day(day):
    guard = api_guard()
    with guard['lock']:
        if time.monotonic() < guard['blocked_until']:
            raise RateLimited('Sağlayıcı istek sınırı nedeniyle geçici olarak beklemede.')
        url = 'https://www.thesportsdb.com/api/v1/json/123/eventsday.php'
        try:
            response = requests.get(url, params={'d': day, 's': 'Soccer'}, timeout=18,
                                    headers={'User-Agent': 'AXIOMIS-V3.1/1.0'})
        except requests.RequestException:
            # Do not hammer a struggling service on repeated reruns.
            guard['blocked_until'] = time.monotonic() + 300
            raise
        if response.status_code == 429:
            # Retry-After may be absent or unreliable; use at least 60 minutes.
            try:
                retry = int(response.headers.get('Retry-After', '3600'))
            except (ValueError, TypeError):
                retry = 3600
            guard['blocked_until'] = time.monotonic() + max(3600, min(retry, 86400))
            guard['reason'] = '429 Too Many Requests'
            raise RateLimited('TheSportsDB 429: İstek limiti doldu. En az 60 dakika yeni istek yapılmayacak.')
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError('Beklenmeyen fikstür yanıtı')
        return payload.get('events') or []

def league_match(raw, league_label):
    n = norm(raw)
    aliases = LEAGUES[league_label][1]
    # TheSportsDB often prefixes leagues with the country.
    return any(n == norm(a) or n.endswith(' ' + norm(a)) for a in aliases)

def parse_kickoff(event):
    stamp = event.get('strTimestamp')
    if stamp:
        try:
            return datetime.fromisoformat(stamp.replace('Z', '+00:00')).astimezone(TZ)
        except ValueError:
            pass
    d = event.get('dateEvent')
    t = event.get('strTime') or '00:00:00'
    try:
        return datetime.fromisoformat(f'{d}T{t[:8]}').replace(tzinfo=timezone.utc).astimezone(TZ)
    except (ValueError, TypeError):
        return None

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
    st.caption('Önbellek 6 saat geçerlidir. Sayfayı yenilemek sağlayıcıya zorla istek göndermez.')
    st.caption('Fikstür: TheSportsDB. Saatler Türkiye saatidir. Veri kaynağı bazı ligleri/günleri eksik sunabilir.')

results, result_error = load_results()
if result_error:
    st.warning(result_error + ' Fikstür gösterilir ancak tahmin yapılamaz.')

items = []
errors = []
limited = False
# Limit cold API requests per page load. Previously 7 days meant 7 immediate requests.
# A user can move the start date to see a different window.
requested_days = min(days, 2)
if days > requested_days:
    st.info(f'İstek sınırını korumak için bu açılışta ilk {requested_days} gün sorgulanıyor. Diğer günler için başlangıç tarihini ilerletin.')
with st.spinner('Gerçek fikstür sorgulanıyor...'):
    for n in range(requested_days):
        day = (start + timedelta(days=n)).isoformat()
        try:
            items.extend(fetch_day(day))
        except RateLimited as exc:
            limited = True
            errors.append(str(exc))
            break  # Never continue sending requests after 429.
        except (requests.RequestException, ValueError) as exc:
            errors.append(f'{day}: Fikstür alınamadı ({type(exc).__name__}).')
if errors:
    st.warning(' | '.join(errors[:3]))
if limited and not items:
    st.error('Fikstür servisi şu anda istek kabul etmiyor. Bu durum maç olmadığı anlamına gelmez. Daha sonra tekrar deneyin.')
    st.stop()

league_events = []
seen = set()
now = datetime.now(TZ)
for e in items:
    if not league_match(e.get('strLeague') or '', league_label):
        continue
    home, away = e.get('strHomeTeam'), e.get('strAwayTeam')
    kickoff = parse_kickoff(e)
    if not home or not away or not kickoff:
        continue
    if only_future and kickoff <= now:
        continue
    key = e.get('idEvent') or (kickoff.isoformat(), home, away)
    if key in seen:
        continue
    seen.add(key)
    league_events.append({'home':home, 'away':away, 'kickoff':kickoff, 'id':key})
league_events.sort(key=lambda x:x['kickoff'])
st.subheader(f'{league_label} · {len(league_events)} maç')
if not league_events:
    st.info('Sorgulanabilen günlerde bu lig için maç bulunamadı. Sağlayıcının eksik kapsamı veya geçici kısıtlaması nedeniyle maçlar görünmüyor olabilir.')
    st.stop()

labels = [f"{e['kickoff']:%d.%m %H:%M} · {e['home']} — {e['away']}" for e in league_events]
idx = st.selectbox('Analiz edilecek gerçek fikstür maçı', range(len(labels)), format_func=lambda i:labels[i])
e = league_events[idx]
st.markdown(f"### {e['home']} — {e['away']}")
st.caption(f"Başlama: {e['kickoff']:%d.%m.%Y %H:%M} (Türkiye saati) · Kaynak: TheSportsDB")
if results is None:
    st.stop()
league_code = LEAGUES[league_label][0]
league_history = results[results.league == league_code]
candidates = sorted(set(league_history.home_team) | set(league_history.away_team))
home_match, home_kind = match_team(e['home'], candidates)
away_match, away_kind = match_team(e['away'], candidates)
if not home_match or not away_match:
    st.warning('Fikstür bulundu; ancak takım adları geçmiş sonuç verisiyle güvenle eşleştirilemedi. Yanıltıcı tahmin göstermiyoruz.')
    st.caption(f"Eşleştirme: ev sahibi {home_match or 'bulunamadı'}, deplasman {away_match or 'bulunamadı'}")
    st.stop()
if home_kind != 'exact' or away_kind != 'exact':
    st.info(f'Yaklaşık isim eşleştirmesi: {e["home"]} → {home_match}; {e["away"]} → {away_match}. Lütfen kontrol edin.')
values, reason = model(results, league_code, home_match, away_match, e['kickoff'])
if reason:
    st.warning('Fikstür bulundu fakat tahmin üretilemedi: ' + reason)
    st.stop()
a,b,c = st.columns(3)
a.metric('Beklenen ev golü', f"{values['lh']:.2f}")
b.metric('Beklenen deplasman golü', f"{values['la']:.2f}")
c.metric('Beklenen toplam gol', f"{values['lh']+values['la']:.2f}")
st.subheader('1-X-2 olasılıkları')
a,b,c=st.columns(3)
for col, market in zip((a,b,c),('1','X','2')):
    col.metric(market, f"%{100*values[market]:.1f}")
st.subheader('Gol pazarları')
cols=st.columns(4)
for col, market in zip(cols, ('1.5 Üst','2.5 Üst','3.5 Üst','KG Var')):
    col.metric(market, f"%{100*values[market]:.1f}")
st.progress(float(values['2.5 Üst']), text='2.5 Üst model olasılığı')
st.subheader('En olası skorlar')
st.dataframe(pd.DataFrame([{'Skor':f'{i}-{j}', 'Olasılık':f'%{p*100:.1f}'} for i,j,p in values['top']]), hide_index=True)
st.caption(f"Model geçmişi: {values['n']} lig maçı. Yalnızca maç gününden önceki sonuçlar kullanıldı. Yerel CSV en son {results.date.max():%d.%m.%Y} tarihine kadar kayıt içeriyor.")
st.info('V3 beta: Bu bir Poisson tahmin modelidir, doğrulanmış başarı oranı yoktur. Canlı bahis oranı, sakatlık, kadro veya güncel skor verisi içermez. Geçmiş veri eksikliği tahminleri etkileyebilir.')

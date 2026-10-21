import math
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title='AXIOMIS | Football Intelligence',page_icon='⚽',layout='wide')
st.markdown('''<style>.stApp{background:#0d1929;color:#e9f2f7}h1,h2,h3{color:#e9f2f7}.stMetric{background:#15283b;border:1px solid #294457;padding:16px;border-radius:12px}div[data-testid="stMetricValue"]{color:#61e6c5}</style>''',unsafe_allow_html=True)
st.title('◈ AXIOMIS')
st.caption('INTELLIGENCE BEYOND ODDS · Futbol gol olasılıkları')
@st.cache_data
def load():
    df=pd.read_csv(Path(__file__).with_name('matches.csv'),encoding='utf-8-sig',parse_dates=['date'])
    df=df.dropna(subset=['date','league','home_team','away_team','home_goals','away_goals']).copy()
    df[['home_goals','away_goals']]=df[['home_goals','away_goals']].astype(int)
    return df.sort_values('date')
df=load()
with st.sidebar:
    st.header('Maç seçimi')
    league=st.selectbox('Lig',sorted(df.league.unique()))
    ld=df[df.league==league]
    teams=sorted(set(ld.home_team)|set(ld.away_team))
    home=st.selectbox('Ev sahibi',teams)
    away=st.selectbox('Deplasman',[t for t in teams if t!=home])
    asof=st.date_input('Veri kesim tarihi',value=df.date.max().date(),min_value=df.date.min().date(),max_value=df.date.max().date())
    st.caption('Yalnızca seçilen tarihe kadar oynanan maçlar kullanılır. Bu ekran yaklaşan maç fikstürü sağlamaz.')
ld=ld[ld.date.dt.date<=asof]
if ld.empty: st.warning('Bu tarihe kadar lig verisi yok.');st.stop()
H=ld[ld.home_team==home];A=ld[ld.away_team==away]
if len(H)<5 or len(A)<5 or len(ld)<40:
    st.warning('Bu eşleşme için geçmiş veri yetersiz. Güvenilir tahmin hesaplanmadı.');st.stop()
avg_h=(ld.home_goals.sum()+1.45*20)/(len(ld)+20)
avg_a=(ld.away_goals.sum()+1.15*20)/(len(ld)+20)
hgf=(H.home_goals.sum()+avg_h*8)/(len(H)+8)
hga=(H.away_goals.sum()+avg_a*8)/(len(H)+8)
agf=(A.away_goals.sum()+avg_a*8)/(len(A)+8)
aga=(A.home_goals.sum()+avg_h*8)/(len(A)+8)
lh=max(.1,min(5,hgf*aga/avg_h));la=max(.1,min(5,agf*hga/avg_a))
ph=[math.exp(-lh)*lh**i/math.factorial(i) for i in range(16)]
pa=[math.exp(-la)*la**i/math.factorial(i) for i in range(16)]
score=[(i,j,ph[i]*pa[j]) for i in range(16) for j in range(16)]
def prob(test): return sum(p for i,j,p in score if test(i,j))
p1=prob(lambda i,j:i>j);px=prob(lambda i,j:i==j);p2=prob(lambda i,j:i<j)
st.subheader(f'{home}  —  {away}')
a,b,c=st.columns(3)
a.metric('Beklenen ev sahibi golü',f'{lh:.2f}')
b.metric('Beklenen deplasman golü',f'{la:.2f}')
c.metric('Beklenen toplam gol',f'{lh+la:.2f}')
st.subheader('Gol pazarları')
markets={'0,5 Üst':prob(lambda i,j:i+j>=1),'1,5 Üst':prob(lambda i,j:i+j>=2),'2,5 Üst':prob(lambda i,j:i+j>=3),'3,5 Üst':prob(lambda i,j:i+j>=4),'KG Var':prob(lambda i,j:i>0 and j>0)}
cols=st.columns(len(markets))
for col,(name,p) in zip(cols,markets.items()):col.metric(name,f'%{p*100:.1f}')
st.progress(float(markets['2,5 Üst']),text='2,5 Üst model olasılığı')
st.subheader('Maç sonucu olasılıkları')
st.bar_chart(pd.DataFrame({'Olasılık (%)':[p1*100,px*100,p2*100]},index=['1','X','2']))
st.subheader('En olası 5 skor')
top=sorted(score,key=lambda x:x[2],reverse=True)[:5]
st.dataframe(pd.DataFrame([{'Skor':f'{i}-{j}','Olasılık':f'%{p*100:.1f}'} for i,j,p in top]),hide_index=True,use_container_width=True)
st.caption(f'Kaynak: OpenFootball arşivinden alınan yerel CSV · {len(ld):,} lig maçı · Ev sahibi için {len(H)}, deplasman için {len(A)} geçmiş maç. Veriler eksik sezonlar içerebilir.')
st.info('Bu olasılıklar geçmiş maç skorlarından türetilen bağımsız Poisson modelidir. Bahis oranları, sakatlıklar ve güncel kadrolar kullanılmaz. İlk yarı ve korner modeli henüz yoktur. Finansal kazanç garantisi değildir.')

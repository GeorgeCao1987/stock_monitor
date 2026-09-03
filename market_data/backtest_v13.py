from pathlib import Path
import json
import numpy as np
import pandas as pd

from config import PCB_MEMBERS

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
TARGET = "002916"


def load_a(symbol, source="eastmoney"):
    p = DATA / f"a_{symbol.replace('.', '_')}_{source}.csv"
    if not p.exists(): return pd.DataFrame()
    x = pd.read_csv(p); x["ts"] = pd.to_datetime(x["ts"])
    return x.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


def add_target_features(df):
    x = df.copy(); day = x.ts.dt.date
    x["ret1"] = x.close.pct_change()
    x["typical"] = (x.high + x.low + x.close) / 3
    x["cum_pv"] = (x.typical * x.volume).groupby(day).cumsum(); x["cum_v"] = x.volume.groupby(day).cumsum().replace(0, np.nan)
    x["vwap"] = x.cum_pv / x.cum_v
    denominator = x.amount.replace(0, np.nan) if "amount" in x and x.amount.notna().any() else x.volume.replace(0, np.nan)
    x["eff"] = x.ret1.abs() / denominator
    # Mechanical candidates: current completed 5m bar may be used; no later bar enters the trigger.
    x["prior_high3"] = x.high.shift(1).rolling(3).max(); x["prior_low3"] = x.low.shift(1).rolling(3).min()
    x["prior_eff3"] = x.eff.shift(1).rolling(3).mean()
    x["high_candidate"] = (x.high >= x.prior_high3) & (x.eff < x.prior_eff3 * .85)
    x["low_candidate"] = (x.low <= x.prior_low3) & (x.eff < x.prior_eff3 * .85)
    x["hh"] = x.high > x.high.shift(1); x["hl"] = x.low > x.low.shift(1)
    x["lh"] = x.high < x.high.shift(1); x["ll"] = x.low < x.low.shift(1)
    x["up_structure"] = (x.hh.astype(int).rolling(4).sum() + x.hl.astype(int).rolling(4).sum()) / 8
    x["down_structure"] = (x.lh.astype(int).rolling(4).sum() + x.ll.astype(int).rolling(4).sum()) / 8
    x["above_vwap"] = x.close > x.vwap
    return x


def build_pcb_context():
    parts=[]
    for s in PCB_MEMBERS:
        z=load_a(s)
        if z.empty: continue
        z=z[["ts","close"]].copy(); z["day_open"] = z.close.groupby(z.ts.dt.date).transform("first")
        z[f"r_{s}"] = z.close / z.day_open - 1; parts.append(z[["ts",f"r_{s}"]].set_index("ts"))
    if not parts: return pd.DataFrame()
    z=pd.concat(parts,axis=1).sort_index(); members=[c for c in z.columns if c.startswith("r_")]
    z["pcb_ret"] = z[members].mean(axis=1,skipna=True); z["pcb_up_breadth"]=(z[members]>0).sum(axis=1)/z[members].notna().sum(axis=1).replace(0,np.nan)
    return z[["pcb_ret","pcb_up_breadth"]].reset_index()


def build_index_context():
    z=load_a("000001.SH")
    if z.empty: return pd.DataFrame(columns=["ts","index_ret"])
    z=z[["ts","close"]].copy(); z["day_open"] = z.close.groupby(z.ts.dt.date).transform("first"); z["index_ret"] = z.close/z.day_open-1
    return z[["ts","index_ret"]]


def load_overseas(fragment):
    files=list(DATA.glob(f"o_*{fragment}*.csv"))
    if not files: return pd.DataFrame()
    z=pd.read_csv(files[0]); z["ts"]=pd.to_datetime(z.ts,utc=True,errors="coerce").dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    z=z.dropna(subset=["ts","close"]).sort_values("ts"); z["day"] = z.ts.dt.date; z["day_open"] = z.close.groupby(z.day).transform("first"); z["ret_from_open"] = z.close/z.day_open-1
    return z[["ts","close","ret_from_open"]]


def merge_external(x, fragment, prefix, tolerance="30min"):
    z=load_overseas(fragment)
    if z.empty:
        x[prefix+"_ret"]=np.nan; x[prefix+"_level"]=np.nan; return x
    z=z.rename(columns={"ret_from_open":prefix+"_ret","close":prefix+"_level"})
    return pd.merge_asof(x.sort_values("ts"),z.sort_values("ts"),on="ts",direction="backward",tolerance=pd.Timedelta(tolerance))


def _daily_macro(label):
    p=DATA/f"macro_{label}_daily_3y.csv"
    if not p.exists(): return pd.DataFrame()
    z=pd.read_csv(p); z["raw_ts"]=pd.to_datetime(z.ts,utc=True,errors="coerce"); z=z.dropna(subset=["raw_ts","close"]).sort_values("raw_ts")
    # A US daily close is conservatively made available only at 08:00 Beijing on the following calendar day.
    local_date=z.raw_ts.dt.tz_convert("America/New_York").dt.date
    z["available_ts"] = pd.to_datetime(pd.Series(local_date).astype(str)) + pd.Timedelta(days=1,hours=8)
    return z


def attach_macro_history(x):
    tyx=_daily_macro("tyx")
    if not tyx.empty:
        closes=tyx.close.astype(float).to_numpy(); pct=[]
        for i,v in enumerate(closes):
            hist=closes[max(0,i-755):i+1]; pct.append(float(np.mean(hist<=v)))
        tyx["tyx_pct_3y"]=pct; tyx["tyx_daily_level"]=tyx.close.astype(float)
        x=pd.merge_asof(x.sort_values("ts"),tyx[["available_ts","tyx_pct_3y","tyx_daily_level"]].sort_values("available_ts"),left_on="ts",right_on="available_ts",direction="backward").drop(columns=["available_ts"])
    else: x["tyx_pct_3y"]=np.nan; x["tyx_daily_level"]=np.nan
    oil=_daily_macro("oil")
    if not oil.empty:
        oil["oil_mom3d"] = oil.close.astype(float).pct_change(3)
        x=pd.merge_asof(x.sort_values("ts"),oil[["available_ts","oil_mom3d"]].sort_values("available_ts"),left_on="ts",right_on="available_ts",direction="backward").drop(columns=["available_ts"])
    else: x["oil_mom3d"]=np.nan
    sox=_daily_macro("sox")
    if not sox.empty:
        sox["sox_prev_daily_ret"] = sox.close.astype(float).pct_change()
        x=pd.merge_asof(x.sort_values("ts"),sox[["available_ts","sox_prev_daily_ret"]].sort_values("available_ts"),left_on="ts",right_on="available_ts",direction="backward").drop(columns=["available_ts"])
    else: x["sox_prev_daily_ret"]=np.nan
    return x


def attach_news(x):
    p=DATA/"news_events.csv"; x["news_score_60m"]=0.0; x["news_count_60m"]=0
    if not p.exists(): return x
    n=pd.read_csv(p)
    if n.empty or "published_ts" not in n: return x
    n["published_ts"]=pd.to_datetime(n.published_ts,utc=True,errors="coerce").dt.tz_convert("Asia/Shanghai").dt.tz_localize(None); n=n.dropna(subset=["published_ts"]).sort_values("published_ts")
    scores=pd.to_numeric(n.get("event_score",0),errors="coerce").fillna(0); os,oc=[],[]
    for t in x.ts:
        mask=(n.published_ts<=t)&(n.published_ts>t-pd.Timedelta(minutes=60)); os.append(float(scores[mask].sum())); oc.append(int(mask.sum()))
    x["news_score_60m"]=os; x["news_count_60m"]=oc; return x


def evaluate_future(x,i,side):
    row={"future_15m":np.nan,"future_30m":np.nan,"future_60m":np.nan,"mae_30m":np.nan,"mfe_30m":np.nan}; px=x.loc[i,"close"]; day=x.loc[i,"ts"].date()
    for bars,label in [(3,"15m"),(6,"30m"),(12,"60m")]:
        w=x.iloc[i+1:i+1+bars]; w=w[w.ts.dt.date==day]
        if w.empty: continue
        favorable=(px-w.low.min())/px if side=="HIGH" else (w.high.max()-px)/px; adverse=(w.high.max()-px)/px if side=="HIGH" else (px-w.low.min())/px
        row[f"future_{label}"]=favorable
        if bars==6: row["mfe_30m"],row["mae_30m"]=favorable,adverse
    return row


def extreme_recall(x,out,side):
    matched=0; matched_pre=0; total=0; leads=[]
    for d,g in x.groupby(x.ts.dt.date):
        if len(g)<20: continue
        total+=1
        idx=g.high.idxmax() if side=="HIGH" else g.low.idxmin(); et=x.loc[idx,"ts"]
        a=out[(out.side==side)&(out.alert)&(out.ts.dt.date==d)]
        near=a[(a.ts>=et-pd.Timedelta(minutes=15))&(a.ts<=et+pd.Timedelta(minutes=10))]
        pre=a[(a.ts>=et-pd.Timedelta(minutes=15))&(a.ts<=et)]
        if not near.empty: matched+=1
        if not pre.empty:
            matched_pre+=1; leads.append((et-pre.ts.max()).total_seconds()/60)
    return {"daily_extremes":total,"recall_near_extreme":matched/total if total else None,"pre_extreme_recall_15m":matched_pre/total if total else None,"median_lead_min":float(np.median(leads)) if leads else None}


def main():
    target=load_a(TARGET)
    if target.empty: raise SystemExit("missing target data")
    x=add_target_features(target); pcb=build_pcb_context(); idx=build_index_context(); x=x.merge(pcb,on="ts",how="left").merge(idx,on="ts",how="left"); x["pcb_rel"]=x.pcb_ret-x.index_ret
    for frag,prefix in [("000660_KS","hynix"),("005930_KS","samsung"),("IDX_KS11","kospi"),("NQ_F","nq"),("CL_F","oil")]: x=merge_external(x,frag,prefix)
    x=attach_macro_history(x); x=attach_news(x)
    rows=[]
    for i,r in x.iterrows():
        if i<8: continue
        for side,flag in [("HIGH",r.high_candidate),("LOW",r.low_candidate)]:
            if not bool(flag): continue
            prev=x.loc[i-1]
            trend_up=float(r.up_structure)+(0.25 if r.above_vwap else 0)+(0.25 if pd.notna(r.pcb_rel) and r.pcb_rel>0 else 0)
            trend_down=float(r.down_structure)+(0.25 if not r.above_vwap else 0)+(0.25 if pd.notna(r.pcb_rel) and r.pcb_rel<0 else 0)
            pcb_turn_down=pd.notna(r.pcb_rel) and pd.notna(prev.pcb_rel) and r.pcb_rel<prev.pcb_rel; pcb_turn_up=pd.notna(r.pcb_rel) and pd.notna(prev.pcb_rel) and r.pcb_rel>prev.pcb_rel
            korea_down=sum(pd.notna(r.get(c)) and r.get(c)<0 for c in ["hynix_ret","samsung_ret","kospi_ret"]); korea_up=sum(pd.notna(r.get(c)) and r.get(c)>0 for c in ["hynix_ret","samsung_ret","kospi_ret"])
            oil_up=(pd.notna(r.oil_ret) and r.oil_ret>0) or (pd.notna(r.oil_mom3d) and r.oil_mom3d>0)
            oil_bond_stress=bool(oil_up and pd.notna(r.tyx_pct_3y) and r.tyx_pct_3y>=.80)
            if side=="HIGH":
                exhaustion=1.0+(0.50 if pcb_turn_down else 0)+0.10*korea_down+(0.15 if pd.notna(r.nq_ret) and r.nq_ret<0 else 0)+(0.15 if r.news_score_60m<0 else 0)
                exhaustion += 0.10 if oil_bond_stress and (pcb_turn_down or korea_down>=2) else 0; score=exhaustion-max(0,trend_up-.75)*.50
            else:
                exhaustion=1.0+(0.50 if pcb_turn_up else 0)+0.10*korea_up+(0.15 if pd.notna(r.nq_ret) and r.nq_ret>0 else 0)+(0.15 if r.news_score_60m>0 else 0)
                score=exhaustion-max(0,trend_down-.75)*.50-(0.10 if oil_bond_stress else 0)
            alert=score>=1.0
            rec={"ts":r.ts,"side":side,"price":r.close,"alert":alert,"score":score,"trend_up":trend_up,"trend_down":trend_down,"pcb_ret":r.pcb_ret,"pcb_breadth":r.pcb_up_breadth,"index_ret":r.index_ret,"pcb_rel":r.pcb_rel,"hynix_ret":r.hynix_ret,"samsung_ret":r.samsung_ret,"kospi_ret":r.kospi_ret,"sox_prev":r.sox_prev_daily_ret,"nq_ret":r.nq_ret,"oil_ret":r.oil_ret,"oil_mom3d":r.oil_mom3d,"tyx_level":r.tyx_daily_level,"tyx_pct":r.tyx_pct_3y,"oil_bond_stress":oil_bond_stress,"news_score_60m":r.news_score_60m,"news_count_60m":r.news_count_60m}
            rec.update(evaluate_future(x,i,side)); rows.append(rec)
    out=pd.DataFrame(rows); out.to_csv(RESULTS/"candidates_v13.csv",index=False)
    summary={}
    for side in ["HIGH","LOW"]:
        allc=out[out.side==side]; a=allc[allc.alert]
        summary[side]={"candidates":int(len(allc)),"alerts":int(len(a)),"alert_rate":float(len(a)/len(allc)) if len(allc) else None,"win_15m_1pct":float((a.future_15m>=.01).mean()) if len(a) else None,"win_30m_1_5pct":float((a.future_30m>=.015).mean()) if len(a) else None,"median_mfe_30m":float(a.mfe_30m.median()) if len(a) else None,"median_mae_30m":float(a.mae_30m.median()) if len(a) else None}
        summary[side].update(extreme_recall(x,out,side))
    (RESULTS/"summary_v13.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); print("V1.3 SUMMARY"); print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__": main()

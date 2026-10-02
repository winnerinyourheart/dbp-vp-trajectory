import pandas as pd,numpy as np,os,pickle,warnings,json;warnings.filterwarnings("ignore")
import os
from sqlalchemy import create_engine,text
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RepeatedStratifiedKFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# --- path & credential configuration (added for public release) ---
DATA_DIR = os.environ.get("DBP_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
DB_PASSWORD = os.environ.get("DBP_DB_PASSWORD", "postgres")

B = DATA_DIR
M=create_engine(f"postgresql://postgres:{DB_PASSWORD}@localhost:5433/mimiciv")
E=create_engine(f"postgresql://postgres:{DB_PASSWORD}@localhost:5433/eicu")
qm=lambda s:pd.read_sql(text(s),M);qe=lambda s:pd.read_sql(text(s),E)
D=[f"dbp_w{i}"for i in range(8)];V=[f"vaso_w{i}"for i in range(8)]
print("=== MIMIC Binary ===")
c=qm("SELECT stay_id FROM public.ds_mimic_shock")
dp=qm("""SELECT v.stay_id,v.dbp,EXTRACT(EPOCH FROM(v.charttime-c.intime))/3600 h
FROM mimiciv_derived.vitalsign v INNER JOIN public.ds_mimic_shock c ON v.stay_id=c.stay_id
WHERE v.charttime BETWEEN c.intime AND c.intime+INTERVAL'48 hours' AND v.dbp IS NOT NULL""")
dp["w"]=(dp["h"]/6).astype(int);dp=dp[dp["w"].between(0,7)]
dpp=dp.groupby(["stay_id","w"])["dbp"].mean().reset_index().pivot_table(index="stay_id",columns="w",values="dbp").reset_index()
dpp.columns=["stay_id"]+D
for k in D:
    if k not in dpp.columns:dpp[k]=np.nan
nv=qm("""SELECT nd.stay_id,EXTRACT(EPOCH FROM(nd.starttime-c.intime))/3600 h
FROM mimiciv_derived.norepinephrine_equivalent_dose nd INNER JOIN public.ds_mimic_shock c ON nd.stay_id=c.stay_id
WHERE nd.starttime BETWEEN c.intime AND c.intime+INTERVAL'48 hours' AND nd.norepinephrine_equivalent_dose>0""")
nv["w"]=(nv["h"]/6).astype(int);nv=nv[nv["w"].between(0,7)]
nvp=nv.groupby("stay_id")["w"].apply(set).reset_index()
for w in range(8):
    nvp[f"vaso_w{w}"]=nvp["w"].apply(lambda s:1 if w in s else 0)
nvp=nvp.drop("w",axis=1)
f=c.merge(dpp,on="stay_id",how="left").merge(nvp,on="stay_id",how="left")
for k in V:
    if k not in f.columns:f[k]=0
f["n"]=f[D].notna().sum(axis=1);f=f[f["n"]>=4].reset_index(drop=True)
o=qm("SELECT stay_id,mort_28d,LEAST(EXTRACT(EPOCH FROM(COALESCE(dod,intime+INTERVAL'28 days')-intime))/86400,28.0) surv,sofa_total sofa FROM public.ds_mimic_shock")
dt=f.merge(o,on="stay_id",how="inner")
print(f"MIMIC: {len(dt)}, mort={dt['mort_28d'].mean():.1%}, SOFA={dt['sofa'].mean():.1f}")
dt.to_csv(os.path.join(B,"binary_mimic_data.csv"),index=False)
print("=== GMM K=6 ===")
sc=StandardScaler()
Xs=sc.fit_transform(dt[D+V].fillna(0).values)
gm=GaussianMixture(n_components=6,random_state=42,n_init=20,max_iter=500,tol=1e-4).fit(Xs)
cl=gm.predict(Xs);y=dt["mort_28d"].values
cm=sorted([(c,float(y[cl==c].mean()))for c in range(6)],key=lambda x:x[1])
cmap={o:n for n,(o,_)in enumerate(cm)}
cl_o=np.array([cmap[c]for c in cl])
dt["class"]=cl_o
for c in range(6):
    sb=dt[dt["class"]==c]
    print(f"  C{c}: n={len(sb)}, mort={sb['mort_28d'].mean():.1%}")
pickle.dump({"gmm":gm,"scaler":sc,"class_map":cmap},open(os.path.join(B,"binary_gmm.pkl"),"wb"))
print("=== Cox ===")
dm=pd.DataFrame({"surv":dt["surv"],"event":dt["mort_28d"]})
for c in range(1,6):dm[f"c{c}"]=(dt["class"]==c).astype(int)
cu=CoxPHFitter(penalizer=0).fit(dm[["surv","event"]+[f"c{c}"for c in range(1,6)]],"surv","event")
un=cu.concordance_index_
print(f"Unadj C-stat: {un:.4f}")
for idx,row in cu.summary.iterrows():
    cl=np.exp(row["coef"]-1.96*row["se(coef)"])
    ch=np.exp(row["coef"]+1.96*row["se(coef)"])
    print(f"  {idx}: HR={row['exp(coef)']:.2f} ({cl:.2f}-{ch:.2f}), p={row['p']:.4f}")
dm["sofa"]=dt["sofa"].values
ca=CoxPHFitter(penalizer=0).fit(dm,"surv","event")
so=ca.concordance_index_
print(f"+SOFA C-stat: {so:.4f}")
print("=== CV (5x5) ===")
rskf=RepeatedStratifiedKFold(5,2,random_state=42)
uc,ac=[],[]
for tr_i,te_i in rskf.split(dt,dt["mort_28d"]):
    tr,te=dt.iloc[tr_i],dt.iloc[te_i]
    sc2=StandardScaler()
    Xtr_s=sc2.fit_transform(tr[D+V].fillna(0).values)
    Xte_s=sc2.transform(te[D+V].fillna(0).values)
    gc2=GaussianMixture(n_components=6,random_state=42,n_init=10,max_iter=300,tol=1e-3).fit(Xtr_s)
    trc=gc2.predict(Xtr_s);tec=gc2.predict(Xte_s)
    yt=tr["mort_28d"].values
    cm2=sorted([(c,float(yt[trc==c].mean()))for c in range(6)],key=lambda x:x[1])
    cm2p={o:n for n,(o,_)in enumerate(cm2)}
    tec_o=np.array([cm2p.get(c,0)for c in tec])
    d1=pd.DataFrame({"surv":tr["surv"].values,"event":tr["mort_28d"].values})
    d2=pd.DataFrame({"surv":te["surv"].values,"event":te["mort_28d"].values})
    for c in range(1,6):
        d1[f"c{c}"]=(trc==c).astype(int);d2[f"c{c}"]=(tec_o==c).astype(int)
    cp=CoxPHFitter(penalizer=0.5).fit(d1,"surv","event")
    lp=cp.predict_partial_hazard(d2.drop(columns=["surv","event"]))
    uc.append(concordance_index(d2["surv"].values,-lp.values,d2["event"].values))
    d1["sofa"]=tr["sofa"].values;d2["sofa"]=te["sofa"].values
    cp=CoxPHFitter(penalizer=0.5).fit(d1,"surv","event")
    lp=cp.predict_partial_hazard(d2.drop(columns=["surv","event"]))
    ac.append(concordance_index(d2["surv"].values,-lp.values,d2["event"].values))
print(f"CV unadj: {np.mean(uc):.4f}+/-{np.std(uc):.4f}")
print(f"CV SOFA:  {np.mean(ac):.4f}+/-{np.std(ac):.4f}")
print("=== eICU ===")
ec=qe("""WITH adult AS(SELECT patientunitstayid FROM eicu_crd.patient WHERE age NOT IN('','0','<18')AND(age='>89'OR age::numeric>=18)),
vaso_c AS(SELECT DISTINCT patientunitstayid FROM eicu_crd.pivoted_treatment_vasopressor WHERE vasopressor=1 AND chartoffset>=0 AND chartoffset<=2880),
lac AS(SELECT DISTINCT patientunitstayid FROM eicu_crd.pivoted_lab WHERE lactate IS NOT NULL AND lactate>2 AND chartoffset>=0 AND chartoffset<=1440),
first_icu AS(SELECT patientunitstayid FROM(SELECT patientunitstayid,ROW_NUMBER()OVER(PARTITION BY patienthealthsystemstayid ORDER BY unitvisitnumber)rn FROM eicu_crd.patient)sub WHERE rn=1)
SELECT a.patientunitstayid FROM adult a INNER JOIN vaso_c v ON a.patientunitstayid=v.patientunitstayid INNER JOIN lac l ON a.patientunitstayid=l.patientunitstayid INNER JOIN first_icu fi ON a.patientunitstayid=fi.patientunitstayid""")
excl=qe("""SELECT d.patientunitstayid,MAX(CASE WHEN d.icd9code LIKE'427.5%' OR d.icd9code IN('42741','427.41','42742','427.42')THEN 1 ELSE 0 END)ca,MAX(CASE WHEN d.icd9code LIKE'571.2%' OR d.icd9code LIKE'571.5%' OR d.icd9code LIKE'571.6%' OR d.icd9code LIKE'572.3%'THEN 1 ELSE 0 END)ci,MAX(CASE WHEN d.icd9code LIKE'410%'THEN 1 ELSE 0 END)mi,MAX(CASE WHEN d.icd9code LIKE'39.65%' OR d.diagnosisstring ILIKE'%ECMO%' OR d.diagnosisstring ILIKE'%extracorporeal%'THEN 1 ELSE 0 END)ecmo FROM eicu_crd.diagnosis d GROUP BY d.patientunitstayid""")
ec=ec.merge(excl,on="patientunitstayid",how="left")
for c2 in ["ca","ci","mi","ecmo"]:ec[c2]=ec[c2].fillna(0).astype(int)
ec=ec[ec[["ca","ci","mi","ecmo"]].sum(axis=1)==0]
aps=qe("SELECT patientunitstayid,apachescore FROM eicu_crd.apachepatientresult WHERE apacheversion='IV'")
ec=ec.merge(aps,on="patientunitstayid",how="left").rename(columns={"apachescore":"sofa"})
out=qe("SELECT patientunitstayid,hospitaldischargestatus,LEAST(unitdischargeoffset/1440.0,28.0)surv FROM eicu_crd.patient")
out["mort"]=(out["hospitaldischargestatus"]=="Expired").astype(int)
ec=ec.merge(out[["patientunitstayid","mort","surv"]],on="patientunitstayid",how="inner")
eids=ec["patientunitstayid"].tolist()
ea=[]
for i in range(0,len(eids),500):
    ch=",".join(str(x)for x in eids[i:i+500])
    ea.append(qe(f"SELECT patientunitstayid,FLOOR(chartoffset/360)::int w,AVG(COALESCE(ibp_diastolic,nibp_diastolic))dbp FROM eicu_crd.pivoted_vital WHERE patientunitstayid IN({ch})AND COALESCE(ibp_diastolic,nibp_diastolic)IS NOT NULL AND chartoffset>=0 AND chartoffset<2880 GROUP BY patientunitstayid,FLOOR(chartoffset/360)::int"))
ed=pd.concat(ea,ignore_index=True)
ed=ed[ed["w"].between(0,7)]
edp=ed.pivot_table(index="patientunitstayid",columns="w",values="dbp").reset_index()
edp.columns=["patientunitstayid"]+D
for k in D:
    if k not in edp.columns:edp[k]=np.nan
ev=qe("SELECT patientunitstayid,FLOOR(chartoffset/360)::int w FROM eicu_crd.pivoted_treatment_vasopressor WHERE vasopressor=1 AND chartoffset>=0 AND chartoffset<2880")
ev=ev[ev["w"].between(0,7)]
evp=ev.groupby("patientunitstayid")["w"].apply(set).reset_index()
for w in range(8):
    evp[f"vaso_w{w}"]=evp["w"].apply(lambda s:1 if w in s else 0)
evp=evp.drop("w",axis=1)
edat=ec.merge(edp,on="patientunitstayid",how="inner").merge(evp,on="patientunitstayid",how="left")
for k in V:
    if k not in edat.columns:edat[k]=0
edat["n"]=edat[D].notna().sum(axis=1)
edat=edat[edat["n"]>=4].reset_index(drop=True)
edat["surv"]=edat["surv"].clip(upper=28)
print(f"eICU: {len(edat)}, mort={edat['mort'].mean():.1%}")
print("  Model Transport:")
Xmt=edat[D+V].fillna(0).values;Xms=sc.transform(Xmt);clm=gm.predict(Xms)
for c in range(6):
    sb=edat.iloc[clm==c]
    print(f"    C{c}: n={len(sb)}, mort={sb['mort'].mean():.1%}")
dmm=pd.DataFrame({"surv":edat["surv"].values,"event":edat["mort"].values})
for c in range(1,6):dmm[f"c{c}"]=(clm==c).astype(int)
cp_mt=CoxPHFitter(penalizer=0.5).fit(dmm[["surv","event"]+[f"c{c}"for c in range(1,6)]],"surv","event")
tm=cp_mt.concordance_index_
dmm["sofa"]=edat["sofa"].fillna(0).values
cp_mt_a=CoxPHFitter(penalizer=0.5).fit(dmm,"surv","event")
ts=cp_mt_a.concordance_index_
print(f"  Transport: unadj={tm:.4f}, +SOFA={ts:.4f}")
for idx,row in cp_mt.summary.iterrows():
    cl=np.exp(row["coef"]-1.96*row["se(coef)"]);ch=np.exp(row["coef"]+1.96*row["se(coef)"])
    print(f"    {idx}: HR={row['exp(coef)']:.2f} ({cl:.2f}-{ch:.2f}), p={row['p']:.4f}")
print("  Independent GMM:")
e_scl=StandardScaler();eXs=e_scl.fit_transform(Xmt)
e_gmm=GaussianMixture(n_components=6,random_state=42,n_init=20,max_iter=500).fit(eXs)
e_cl=e_gmm.predict(eXs)
ey=edat["mort"].values
ecm=sorted([(c,float(ey[e_cl==c].mean()))for c in range(6)],key=lambda x:x[1])
ecmap={o:n for n,(o,_)in enumerate(ecm)}
e_cl_o=np.array([ecmap.get(c,0)for c in e_cl])
for c in range(6):
    sb=edat.iloc[e_cl_o==c]
    print(f"    C{c}: n={len(sb)}, mort={sb['mort'].mean():.1%}")
dim=pd.DataFrame({"surv":edat["surv"].values,"event":edat["mort"].values})
for c in range(1,6):dim[f"c{c}"]=(e_cl_o==c).astype(int)
cp_ind=CoxPHFitter(penalizer=0.5).fit(dim[["surv","event"]+[f"c{c}"for c in range(1,6)]],"surv","event")
ie=cp_ind.concordance_index_
print(f"  Independent C-stat: {ie:.4f}")
res={"mimic_n":len(dt),"mimic_mort":round(float(dt["mort_28d"].mean()),3),"mimic_sofa":round(float(dt["sofa"].mean()),1),
     "mimic_cstat":round(un,4),"mimic_cstat_sofa":round(so,4),
     "cv_unadj_mean":round(float(np.mean(uc)),4),"cv_unadj_sd":round(float(np.std(uc)),4),
     "cv_sofa_mean":round(float(np.mean(ac)),4),"cv_sofa_sd":round(float(np.std(ac)),4),
     "eicu_n":len(edat),"eicu_mort":round(float(edat["mort"].mean()),3),
     "eicu_transport":round(tm,4),"eicu_transport_sofa":round(ts,4),
     "eicu_independent":round(ie,4)}
json.dump(res,open(os.path.join(B,"binary_final_results.json"),"w"),indent=2)
print(json.dumps(res,indent=2))
print("[DONE]")

from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
K = ROOT / "knowledge"
REVIEW = K / "derived" / "nga_tid47288722_review.jsonl"
PENDING = K / "pending_rules.jsonl"
OUT = K / "derived" / "nga_tid47288722_high_signal_review.jsonl"


def load_jsonl(p: Path):
    out=[]
    if not p.exists(): return out
    for line in p.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        try: x=json.loads(line)
        except Exception: continue
        if isinstance(x,dict): out.append(x)
    return out


def clean_author_response(s: str) -> str:
    s = re.sub(r'\[quote\].*?\[/quote\]', '', s or '', flags=re.I|re.S)
    s = re.sub(r'\[img\].*?\[/img\]', '[图片]', s, flags=re.I|re.S)
    s = re.sub(r'\[[^\]]+\]', '', s)
    s = re.sub(r'<br\s*/?>', '\n', s, flags=re.I)
    s = re.sub(r'<[^>]+>', '', s)
    s = html.unescape(s)
    s = re.sub(r'\n{3,}', '\n\n', s).strip()
    return s

WEIGHTS = {
    '策略':4,'体系':4,'原则':5,'铁律':6,'核心':4,'重点':3,'一定':3,'必须':4,'绝对':4,
    '不要':3,'不做':4,'只看':4,'只要':2,'如果':2,'条件':3,'确认':2,'失效':5,'纠偏':5,
    '仓位':3,'减仓':3,'加仓':3,'止损':4,'止盈':4,'做T':3,'趋势':3,'量价':4,'放量':2,'缩量':2,
    '突破':2,'压力':2,'支撑':2,'主线':3,'板块':2,'基本面':2,'高位':3,'低位':2,'市场':2,
    '确定性':4,'回撤':4,'风险':2,'买点':3,'卖点':3,'技术':2,'资金':2,'主力':2,'机构':2,
}

def score(t: str) -> int:
    v=sum(w*t.count(k) for k,w in WEIGHTS.items())
    if re.search(r'(我.{0,8}(策略|体系|原则)|记住|重申|总结|最重要|最重点|保命)',t): v+=8
    if len(t)>=120: v+=2
    return v


def main():
    review=load_jsonl(REVIEW)
    pending={str(x.get('source_pid')):x for x in load_jsonl(PENDING)}
    rows=[]
    for x in review:
        pid=str(x.get('pid',''))
        if pid not in pending: continue
        t=clean_author_response(str(x.get('content','')))
        if not t: continue
        rows.append({
            'posted_at':x.get('posted_at'), 'pid':pid, 'floor':x.get('floor'), 'page':x.get('page'),
            'url':x.get('url'), 'tags':pending[pid].get('tags',[]), 'score':score(t),
            'author_response':t,
        })
    rows.sort(key=lambda z:(-int(z['score']),str(z['posted_at']),str(z['pid'])))
    high=rows[:140]
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in high),encoding='utf-8')
    print(json.dumps({'candidate_count':len(rows),'high_signal_count':len(high),'top_score':high[0]['score'] if high else 0},ensure_ascii=False))

if __name__=='__main__': main()

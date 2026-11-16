"""TE streamer model: ranks tight ends for the upcoming NFL week and filters to
players available in a fantasy league.

  python te_streamer.py --sleeper-league <ID>      # reads rosters + scoring from Sleeper
  python te_streamer.py --espn-league <ID>         # public ESPN leagues only
  python te_streamer.py --available my_wire.txt    # one player name per line
  python te_streamer.py                            # no league: every TE, with ESPN rostered %

Data: nflverse (2019 to present). Model: LightGBM on trailing usage, role, Vegas and defense.
"""
import argparse, json, os, re, warnings
import numpy as np, pandas as pd, requests, lightgbm as lgb
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data'); OUT = os.path.join(HERE, 'output')
NV = 'https://github.com/nflverse/nflverse-data/releases/download'
FIX = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA', 'LAR': 'LA'}

def norm(s): return re.sub(r'[^a-z]', '', re.sub(r'\b(jr|sr|ii|iii|iv)\b\.?', '', str(s).lower()))

def fetch(rel, season, cur):
    """Download an nflverse file; past seasons are cached, the current one is refreshed."""
    os.makedirs(DATA, exist_ok=True); path = os.path.join(DATA, os.path.basename(rel))
    if not os.path.exists(path) or season is None or season >= cur:
        r = requests.get(f'{NV}/{rel}', timeout=180); r.raise_for_status(); open(path, 'wb').write(r.content)
    return path

def load(cur):
    yrs = range(2019, cur + 1)
    rd = lambda t, y, **k: pd.read_parquet(fetch(t.format(y=y), y, cur), **k)
    st = pd.concat(rd('stats_player/stats_player_week_{y}.parquet', y) for y in yrs)
    sn = pd.concat(rd('snap_counts/snap_counts_{y}.parquet', y) for y in yrs)
    ro = pd.concat(rd('weekly_rosters/roster_weekly_{y}.parquet', y) for y in yrs)
    pcols = ['season', 'week', 'season_type', 'receiver_player_id', 'yardline_100', 'air_yards', 'pass_attempt', 'two_point_attempt']
    pb = pd.concat(rd('pbp/play_by_play_{y}.parquet', y, columns=pcols) for y in yrs)
    g = pd.read_parquet(fetch('schedules/games.parquet', None, cur))
    st = st[st.season_type == 'REG']; sn = sn[sn.game_type == 'REG']; ro = ro[ro.game_type == 'REG']
    g = g[(g.game_type == 'REG') & g.season.isin(yrs)]
    for d, c in [(st, 'team'), (st, 'opponent_team'), (sn, 'team'), (ro, 'team'), (g, 'home_team'), (g, 'away_team')]:
        d[c] = d[c].replace(FIX)
    return st, sn, ro, pb, g

def build(st, sn, ro, pb, g, S, W, rec_pts):
    pb = pb[(pb.season_type == 'REG') & pb.receiver_player_id.notna() & (pb.pass_attempt == 1) & (pb.two_point_attempt != 1)].copy()
    pb['rz_tgt'] = (pb.yardline_100 <= 20).astype(int); pb['ez_tgt'] = (pb.air_yards >= pb.yardline_100).astype(int)
    pb = pb.groupby(['season', 'week', 'receiver_player_id'])[['rz_tgt', 'ez_tgt']].sum().reset_index().rename(columns={'receiver_player_id': 'player_id'})
    idm = ro.dropna(subset=['pfr_id', 'gsis_id']).drop_duplicates('pfr_id').set_index('pfr_id').gsis_id
    sn = sn.assign(player_id=sn.pfr_player_id.map(idm))
    snt = sn[(sn.position == 'TE') & sn.player_id.notna() & (sn.offense_snaps > 0)][['season', 'week', 'team', 'player_id', 'player', 'offense_snaps', 'offense_pct']]
    num = ['targets', 'receptions', 'receiving_yards', 'receiving_tds', 'receiving_air_yards', 'target_share', 'fantasy_points']
    stt = st[st.position == 'TE'][['season', 'week', 'team', 'player_id', 'player_display_name'] + num]
    df = snt.merge(stt, on=['season', 'week', 'team', 'player_id'], how='outer')
    df['name'] = df.player_display_name.fillna(df.player); df[num] = df[num].fillna(0)
    df['pts'] = df.fantasy_points + rec_pts * df.receptions          # league scoring
    df = df.merge(pb, on=['season', 'week', 'player_id'], how='left'); df[['rz_tgt', 'ez_tgt']] = df[['rz_tgt', 'ez_tgt']].fillna(0)
    # team context: Vegas implied total, pass volume, TE share, opponent defense vs TEs
    h = g[['season', 'week', 'home_team', 'away_team', 'spread_line', 'total_line']]
    a = h.rename(columns={'home_team': 'opp', 'away_team': 'team'}).assign(is_home=0); a['spread'] = -a.spread_line
    b = h.rename(columns={'home_team': 'team', 'away_team': 'opp'}).assign(is_home=1); b['spread'] = b.spread_line
    tm = pd.concat([a, b]); tm['implied'] = tm.total_line / 2 + tm.spread / 2
    tp = st.groupby(['season', 'week', 'team']).agg(team_att=('attempts', 'sum'), team_tgt=('targets', 'sum')).reset_index()
    tet = df.groupby(['season', 'week', 'team']).agg(te_tgt=('targets', 'sum'), te_pts=('pts', 'sum')).reset_index()
    tm = tm.merge(tp, on=['season', 'week', 'team'], how='left').merge(tet, on=['season', 'week', 'team'], how='left').sort_values(['team', 'season', 'week'])
    tm['te_tshare'] = tm.te_tgt / tm.team_tgt
    for c in ['team_att', 'te_tshare', 'te_tgt']:
        tm['tm_' + c] = tm.groupby('team')[c].transform(lambda s: s.shift().ewm(halflife=4, min_periods=1, ignore_na=True).mean())
    dv = tm[['season', 'week', 'opp', 'te_pts', 'te_tgt']].sort_values(['opp', 'season', 'week'])
    def shr(s, m, k=4):
        x = s.shift(); return (x.fillna(0).cumsum() + k * m) / (x.notna().cumsum() + k)
    for c, o in [('te_pts', 'def_te_pts'), ('te_tgt', 'def_te_tgt')]:
        m = dv[c].mean(); dv[o] = dv.groupby(['opp', 'season'])[c].transform(lambda s: shr(s, m))
    tm = tm.merge(dv[['season', 'week', 'opp', 'def_te_pts', 'def_te_tgt']], on=['season', 'week', 'opp'], how='left')
    # rows to predict: TEs on each team's latest roster, for teams playing in week W
    lr = ro[(ro.season == S) & (ro.position == 'TE')]; lr = lr[lr.week == lr.groupby('team').week.transform('max')]
    lr = lr[lr.status.isin(['ACT', 'INA'])]
    pr = lr[['team', 'gsis_id', 'full_name', 'status', 'sleeper_id']].rename(columns={'gsis_id': 'player_id', 'full_name': 'name'}).assign(season=S, week=W, pred=1)
    pr = pr[pr.team.isin(tm[(tm.season == S) & (tm.week == W)].team)]
    df['pred'] = 0; df = df[~((df.season == S) & (df.week >= W))]
    df = pd.concat([df, pr]).sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
    gp = df.groupby('player_id'); F = []
    for c in ['targets', 'receptions', 'receiving_yards', 'receiving_tds', 'receiving_air_yards', 'target_share', 'pts', 'offense_pct', 'offense_snaps', 'rz_tgt', 'ez_tgt']:
        df[c + '_e'] = gp[c].transform(lambda s: s.shift().ewm(halflife=3, min_periods=1, ignore_na=True).mean())
        df[c + '_r3'] = gp[c].transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
        df[c + '_l1'] = gp[c].shift(); F += [c + '_e', c + '_r3', c + '_l1']
    df['pts_max5'] = gp.pts.transform(lambda s: s.shift().rolling(5, min_periods=1).max())
    df['career_g'] = gp.cumcount(); df['season_g'] = df.groupby(['player_id', 'season']).cumcount()
    df['tpr'] = df.targets_e / (df.offense_snaps_e + 1)
    ex = ro.dropna(subset=['gsis_id']).groupby(['gsis_id', 'season']).agg(years_exp=('years_exp', 'max'), draft_number=('draft_number', 'max')).reset_index().rename(columns={'gsis_id': 'player_id'})
    df = df.merge(ex, on=['player_id', 'season'], how='left'); df['draft_number'] = df.draft_number.fillna(270)
    # depth role by trailing snap share, and snap share vacated by absent TEs ranked above
    rost = ro[(ro.position == 'TE') & ro.status.isin(['ACT', 'INA', 'RES'])].groupby(['season', 'week', 'team']).gsis_id.apply(set).to_dict()
    df = df.sort_values(['season', 'week']).reset_index(drop=True); n = len(df)
    state = {}; rk = np.full(n, np.nan); vac = np.zeros(n); nc = np.zeros(n); t1 = np.empty(n, dtype=object); t1o = np.zeros(n)
    for (s, w), wk in df.groupby(['season', 'week'], sort=True):
        isp = (s == S and w == W)
        for t, tw in wk.groupby('team'):
            playing = set(tw[tw.status == 'ACT'].player_id) if isp else set(tw.player_id)
            ros = set(tw.player_id) if isp else rost.get((s, w, t), set()) | playing
            cand = {}
            for p in ros:
                v = state.get(p)
                if v and ((v[1] == s and w - v[2] <= 6) or (v[1] == s - 1 and w <= 4)): cand[p] = v[0]
                elif p in playing: cand[p] = 0.0
            order = sorted(cand, key=lambda p: -cand[p])
            for i, p in zip(tw.index, tw.player_id):
                if p not in cand: continue
                r = order.index(p); rk[i] = r + 1; nc[i] = len(order); t1[i] = order[0]
                vac[i] = sum(cand[q] for q in order[:r] if q not in playing); t1o[i] = float(order[0] not in playing and r > 0)
        if not isp:
            for p, l in zip(wk.player_id, wk.offense_pct):
                pv = state.get(p, (np.nan,))[0]
                new = l if pv != pv else (pv if l != l else 0.6 * pv + 0.4 * l)
                state[p] = (0.0 if new != new else new, s, w)
    df['role_rank'] = rk; df['vac_above'] = vac; df['n_te'] = nc; df['te1_id'] = t1; df['te1_out'] = t1o
    ctx = ['opp', 'is_home', 'spread', 'total_line', 'implied', 'tm_team_att', 'tm_te_tshare', 'tm_te_tgt', 'def_te_pts', 'def_te_tgt']
    df = df.merge(tm[['season', 'week', 'team'] + ctx], on=['season', 'week', 'team'], how='left')
    F += ['pts_max5', 'career_g', 'season_g', 'tpr', 'years_exp', 'draft_number', 'role_rank', 'vac_above', 'n_te', 'te1_out'] + ctx[1:]
    return df, F

P = dict(n_estimators=400, learning_rate=0.02, num_leaves=12, min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, verbose=-1, random_state=0)
def fit(a, F):
    return (lgb.LGBMRegressor(objective='tweedie', tweedie_variance_power=1.3, **P).fit(a[F], a.y.clip(lower=0)),
            lgb.LGBMClassifier(**P).fit(a[F], a.hit),
            lgb.LGBMRegressor(objective='quantile', alpha=0.85, **P).fit(a[F], a.y))

def backtest(tr, F, S):
    o = []
    for s in range(S - 4, S + 1):
        a, b = tr[tr.season < s], tr[tr.season == s].copy()
        b['p'] = fit(a, F)[0].predict(b[F]); o.append(b)
    o = pd.concat(o); o['base'] = o.pts_e.fillna(0); o = o[o.role_rank >= 2]   # non-lead TEs: the streamer pool
    top = lambda c, k: o.sort_values(c, ascending=False).groupby(['season', 'week']).head(k)
    L = [f'Backtest {S-4} to {S}, non-lead TEs, n={len(o)} (each season predicted using only earlier seasons)']
    for lab, c in [('model', 'p'), ('trailing average', 'base')]:
        L.append(f'  {lab:17s} top pick/week: {top(c,1).y.mean():.1f} pts, {top(c,1).hit.mean():.0%} hit 10+ | top 3: {top(c,3).y.mean():.1f} pts | rank corr {spearmanr(o.y,o[c])[0]:.2f}')
    return '\n'.join(L)

def league_sleeper(lid):
    lg = requests.get(f'https://api.sleeper.app/v1/league/{lid}', timeout=30).json()
    ros = requests.get(f'https://api.sleeper.app/v1/league/{lid}/rosters', timeout=30).json()
    taken = {str(p) for r in ros for p in (r.get('players') or [])}
    sc = lg.get('scoring_settings', {}); rec = float(sc.get('rec', 0)) + float(sc.get('bonus_rec_te', 0))
    return dict(name=lg.get('name'), rec=rec, taken_sleeper=taken)

def league_espn(lid, season):
    flt = {'players': {'filterStatus': {'value': ['FREEAGENT', 'WAIVERS']}, 'filterSlotIds': {'value': [6]}, 'limit': 300}}
    r = requests.get(f'https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{lid}',
                     params={'view': ['kona_player_info', 'mSettings']}, headers={'x-fantasy-filter': json.dumps(flt)}, timeout=30)
    if r.status_code != 200: raise SystemExit(f'ESPN returned {r.status_code}: the league is private. Make it viewable to the public, or use --available with a pasted list.')
    j = r.json(); rec = 0.0
    for it in j.get('settings', {}).get('scoringSettings', {}).get('scoringItems', []):
        if it.get('statId') == 53: rec = float(it.get('pointsOverrides', {}).get('6', it.get('points', 0)))
    return dict(name=j.get('settings', {}).get('name'), rec=rec, avail_names={norm(p['player']['fullName']) for p in j['players']})

def espn_public(season):
    flt = {'players': {'filterSlotIds': {'value': [6]}, 'limit': 200, 'sortPercOwned': {'sortPriority': 1, 'sortAsc': False}}}
    try:
        j = requests.get(f'https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leaguedefaults/3',
                         params={'view': 'kona_player_info'}, headers={'x-fantasy-filter': json.dumps(flt)}, timeout=30).json()
        return {norm(p['player']['fullName']): (round(p['player']['ownership']['percentOwned']), p['player'].get('injuryStatus', '')) for p in j['players']}
    except Exception: return {}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sleeper-league'); ap.add_argument('--espn-league'); ap.add_argument('--available', help='text file, one available player per line')
    ap.add_argument('--scoring', choices=['ppr', 'half', 'std'], help='overrides league scoring'); ap.add_argument('--season', type=int); ap.add_argument('--week', type=int)
    ap.add_argument('--top', type=int, default=15)
    a = ap.parse_args()
    today = pd.Timestamp.today(); S = a.season or (today.year if today.month >= 8 else today.year - 1)
    lg = league_sleeper(a.sleeper_league) if a.sleeper_league else league_espn(a.espn_league, S) if a.espn_league else {}
    rec = {'ppr': 1.0, 'half': 0.5, 'std': 0.0}[a.scoring] if a.scoring else lg.get('rec', 1.0)
    st, sn, ro, pb, g = load(S)
    gs = g[g.season == S]; played = gs.groupby('week').home_score.apply(lambda s: s.notna().any())
    W = a.week or int(played[~played].index.min())     # first week with no games played yet
    df, F = build(st, sn, ro, pb, g, S, W, rec)
    tr = df[(df.pred == 0) & df.role_rank.notna() & df.implied.notna()].copy(); tr['y'] = tr.pts; tr['hit'] = (tr.y >= 10).astype(int)
    bt = backtest(tr, F, S)
    m, c, q = fit(tr, F); pr = df[df.pred == 1].copy()
    pr['proj'] = m.predict(pr[F]); pr['p10'] = c.predict_proba(pr[F])[:, 1]; pr['ceiling'] = q.predict(pr[F])
    nm = df.dropna(subset=['name']).drop_duplicates('player_id', keep='last').set_index('player_id').name
    pr['ahead_of_him'] = np.where(pr.role_rank > 1, pr.te1_id.map(nm), ''); pr['k'] = pr.name.map(norm)
    pub = espn_public(S); pr['espn_rostered'] = pr.k.map(lambda k: pub.get(k, (0, ''))[0]); pr['injury'] = pr.k.map(lambda k: pub.get(k, (0, ''))[1])
    if 'taken_sleeper' in lg: pr['available'] = ~pr.sleeper_id.astype(str).str.replace(r'\.0$', '', regex=True).isin(lg['taken_sleeper'])
    elif 'avail_names' in lg: pr['available'] = pr.k.isin(lg['avail_names'])
    elif a.available: pr['available'] = pr.k.isin({norm(l) for l in open(a.available) if l.strip()})
    else: pr['available'] = True
    pr.loc[pr.status == 'INA', 'injury'] = pr.injury.replace('', 'INACTIVE LAST GAME').fillna('INACTIVE LAST GAME')
    cols = ['name', 'team', 'opp', 'available', 'proj', 'p10', 'ceiling', 'role_rank', 'ahead_of_him', 'te1_out', 'targets_e', 'offense_pct_e', 'rz_tgt_e', 'implied', 'def_te_pts', 'espn_rostered', 'injury']
    res = pr.sort_values('proj', ascending=False)[cols].rename(columns={'p10': 'chance_10plus', 'targets_e': 'targets_pg', 'offense_pct_e': 'snap_share', 'rz_tgt_e': 'rz_targets_pg', 'implied': 'team_implied_pts', 'def_te_pts': 'opp_pts_allowed_to_te', 'te1_out': 'lead_te_out'}).round(2)
    os.makedirs(OUT, exist_ok=True); path = os.path.join(OUT, f'te_{S}_week{W}.csv'); res.to_csv(path, index=False)
    sc = {1.0: 'PPR', 0.5: 'half PPR', 0.0: 'standard'}.get(rec, f'{rec} per reception')
    print(f"\nTE streamer, {S} week {W}, {sc}" + (f", league: {lg.get('name')}" if lg else ', no league connected (showing all TEs)'))
    print(bt + '\n')
    print(res[res.available].head(a.top).drop(columns=['available', 'ceiling', 'rz_targets_pg']).to_string(index=False))
    print(f'\nFull list: {path}')

if __name__ == '__main__': main()

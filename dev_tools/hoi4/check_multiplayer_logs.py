"""Compare two multiplayer game.log files without starting or controlling either game."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

DATE = re.compile(r'\]\[(\d+)\.(\d+)\.(\d+)\.(\d+)\]')
ACT = re.compile(r'JEV\|ACT\|([A-Z0-9]{3})\|posture=([0-6])(?:\s|$)')
OOS = re.compile(r'out of synch?|not in sync|oos detected', re.I)


def parse(text):
    mode = None
    actions, dates, errors = [], [], []
    for line in text.splitlines():
        if '[[ Launching ' in line:
            mode = 'multiplayer' if 'MULTIPLAYER-game' in line else 'other'
            actions, dates, errors = [], [], []
        if mode != 'multiplayer':
            continue
        match = DATE.search(line)
        if match:
            when = tuple(map(int, match.groups()))
            dates.append(when)
            act = ACT.search(line)
            if act:
                actions.append((*when, act[1], int(act[2])))
        if OOS.search(line):
            errors.append(line)
    return dict(mode=mode, actions=actions, dates=dates, desync_lines=errors)


def compare(host, client):
    a, b = parse(host), parse(client)
    ca, cb = Counter(a['actions']), Counter(b['actions'])
    def elapsed(log):
        if not log['actions'] or not log['dates']:
            return 0
        first = min(x[:4] for x in log['actions'])
        last = max(log['dates'])
        return (last[0]-first[0])*12 + last[1]-first[1] - (last[2:] < first[2:])
    checks = dict(multiplayer_sessions=a['mode']==b['mode']=='multiplayer',
        nonempty_actions=bool(ca and cb), identical_dated_actions=ca==cb,
        three_months_after_first_action=elapsed(a)>=3 and elapsed(b)>=3,
        no_desync_marker=not a['desync_lines'] and not b['desync_lines'])
    return dict(log_checks_passed=all(checks.values()), checks=checks,
        host_actions=sum(ca.values()), client_actions=sum(cb.values()),
        host_only=[list(x) for x in (ca-cb).elements()][:20],
        client_only=[list(x) for x in (cb-ca).elements()][:20],
        scope='Latest logged multiplayer session only; compares game date/hour, country and posture, not wall-clock time.',
        manual_checks_still_required=['Two distinct legitimate Steam accounts and matching playsets.',
            'Every order set arrived before its target date; verify both relay logs.',
            'Forced relay failure and game resynchronization recovery.',
            'No in-game desync indication. Absence of a log string is not proof.',
            'Daily load_oob reread verified with successive changed files.'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host-game',type=Path,required=True)
    ap.add_argument('--client-game',type=Path,required=True)
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    report=compare(args.host_game.read_text(encoding='utf-8',errors='replace'),
                   args.client_game.read_text(encoding='utf-8',errors='replace'))
    data=json.dumps(report,indent=2)
    if args.out:
        with args.out.open('x',encoding='utf-8') as f:f.write(data+'\n')
    print(data)
    return 0 if report['log_checks_passed'] else 1


if __name__=='__main__':raise SystemExit(main())

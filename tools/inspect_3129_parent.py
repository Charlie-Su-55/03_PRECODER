#!/usr/bin/env python3
"""Read-only parent-host metadata; never print proxy credentials or argv."""
import getopt, json, os, pwd, re, subprocess
OPTIONS = 'a:c:d:fghIl:p:r:su:vw:A:BD:F:G:HL:M:N:O:P:R:S:T:U:'
def read(args, binary=False):
    result = subprocess.run(args, capture_output=True, text=not binary)
    if result.returncode: raise ValueError()
    return result.stdout

def parents(tokens):
    tokens, result = list(tokens), []
    while tokens:
        value = tokens.pop(0)
        if ':' not in value and tokens:
            value += ':' + tokens.pop(0)
        match = re.fullmatch(r'([A-Za-z0-9][A-Za-z0-9_.-]{0,252}):([0-9]{1,5})', value)
        if not match or not 0 < int(match[2]) < 65536:
            raise ValueError()
        result.append(match[1]+':'+str(int(match[2])))
    if not result: raise ValueError()
    return result

try:
    if pwd.getpwuid(os.getuid()).pw_name != 'rong' or os.geteuid() != os.getuid():
        raise ValueError()
    pids = set(re.findall(r'\("cntlm",pid=(\d+),', read(['sudo','-n','ss','-H','-ltnp','( sport = :3129 )'])))
    if len(pids) != 1: raise ValueError()
    base = '/proc/'+pids.pop()
    if read(['sudo','-n','readlink',base+'/exe']).strip() != '/usr/sbin/cntlm': raise ValueError()
    raw = read(['sudo','-n','cat',base+'/cmdline'], binary=True)
    if not raw.endswith(b'\0') or len(raw) > 1024*1024: raise ValueError()
    options, remaining = getopt.gnu_getopt(raw[:-1].decode().split('\0')[1:], OPTIONS)
    if [v for k,v in options if k == '-c'] != ['/dev/null']: raise ValueError()
    print(json.dumps({'parent_proxies': parents(remaining)}))
except Exception:
    print('CHECK_STOPPED: proxy layout or permission requires review.')
    raise SystemExit(1)

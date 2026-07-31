import os
import math
import datetime
import urllib.request
import gzip
import shutil
import ssl
import json
import threading
import re
import functools
import time
from flask import Flask, request, send_file, Response
import tempfile

app = Flask(__name__)

# =====================================================================
# CONFIGURACIÓN DEL ENTORNO Y GESTIÓN DE ESTADO (AISLAMIENTO EN RAM)
# =====================================================================
BASE_DIR = tempfile.gettempdir()
REPORT_FOLDER = os.path.join(BASE_DIR, 'informes')
os.makedirs(REPORT_FOLDER, exist_ok=True)

STATE_LOCK = threading.Lock()
SP3_LOCK = threading.Lock() 

# --- CONSTANTES GEODÉSICAS INMUTABLES ---
C_LIGHT = 299792458.0
OMEGA_E = 7.2921151467e-5
MU = 3.986005e14
FREQ_L1 = 1575.42e6
FREQ_L5 = 1176.45e6
WAVE_L1 = C_LIGHT / FREQ_L1
WAVE_L5 = C_LIGHT / FREQ_L5

def f_14(val):
    if val is None: return "0.00000000000000"
    s = f"{float(val):.14f}"
    return s

def safe_f(val, default=0.0):
    try: return float(val) if val and str(val).strip() != '' else default
    except: return default

def safe_i(val, default=19):
    try: return int(val) if val and str(val).strip() != '' else default
    except: return default

def get_workspace(uid):
    ws = os.path.join(BASE_DIR, f'temp_rinex_{uid}')
    os.makedirs(ws, exist_ok=True)
    return ws

def guardar_estado(uid, clave, valor):
    state_file = os.path.join(get_workspace(uid), 'estado_proyecto.json')
    with STATE_LOCK:
        estado = {}
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f: estado = json.load(f)
            except: pass
        estado[clave] = valor
        try:
            with open(state_file, 'w', encoding='utf-8') as f: json.dump(estado, f)
        except: pass

def leer_estado(uid, clave):
    state_file = os.path.join(get_workspace(uid), 'estado_proyecto.json')
    with STATE_LOCK:
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f: return json.load(f).get(clave)
            except: pass
        return None

def gps_time_to_tow(year, month, day, hour, minute, second):
    sec_int, sec_frac = int(second), second - int(second)
    total = (datetime.datetime(year, month, day, hour, minute, sec_int) - datetime.datetime(1980, 1, 6)).total_seconds() + sec_frac
    return total - (int(total // 604800) * 604800)

# =====================================================================
# INTEGRACIÓN E/S GOOGLE DRIVE (TOLERANCIA A FALLOS SSL)
# =====================================================================
def descargar_desde_gdrive(url, filepath):
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if not match:
        match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if not match:
        raise ValueError("URL de Google Drive no reconocida o malformada.")
    
    file_id = match.group(1)
    direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(direct_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as response, open(filepath, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    
    return True

# =====================================================================
# ÁLGEBRA LINEAL Y TRANSFORMACIONES GEODÉSICAS RIGUROSAS (ENUR)
# =====================================================================
def transpose_matrix(M):
    if not M or not M[0]: return []
    try: return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]
    except IndexError: return []

def matmul(A, B):
    if not A or not B or not A[0] or not B[0]: return []
    try:
        result = [[0.0 for _ in range(len(B[0]))] for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]
        return result
    except IndexError: return []

def cholesky_decompose(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            sum1 = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - sum1
                if val <= 0: raise ValueError("Matriz no definida positiva")
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (A[i][j] - sum1) / L[j][j]
    return L

def invert_lower_triangular(L):
    n = len(L)
    inv = [[0.0] * n for _ in range(n)]
    for i in range(n):
        inv[i][i] = 1.0 / L[i][i]
        for j in range(i):
            sum1 = sum(L[i][k] * inv[k][j] for k in range(j, i))
            inv[i][j] = -sum1 / L[i][i]
    return inv

def gauss_jordan_inverse(M):
    n = len(M)
    A = [[float(M[i][j]) for j in range(n)] for i in range(n)]
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        max_k = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > abs(A[max_k][i]): max_k = k
        if max_k != i:
            A[i], A[max_k] = A[max_k], A[i]
            I[i], I[max_k] = I[max_k], I[i]
        pivot = A[i][i]
        if abs(pivot) < 1e-15: return None 
        for j in range(n):
            A[i][j] /= pivot
            I[i][j] /= pivot
        for k in range(n):
            if k == i: continue
            factor = A[k][i]
            for j in range(n):
                A[k][j] -= factor * A[i][j]
                I[k][j] -= factor * I[i][j]
    return I

def invert_matrix_3x3(M):
    det = M[0][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1]) - \
          M[0][1]*(M[1][0]*M[2][2] - M[1][2]*M[2][0]) + \
          M[0][2]*(M[1][0]*M[2][1] - M[1][1]*M[2][0])
    if abs(det) < 1e-15: return None
    return [
        [(M[1][1]*M[2][2] - M[1][2]*M[2][1])/det, (M[0][2]*M[2][1] - M[0][1]*M[2][2])/det, (M[0][1]*M[1][2] - M[0][2]*M[1][1])/det],
        [(M[1][2]*M[2][0] - M[1][0]*M[2][2])/det, (M[0][0]*M[2][2] - M[0][2]*M[2][0])/det, (M[0][2]*M[1][0] - M[0][0]*M[1][2])/det],
        [(M[1][0]*M[2][1] - M[1][1]*M[2][0])/det, (M[0][1]*M[2][0] - M[0][0]*M[2][1])/det, (M[0][0]*M[1][1] - M[0][1]*M[1][0])/det]
    ]

def invert_matrix_nxn(M):
    if not M or not M[0]: return None
    if len(M) == 3 and len(M[0]) == 3:
        return invert_matrix_3x3(M)
    try:
        L = cholesky_decompose(M)
        L_inv = invert_lower_triangular(L)
        return matmul(transpose_matrix(L_inv), L_inv)
    except:
        return gauss_jordan_inverse(M)

def obtener_matriz_rotacion_enu(lat_deg, lon_deg):
    lat_r = math.radians(lat_deg)
    lon_r = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat_r), math.cos(lat_r)
    sin_lon, cos_lon = math.sin(lon_r), math.cos(lon_r)
    return [
        [-sin_lon,             cos_lon,             0.0     ],
        [-sin_lat * cos_lon,  -sin_lat * sin_lon,   cos_lat ],
        [cos_lat * cos_lon,    cos_lat * sin_lon,   sin_lat ]
    ]

def multiplicar_matriz_vector_3x3(R, v):
    return [
        R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
        R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
        R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2]
    ]

# =====================================================================
# PARSERS Y GESTIÓN DE ARCHIVOS CON INTERPOLACIÓN TEMPORAL EXACTA
# =====================================================================
def parse_rinex_obs_completo(path):
    obs = {}
    sys_idx = {}
    sys_tokens = {}
    last_sys_char = None
    
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        in_h = True
        tow = None
        for line in f:
            if in_h:
                if "SYS / # / OBS TYPES" in line:
                    sys_char = line[0].strip()
                    if sys_char: last_sys_char = sys_char
                    if last_sys_char:
                        tokens = [x.strip() for x in line[6:60].split() if x.strip()]
                        if last_sys_char not in sys_tokens: sys_tokens[last_sys_char] = []
                        sys_tokens[last_sys_char].extend(tokens)
                elif "END OF HEADER" in line: 
                    in_h = False
                    for sc, t in sys_tokens.items():
                        sys_idx[sc] = {
                            'C1': next((i for i, x in enumerate(t) if x.startswith('C1')), -1),
                            'L1': next((i for i, x in enumerate(t) if x.startswith('L1')), -1),
                            'C5': next((i for i, x in enumerate(t) if x.startswith('C5')), -1),
                            'L5': next((i for i, x in enumerate(t) if x.startswith('L5')), -1),
                            'S1': next((i for i, x in enumerate(t) if x.startswith('S1')), -1),
                            'S5': next((i for i, x in enumerate(t) if x.startswith('S5')), -1)
                        }
            elif line.startswith('>'):
                p = line[1:].split()
                if len(p) >= 6:
                    y, m, d, h, mn, sec = int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]), float(p[5])
                    if y < 100: y += 2000 
                    tow = round(gps_time_to_tow(y, m, d, h, mn, sec), 6)
                    obs[tow] = {'_meta': (y, m, d, h, mn, sec)}
            elif tow and len(line) > 3 and line[0] in 'GRECSJ':
                sys_char = line[0]
                idx_c1 = sys_idx.get(sys_char, {}).get('C1', -1)
                idx_l1 = sys_idx.get(sys_char, {}).get('L1', -1)
                idx_c5 = sys_idx.get(sys_char, {}).get('C5', -1)
                idx_l5 = sys_idx.get(sys_char, {}).get('L5', -1)
                idx_s1 = sys_idx.get(sys_char, {}).get('S1', -1)
                idx_s5 = sys_idx.get(sys_char, {}).get('S5', -1)
                
                data = {}
                if idx_c1 >= 0 and len(line) >= 17 + 16 * idx_c1:
                    v = line[3+16*idx_c1 : 17+16*idx_c1].strip()
                    if v: data['C1'] = float(v.replace('D', 'E').replace('d', 'e'))
                if idx_c5 >= 0 and len(line) >= 17 + 16 * idx_c5:
                    v = line[3+16*idx_c5 : 17+16*idx_c5].strip()
                    if v: data['C5'] = float(v.replace('D', 'E').replace('d', 'e'))
                if idx_l1 >= 0 and len(line) >= 17 + 16 * idx_l1:
                    v = line[3+16*idx_l1 : 17+16*idx_l1].strip()
                    if v: data['L1'] = float(v.replace('D', 'E').replace('d', 'e'))
                if idx_l5 >= 0 and len(line) >= 17 + 16 * idx_l5:
                    v = line[3+16*idx_l5 : 17+16*idx_l5].strip()
                    if v: data['L5'] = float(v.replace('D', 'E').replace('d', 'e'))
                if idx_s1 >= 0 and len(line) >= 17 + 16 * idx_s1:
                    v = line[3+16*idx_s1 : 17+16*idx_s1].strip()
                    if v: data['S1'] = float(v.replace('D', 'E').replace('d', 'e'))
                if idx_s5 >= 0 and len(line) >= 17 + 16 * idx_s5:
                    v = line[3+16*idx_s5 : 17+16*idx_s5].strip()
                    if v: data['S5'] = float(v.replace('D', 'E').replace('d', 'e'))
                
                valid_p = ('C1' in data and data['C1'] > 15000000.0) or ('C5' in data and data['C5'] > 15000000.0)
                if valid_p:
                    if tow not in obs: obs[tow] = {}
                    obs[tow][line[0:3].strip()] = data
    return obs

def interpolar_base_a_rover(obs_base, tr, max_gap=2.0, tiempos_base=None):
    if tiempos_base is None:
        tiempos_base = sorted(list(obs_base.keys()), key=lambda k: obs_base[k].get('_meta', (0,0,0,0,0,0)))
    if not tiempos_base: return None
    
    exact_idx = min(range(len(tiempos_base)), key=lambda i: abs(tiempos_base[i] - tr))
    if abs(tiempos_base[exact_idx] - tr) <= 1e-3:
        return obs_base[tiempos_base[exact_idx]].copy()
        
    idx_after = 0
    while idx_after < len(tiempos_base) and tiempos_base[idx_after] < tr:
        idx_after += 1
        
    if idx_after == 0 or idx_after >= len(tiempos_base):
        if abs(tiempos_base[exact_idx] - tr) <= max_gap:
            return obs_base[tiempos_base[exact_idx]].copy()
        return None
        
    t1 = tiempos_base[idx_after - 1]
    t2 = tiempos_base[idx_after]
    
    if (t2 - t1) > (max_gap * 4.0):
        if abs(tiempos_base[exact_idx] - tr) <= max_gap:
            return obs_base[tiempos_base[exact_idx]].copy()
        return None
        
    factor = (tr - t1) / (t2 - t1)
    obs_t1 = obs_base[t1]
    obs_t2 = obs_base[t2]
    
    obs_interp = {'_meta': obs_t1.get('_meta')}
    sats_comunes = set(obs_t1.keys()).intersection(set(obs_t2.keys()))
    sats_comunes.discard('_meta')
    
    for s in sats_comunes:
        obs_interp[s] = {}
        for key_obs in ['C1', 'C5', 'L1', 'L5']:
            if key_obs in obs_t1[s] and key_obs in obs_t2[s]:
                v1 = obs_t1[s][key_obs]
                v2 = obs_t2[s][key_obs]
                obs_interp[s][key_obs] = v1 + factor * (v2 - v1)
            elif key_obs in obs_t1[s]:
                obs_interp[s][key_obs] = obs_t1[s][key_obs]
            elif key_obs in obs_t2[s]:
                obs_interp[s][key_obs] = obs_t2[s][key_obs]
                
        for key_snr in ['S1', 'S5']:
            if key_snr in obs_t1[s]:
                obs_interp[s][key_snr] = obs_t1[s][key_snr]
            elif key_snr in obs_t2[s]:
                obs_interp[s][key_snr] = obs_t2[s][key_snr]
                
    if len(obs_interp) > 1:
        return obs_interp
        
    if abs(tiempos_base[exact_idx] - tr) <= max_gap:
        return obs_base[tiempos_base[exact_idx]].copy()
    return None

def generar_rinex_sincronizado(raw_path, out_path, obs_dict):
    header_lines = []
    with open(raw_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "SYS / # / OBS TYPES" in line: continue 
            header_lines.append(line)
            if "END OF HEADER" in line: break
    
    idx = -1
    for i, l in enumerate(header_lines):
        if "END OF HEADER" in l:
            idx = i
            break
            
    if idx != -1:
        constelaciones_requeridas = ['G', 'E', 'C', 'R', 'S', 'J']
        offset = 0
        for c in constelaciones_requeridas:
            header_lines.insert(idx + offset, f"{c}    4 C1 L1 C5 L5                                       SYS / # / OBS TYPES\n")
            offset += 1
            
    with open(out_path, 'w', encoding='utf-8') as f_out:
        for line in header_lines: f_out.write(line)
        for tow in sorted(obs_dict.keys(), key=lambda k: obs_dict[k].get('_meta', (0,0,0,0,0,0))):
            meta = obs_dict[tow].get('_meta')
            if not meta: continue
            y, m, d, h, mn, sec = meta[0], meta[1], meta[2], meta[3], meta[4], meta[5]
            sats = [k for k in obs_dict[tow].keys() if k != '_meta']
            f_out.write(f"> {y} {m:02d} {d:02d} {h:02d} {mn:02d} {sec:11.7f}  0 {len(sats):2d}\n")
            
            for sat in sats:
                c1 = obs_dict[tow][sat].get('C1', 0.0)
                l1 = obs_dict[tow][sat].get('L1', 0.0)
                c5 = obs_dict[tow][sat].get('C5', 0.0)
                l5 = obs_dict[tow][sat].get('L5', 0.0)
                c1_s = f"{c1:14.3f}" if c1 != 0.0 else "              "
                l1_s = f"{l1:14.3f}" if l1 != 0.0 else "              "
                c5_s = f"{c5:14.3f}" if c5 != 0.0 else "              "
                l5_s = f"{l5:14.3f}" if l5 != 0.0 else "              "
                f_out.write(f"{sat}{c1_s}  {l1_s}  {c5_s}  {l5_s}  \n")

def obtener_fecha_obs(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('>'):
                partes = line[1:].strip().split()
                if len(partes) >= 6: 
                    try:
                        y = int(partes[0])
                        year = y if y > 100 else y + 2000
                        return year, int(partes[1]), int(partes[2]), int(partes[3]), int(partes[4]), float(partes[5])
                    except: pass
    return None

# =====================================================================
# PRODUCTOS IGS Y EFEMÉRIDES (HÍBRIDO NAV / SP3)
# =====================================================================
SP3_CACHE = {}
SP3_CACHE_KEYS = []
MAX_CACHE_SIZE = 2048

def parse_sp3_preciso(path):
    sp3_data = {}
    if not path or not os.path.exists(path): return sp3_data
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        current_time = None
        for line in f:
            if line.startswith('* '):
                p = line.split()
                if len(p) >= 7:
                    try:
                        y, m, d, h, mn, s = int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]), float(p[6])
                        if y < 100: y += 2000
                        current_time = gps_time_to_tow(y, m, d, h, mn, s)
                    except: pass
            elif line.startswith('P') and current_time:
                sys_char = line[1]
                if sys_char in 'GECR':
                    sat_id = line[1:4].strip()
                    try:
                        x = float(line[4:18]) * 1000.0
                        y = float(line[18:32]) * 1000.0
                        z = float(line[32:46]) * 1000.0
                        clk = float(line[46:60]) / 1e6 if len(line) > 46 and line[46:60].strip() else 0.0
                        if sat_id not in sp3_data: sp3_data[sat_id] = []
                        sp3_data[sat_id].append((current_time, x, y, z, clk))
                    except: pass
    for sat in sp3_data: sp3_data[sat].sort(key=lambda item: item[0])
    return sp3_data

def lagrange_weights(x, x_pts):
    n = len(x_pts)
    weights = [1.0] * n
    for i in range(n):
        for j in range(n):
            if i != j: 
                weights[i] *= (x - x_pts[j]) / (x_pts[i] - x_pts[j])
    return weights

def interpolate_sp3(sp3_data, sat, t_emision, degree=9):
    global SP3_CACHE, SP3_CACHE_KEYS
    cache_key = f"{sat}_{t_emision}"
    with SP3_LOCK:
        if cache_key in SP3_CACHE: return SP3_CACHE[cache_key]

    if sat not in sp3_data: return None
    data = sp3_data[sat]
    if len(data) < degree + 1: return None
    
    idx = min(range(len(data)), key=lambda i: abs(data[i][0] - t_emision))
    half = degree // 2
    start = max(0, idx - half)
    end = min(len(data), start + degree + 1)
    if end - start < degree + 1: start = max(0, end - degree - 1)
        
    pts = data[start:end]
    t_pts, x_pts, y_pts, z_pts = [], [], [], []
    for p in pts:
        t_pts.append(p[0]); x_pts.append(p[1]); y_pts.append(p[2]); z_pts.append(p[3])
    
    start_clk = max(0, idx - 1)
    end_clk = min(len(data), start_clk + 2)
    if end_clk - start_clk < 2: start_clk = max(0, end_clk - 2)
    pts_clk = data[start_clk:end_clk]
    t_pts_clk, clk_pts = [], []
    for p in pts_clk:
        t_pts_clk.append(p[0]); clk_pts.append(p[4])
    
    w_xyz = lagrange_weights(t_emision, t_pts)
    val_x = sum(w * x for w, x in zip(w_xyz, x_pts))
    val_y = sum(w * y for w, y in zip(w_xyz, y_pts))
    val_z = sum(w * z for w, z in zip(w_xyz, z_pts))
    
    w_clk = lagrange_weights(t_emision, t_pts_clk)
    val_clk = sum(w * c for w, c in zip(w_clk, clk_pts))
    
    result = (val_x, val_y, val_z, val_clk)
    
    with SP3_LOCK:
        if len(SP3_CACHE) >= MAX_CACHE_SIZE:
            oldest_key = SP3_CACHE_KEYS.pop(0)
            SP3_CACHE.pop(oldest_key, None)
        SP3_CACHE[cache_key] = result
        SP3_CACHE_KEYS.append(cache_key)
    return result

def parse_rinex_nav_real(path):
    ephemeris = {'_iono': {'alpha': [0.0]*4, 'beta': [0.0]*4}}
    if not path or not os.path.exists(path): return ephemeris
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        in_h, sat, data = True, None, []
        for line in f:
            if in_h:
                if "IONOSPHERIC CORR" in line:
                    sys_type = line[0:4].strip()
                    vals = []
                    for i in range(4):
                        try:
                            chunk = line[5+i*12 : 5+(i+1)*12].strip().replace('D', 'E').replace('d', 'e')
                            vals.append(float(chunk) if chunk else 0.0)
                        except: vals.append(0.0)
                    if sys_type == 'GPSA': ephemeris['_iono']['alpha'] = vals
                    elif sys_type == 'GPSB': ephemeris['_iono']['beta'] = vals
                elif "END OF HEADER" in line: in_h = False
                continue
            if len(line) > 8 and line[0] in 'GECSJ' and line[1:3].isdigit():
                if sat and len(data) >= 20: 
                    ephemeris.setdefault(sat, []).append({'af0':data[0],'af1':data[1],'af2':data[2],'Crs':data[4],'Delta_n':data[5],'M0':data[6],'Cuc':data[7],'e':data[8],'Cus':data[9],'sqrtA':data[10],'Toe':data[11],'Cic':data[12],'OMEGA':data[13],'Cis':data[14],'i0':data[15],'Crc':data[16],'omega':data[17],'OMEGA_DOT':data[18],'IDOT':data[19]})
                sat = line[0:3].strip()
                data = [float(line[23:42].replace('D','E').replace('d','e')), float(line[42:61].replace('D','E').replace('d','e')), float(line[61:80].replace('D','E').replace('d','e'))]
            elif sat and line.startswith('    '): 
                data.extend([float(line[i:i+19].replace('D','E').replace('d','e').strip()) for i in range(4, 80, 19) if line[i:i+19].strip()])
        if sat and len(data) >= 20: 
            ephemeris.setdefault(sat, []).append({'af0':data[0],'af1':data[1],'af2':data[2],'Crs':data[4],'Delta_n':data[5],'M0':data[6],'Cuc':data[7],'e':data[8],'Cus':data[9],'sqrtA':data[10],'Toe':data[11],'Cic':data[12],'OMEGA':data[13],'Cis':data[14],'i0':data[15],'Crc':data[16],'omega':data[17],'OMEGA_DOT':data[18],'IDOT':data[19]})
    return ephemeris

def seleccionar_efemeride_optima(eph_list, t_target):
    if not eph_list: return None
    valid_ephs = []
    for eph in eph_list:
        dt = t_target - eph.get('Toe', 0)
        if dt > 302400: dt -= 604800
        elif dt < -302400: dt += 604800
        if abs(dt) <= 7200:
            valid_ephs.append((abs(dt), eph))
    if not valid_ephs: return None
    return min(valid_ephs, key=lambda x: x[0])[1]

# =====================================================================
# GEODESIA ESPACIAL Y CORRECCIONES ATMOSFÉRICAS
# =====================================================================
def correccion_mareas_solidas(X, Y, Z, tow, year, month, day):
    try:
        if year < 100: year += 2000
        h2, l2 = 0.609, 0.085
        Re = 6378137.0
        GM_earth, GM_sun, GM_moon = 3.986004418e14, 1.327124e20, 4.902801e12
        
        jd = 367 * year - (7 * (year + (month + 9) // 12)) // 4 + (275 * month) // 9 + day + 1721013.5
        t_jc = (jd - 2451545.0 + (tow / 86400.0)) / 36525.0
        
        mean_long_sun = 280.460 + 36000.771 * t_jc
        mean_anom_sun = 357.528 + 35999.050 * t_jc
        ecl_lon_sun = mean_long_sun + 1.915 * math.sin(math.radians(mean_anom_sun)) + 0.020 * math.sin(math.radians(2 * mean_anom_sun))
        dist_sun = 1.495978707e11 * (1.00014 - 0.01671 * math.cos(math.radians(mean_anom_sun)) - 0.00014 * math.cos(math.radians(2 * mean_anom_sun)))
        obliquity = 23.439 - 0.013 * t_jc
        
        xs_sun = dist_sun * math.cos(math.radians(ecl_lon_sun))
        ys_sun = dist_sun * math.cos(math.radians(obliquity)) * math.sin(math.radians(ecl_lon_sun))
        zs_sun = dist_sun * math.sin(math.radians(obliquity)) * math.sin(math.radians(ecl_lon_sun))
        
        mean_long_moon = 218.316 + 481267.881 * t_jc
        mean_anom_moon = 134.963 + 477198.867 * t_jc
        mean_dist_moon = 93.272 + 483202.017 * t_jc
        ecl_lon_moon = mean_long_moon + 6.289 * math.sin(math.radians(mean_anom_moon))
        ecl_lat_moon = 5.128 * math.sin(math.radians(mean_dist_moon))
        dist_moon = 385000000.0 - 20905000.0 * math.cos(math.radians(mean_anom_moon))
        
        xs_moon = dist_moon * math.cos(math.radians(ecl_lon_moon)) * math.cos(math.radians(ecl_lat_moon))
        ys_moon = dist_moon * (math.cos(math.radians(obliquity)) * math.sin(math.radians(ecl_lon_moon)) * math.cos(math.radians(ecl_lat_moon)) - math.sin(math.radians(obliquity)) * math.sin(math.radians(ecl_lat_moon)))
        zs_moon = dist_moon * (math.sin(math.radians(obliquity)) * math.sin(math.radians(ecl_lon_moon)) * math.cos(math.radians(ecl_lat_moon)) + math.cos(math.radians(obliquity)) * math.sin(math.radians(ecl_lat_moon)))
        
        r_sta = math.sqrt(X**2 + Y**2 + Z**2)
        if r_sta == 0: return 0.0, 0.0, 0.0
        
        rx, ry, rz = X/r_sta, Y/r_sta, Z/r_sta
        
        def deformacion_cuerpo(mass_ratio, R_body, xs, ys, zs):
            dist_body = math.sqrt(xs**2 + ys**2 + zs**2)
            if dist_body == 0: return 0.0, 0.0, 0.0
            ux, uy, uz = xs/dist_body, ys/dist_body, zs/dist_body
            cos_theta = rx*ux + ry*uy + rz*uz
            
            p2 = 1.5 * cos_theta**2 - 0.5
            p2_prime = 3.0 * cos_theta
            
            coef = (GM_earth / Re**2) * mass_ratio * (Re / dist_body)**3 * Re
            
            dr_radial = h2 * coef * p2
            dr_tangent = l2 * coef * p2_prime
            
            dx = dr_radial * rx + dr_tangent * (ux - cos_theta * rx)
            dy = dr_radial * ry + dr_tangent * (uy - cos_theta * ry)
            dz = dr_radial * rz + dr_tangent * (uz - cos_theta * rz)
            return dx, dy, dz

        dx_sun, dy_sun, dz_sun = deformacion_cuerpo(GM_sun/GM_earth, dist_sun, xs_sun, ys_sun, zs_sun)
        dx_moon, dy_moon, dz_moon = deformacion_cuerpo(GM_moon/GM_earth, dist_moon, xs_moon, ys_moon, zs_moon)
        
        return dx_sun + dx_moon, dy_sun + dy_moon, dz_sun + dz_moon
    except:
        return 0.0, 0.0, 0.0 

def geodesicas_a_ecef(lat_deg, lon_deg, alt):
    a, e2 = 6378137.0, 0.0066943799901413155
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    N = a / math.sqrt(1 - e2 * (math.sin(lat) ** 2))
    return (N + alt) * math.cos(lat) * math.cos(lon), (N + alt) * math.cos(lat) * math.sin(lon), (N * (1 - e2) + alt) * math.sin(lat)

def ecef_a_geodesicas(x, y, z):
    a, e2 = 6378137.0, 0.0066943799901413155
    b = math.sqrt(a**2 * (1 - e2)); ep2 = (a**2 - b**2) / b**2
    p = math.sqrt(x**2 + y**2); th = math.atan2(a * z, b * p)
    lat = math.atan2((z + ep2 * b * (math.sin(th) ** 3)), (p - e2 * a * (math.cos(th) ** 3)))
    N = a / math.sqrt(1 - e2 * (math.sin(lat) ** 2))
    return math.degrees(lat), math.degrees(math.atan2(y, x)), p / math.cos(lat) - N

def geodesicas_a_utm(lat, lon, force_zone=19):
    a, e2 = 6378137.0, 0.0066943799901413155
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    LongOrig = math.radians((force_zone - 1) * 6 - 180 + 3)
    ep2 = e2 / (1 - e2)
    N = a / math.sqrt(1 - e2 * math.sin(lat_r)**2)
    T = math.tan(lat_r)**2; C = ep2 * math.cos(lat_r)**2; A = math.cos(lat_r) * (lon_r - LongOrig)
    M = a * ((1 - e2/4 - 3*e2**2/64 - 5*e2**3/256)*lat_r - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024)*math.sin(2*lat_r) + (15*e2**2/256 + 45*e2**3/1024)*math.sin(4*lat_r) - (35*e2**3/3072)*math.sin(6*lat_r))
    Easting = 0.9996 * N * (A + (1-T+C)*A**3/6 + (5-18*T+T**2+72*C-58*ep2)*A**5/120) + 500000.0
    Northing = 0.9996 * (M + N*math.tan(lat_r)*(A**2/2 + (5-T+9*C+4*C**2)*A**4/24 + (61-58*T+T**2+600*C-330*ep2)*A**6/720))
    return (Northing + 10000000.0 if lat < 0 else Northing), Easting

def utm_a_geodesicas(easting, northing, zone=19, hemisferio='N'):
    a, e2 = 6378137.0, 0.0066943799901413155
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    x, y = easting - 500000.0, northing if hemisferio.upper() == 'N' else northing - 10000000.0
    m = y / 0.9996; mu = m / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
    phi1_rad = mu + (3*e1/2 - 27*e1**3/32)*math.sin(2*mu) + (21*e1**2/16 - 55*e1**4/32)*math.sin(4*mu)
    n1 = a / math.sqrt(1 - e2*math.sin(phi1_rad)**2)
    t1, c1 = math.tan(phi1_rad)**2, e2 / (1 - e2) * math.cos(phi1_rad)**2
    r1 = a * (1 - e2) / ((1 - e2*math.sin(phi1_rad)**2)**1.5)
    d = x / (n1 * 0.9996)
    lat_rad = phi1_rad - (n1*math.tan(phi1_rad)/r1) * (d**2/2 - (5 + 3*t1 + 10*c1)*d**4/24)
    lon_rad = (d - (1 + 2*t1 + c1)*d**3/6) / math.cos(phi1_rad)
    lon_origen = math.radians((zone - 1) * 6 - 180 + 3)
    return math.degrees(lat_rad), math.degrees(lon_rad + lon_origen), 0.0

def calcular_topocentricas(xs, ys, zs, X_usr, Y_usr, Z_usr):
    lat_val, lon_val, alt_val = ecef_a_geodesicas(X_usr, Y_usr, Z_usr)
    lat_r = math.radians(lat_val)
    lon_r = math.radians(lon_val)
    dx, dy, dz = xs - X_usr, ys - Y_usr, zs - Z_usr
    sin_lat, cos_lat = math.sin(lat_r), math.cos(lat_r)
    sin_lon, cos_lon = math.sin(lon_r), math.cos(lon_r)
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    if dist < 1e-6: return 0.0, 0.0
    val_asin = max(-1.0, min(1.0, u / dist))
    el = math.degrees(math.asin(val_asin))
    az = math.degrees(math.atan2(e, n))
    if az < 0: az += 360.0
    return el, az

def calcular_posicion_satelite_wgs84(eph, t_emision, tau_vuelo, sys_char='G'):
    if not eph or eph['sqrtA'] <= 0.0: return None
    mu_sys = 3.986004418e14 if sys_char in 'EC' else MU
    omega_e_sys = 7.292115e-5 if sys_char == 'C' else OMEGA_E
    F_REL = -4.442807633e-10
    
    A = eph['sqrtA'] ** 2
    n0 = math.sqrt(mu_sys / (A ** 3))
    t_k = t_emision - eph['Toe']
    if sys_char == 'C': t_k -= 14.0
    if t_k > 302400: t_k -= 604800
    elif t_k < -302400: t_k += 604800
    M_k = eph['M0'] + (n0 + eph['Delta_n']) * t_k; E_k = M_k
    for _ in range(5): E_k = M_k + eph['e'] * math.sin(E_k)
    
    delta_tr = F_REL * eph['e'] * eph['sqrtA'] * math.sin(E_k)
    dt_sat = eph['af0'] + eph['af1'] * t_k + eph['af2'] * (t_k ** 2) + delta_tr
    
    nu_k = math.atan2((math.sqrt(1 - eph['e']**2) * math.sin(E_k)), (math.cos(E_k) - eph['e']))
    phi_k = nu_k + eph['omega']
    u_k = phi_k + eph['Cus'] * math.sin(2 * phi_k) + eph['Cuc'] * math.cos(2 * phi_k)
    r_k = A * (1 - eph['e'] * math.cos(E_k)) + eph['Crs'] * math.sin(2 * phi_k) + eph['Crc'] * math.cos(2 * phi_k)
    i_k = eph['i0'] + eph['Cic'] * math.cos(2 * phi_k) + eph['Cis'] * math.sin(2 * phi_k) + eph['IDOT'] * t_k
    x_k, y_k = r_k * math.cos(u_k), r_k * math.sin(u_k)
    omega_k = eph['OMEGA'] + (eph['OMEGA_DOT'] - omega_e_sys) * t_k - omega_e_sys * eph['Toe']
    xs = x_k * math.cos(omega_k) - y_k * math.cos(i_k) * math.sin(omega_k)
    ys = x_k * math.sin(omega_k) + y_k * math.cos(i_k) * math.cos(omega_k)
    zs = y_k * math.sin(i_k)
    theta = omega_e_sys * tau_vuelo
    return (xs * math.cos(theta) + ys * math.sin(theta), -xs * math.sin(theta) + ys * math.cos(theta), zs, dt_sat)

# =====================================================================
# ENRUTADOR AUTOMÁTICO BASADO EN SELECCIÓN DE INTERFAZ DE USUARIO
# =====================================================================
def analizar_calidad_y_senales_rinex(obs_b, obs_r, modo_hardware="iguales", max_gap_tolerado=0.5):
    tows_b = sorted(list(obs_b.keys()), key=lambda k: obs_b[k].get('_meta', (0,0,0,0,0,0)))
    tows_r = sorted(list(obs_r.keys()), key=lambda k: obs_r[k].get('_meta', (0,0,0,0,0,0)))
    
    if not tows_b or not tows_r: return "MODO_D_DGPS", 0.0, "Cero épocas. Archivo vacío o corrupto."
    
    sync_epochs = min(len(tows_b), len(tows_r))
    total_eval = max(len(tows_b), len(tows_r))
    ratio_sync = (sync_epochs / total_eval) if total_eval > 0 else 0.0
    
    tiene_l5_real = False
    for t in tows_r[:20]:
        for s, d in obs_r[t].items():
            if s != '_meta' and 'L5' in d and d['L5'] != 0.0:
                tiene_l5_real = True
                break
        if tiene_l5_real: break

    if tiene_l5_real and modo_hardware != "iguales":
        return "MODO_B_ASINCRONO", ratio_sync, "Análisis de Señal: L1/L5 detectada y Teléfonos Distintos -> Enrutando a Módulo B (Asincrónico)."
    else:
        return "MODO_D_DGPS", ratio_sync, "Análisis de Señal: Código Puro o Teléfonos Iguales -> Enrutando a Módulo D (DGPS Estricto)."

# =====================================================================
# AISLAMIENTO DE OBSERVABLES (MODO B Y MODO D)
# =====================================================================
def aislar_diferencias_MODO_B(obs_b, obs_r):
    sd_suavizada = {}
    for tow in sorted(list(obs_r.keys()), key=lambda k: obs_r[k].get('_meta', (0,0,0,0,0,0))):
        if tow not in obs_b: continue
        
        sd_epoca = {'_meta': obs_r[tow]['_meta'], '_tow_b': tow}
        for s, d_r in obs_r[tow].items():
            if s == '_meta' or s not in obs_b[tow]: continue
            d_b = obs_b[tow]
            
            freq = 'L1' 
            if 'C5' in d_b[s] and 'C5' in d_r and 'L5' in d_b[s] and 'L5' in d_r:
                freq = 'L5' 
            elif not ('C1' in d_b[s] and 'C1' in d_r): continue
            
            pr_b = d_b[s]['C5'] if freq == 'L5' else d_b[s]['C1']
            pr_r = d_r['C5'] if freq == 'L5' else d_r['C1']
            
            snr_b = d_b[s].get('S5', 30.0) if freq == 'L5' else d_b[s].get('S1', 30.0)
            snr_r = d_r.get('S5', 30.0) if freq == 'L5' else d_r.get('S1', 30.0)
            
            sd_P = pr_r - pr_b
            
            sd_epoca[s] = {
                'sd_P': sd_P,
                'pr_b': pr_b, 'pr_r': pr_r,
                'snr': min(snr_b, snr_r)
            }
        if len(sd_epoca) > 1: sd_suavizada[tow] = sd_epoca
    return sd_suavizada

def aislar_diferencias_MODO_D(obs_b, obs_r):
    sd_suavizada = {}
    for tow in sorted(list(obs_r.keys()), key=lambda k: obs_r[k].get('_meta', (0,0,0,0,0,0))):
        if tow not in obs_b: continue
        
        sd_epoca = {'_meta': obs_r[tow]['_meta'], '_tow_b': tow}
        for s, d_r in obs_r[tow].items():
            if s == '_meta' or s not in obs_b[tow]: continue
            d_b = obs_b[tow][s]
            
            freq = 'C1'
            if 'C5' in d_b and 'C5' in d_r:
                if 'C1' not in d_b or 'C1' not in d_r:
                    freq = 'C5'
            elif not ('C1' in d_b and 'C1' in d_r):
                continue
                
            pr_b = d_b['C5'] if freq == 'C5' else d_b['C1']
            pr_r = d_r['C5'] if freq == 'C5' else d_r['C1']
            
            snr_b = d_b.get('S5', 30.0) if freq == 'C5' else d_b.get('S1', 30.0)
            snr_r = d_r.get('S5', 30.0) if freq == 'C5' else d_r.get('S1', 30.0)
            
            sd_P = pr_r - pr_b
            
            sd_epoca[s] = {
                'sd_P': sd_P,
                'pr_b': pr_b, 'pr_r': pr_r,
                'snr': min(snr_b, snr_r),
                'sys': s[0]
            }
        if len(sd_epoca) > 1: sd_suavizada[tow] = sd_epoca
    return sd_suavizada

# =====================================================================
# VÍA 1 -> MÓDULO B: MOTOR IRLS ASINCRÓNICO CON PROYECCIÓN ENU RIGUROSA
# =====================================================================
def calcular_IRLS_MODO_B(sd_epoca, nav, sp3, X_b, Y_b, Z_b, tr, mask_angle, min_snr=18.5, geom_cache=None):
    try:
        tow_b = sd_epoca.get('_tow_b', tr)
        y_m, m_m, d_m, h_m, mn_m, sec_m = sd_epoca['_meta']
        
        if geom_cache is not None and 'tide' in geom_cache:
            dx_tide, dy_tide, dz_tide = geom_cache['tide']
        else:
            dx_tide, dy_tide, dz_tide = correccion_mareas_solidas(X_b, Y_b, Z_b, tow_b, y_m, m_m, d_m)
            if geom_cache is not None:
                if tr not in geom_cache: geom_cache[tr] = {}
                geom_cache['tide'] = (dx_tide, dy_tide, dz_tide)
        
        X_b_corr, Y_b_corr, Z_b_corr = X_b + dx_tide, Y_b + dy_tide, Z_b + dz_tide
        X_iter, Y_iter, Z_iter = X_b_corr, Y_b_corr, Z_b_corr 
        
        if geom_cache is not None and 'R_enu' in geom_cache:
            R_enu = geom_cache['R_enu']
        else:
            lat_init, lon_init, _ = ecef_a_geodesicas(X_b_corr, Y_b_corr, Z_b_corr)
            R_enu = obtener_matriz_rotacion_enu(lat_init, lon_init)
            if geom_cache is not None: geom_cache['R_enu'] = R_enu

        sat_positions = {}
        for s, d in sd_epoca.items():
            if s == '_meta' or s == '_tow_b' or d['sd_P'] is None: continue 
            
            sp_r = None; el_r = 0.0; az_r = 0.0
            if geom_cache is not None and tr in geom_cache and s in geom_cache[tr]:
                sp_r = geom_cache[tr][s]['sp_r']
                el_r = geom_cache[tr][s]['el']
                az_r = geom_cache[tr][s]['az']
            else:
                tau_r = d['pr_r'] / C_LIGHT
                t_emision_r = tr - tau_r
                
                if sp3 and s in sp3:
                    sp3_res_r = interpolate_sp3(sp3, s, t_emision_r)
                    if sp3_res_r:
                        theta_r = OMEGA_E * tau_r
                        xs_r = sp3_res_r[0] * math.cos(theta_r) + sp3_res_r[1] * math.sin(theta_r)
                        ys_r = -sp3_res_r[0] * math.sin(theta_r) + sp3_res_r[1] * math.cos(theta_r)
                        sp_r = (xs_r, ys_r, sp3_res_r[2], sp3_res_r[3])
                        
                if not sp_r:
                    sp_r = calcular_posicion_satelite_wgs84(seleccionar_efemeride_optima(nav.get(s), t_emision_r), t_emision_r, tau_r, s[0])
                    
                if sp_r:
                    el_r, az_r = calcular_topocentricas(sp_r[0], sp_r[1], sp_r[2], X_b_corr, Y_b_corr, Z_b_corr)
                    if geom_cache is not None:
                        if tr not in geom_cache: geom_cache[tr] = {}
                        geom_cache[tr][s] = {'sp_r': sp_r, 'el': el_r, 'az': az_r}
                
            if sp_r:
                snr_val = d.get('snr', 30.0)
                if el_r >= max(8.0, mask_angle) and snr_val >= min_snr:
                    sat_positions[s] = {'sp': sp_r, 'el': el_r, 'az': az_r, 'sd_P': d['sd_P'], 'snr': snr_val}
        
        if len(sat_positions) < 4: return None, "FAILED", None
        
        # [MODIFICACIÓN QUIRÚRGICA: Truncamiento Heurístico (Top 12 Sats)]
        sorted_sats = sorted(
            sat_positions.keys(), 
            key=lambda k: sat_positions[k].get('snr', 30.0) * math.sin(math.radians(sat_positions[k]['el'])), 
            reverse=True
        )
        sat_list_full = sorted_sats[:12]
        
        constellations = set([s[0] for s in sat_list_full])
        ref_sats = {}
        sat_list = []
        
        for c in constellations:
            c_sats = [s for s in sat_list_full if s[0] == c]
            if len(c_sats) >= 2:
                ref_sats[c] = max(c_sats, key=lambda k: sat_positions[k]['el'])
                c_sats.remove(ref_sats[c])
                sat_list.extend(c_sats)
        
        if len(sat_list) < 3: return None, "FAILED", None
        
        base_rho = {}
        for s_key, s_data in sat_positions.items():
            base_rho[s_key] = math.sqrt((s_data['sp'][0]-X_b_corr)**2 + (s_data['sp'][1]-Y_b_corr)**2 + (s_data['sp'][2]-Z_b_corr)**2)
            
        w_P_cache = {}
        for i, s in enumerate(sat_list):
            el_rad_i = math.radians(sat_positions[s]['el'])
            el_rad_ref = math.radians(sat_positions[ref_sats[s[0]]]['el'])
            snr_i = sat_positions[s].get('snr', 30.0)
            snr_ref = sat_positions[ref_sats[s[0]]].get('snr', 30.0)
            factor_snr_i = min(1.0, max(0.1, snr_i / 40.0))
            factor_snr_ref = min(1.0, max(0.1, snr_ref / 40.0))
            w_P_cache[s] = (math.sin(el_rad_i) ** 2) * (math.sin(el_rad_ref) ** 2) * factor_snr_i * factor_snr_ref

        prev_residuals = [0.0] * len(sat_list)
        pdop = 99.9

        # [MODIFICACIÓN QUIRÚRGICA: Límite estricto a 4 iteraciones]
        for iteracion in range(4):
            H = []; L = []; W_diag = [] 
            
            ref_calcs = {}
            for c, r_sat in ref_sats.items():
                r_data = sat_positions[r_sat]
                dist_ref_r = math.sqrt((r_data['sp'][0]-X_iter)**2 + (r_data['sp'][1]-Y_iter)**2 + (r_data['sp'][2]-Z_iter)**2)
                
                ref_calcs[c] = {
                    'dist_ref_r': dist_ref_r,
                    'SD_P_calc_ref': dist_ref_r - base_rho[r_sat],
                    'sp': r_data['sp'],
                    'el': r_data['el'],
                    'snr': r_data.get('snr', 30.0),
                    'sd_P': r_data['sd_P']
                }
            
            res_idx = 0
            for i, s in enumerate(sat_list):
                c = s[0]
                data = sat_positions[s]
                rc = ref_calcs[c]
                
                dist_i_r = math.sqrt((data['sp'][0]-X_iter)**2 + (data['sp'][1]-Y_iter)**2 + (data['sp'][2]-Z_iter)**2)
                
                SD_P_calc_i = dist_i_r - base_rho[s]
                DD_P_calc = SD_P_calc_i - rc['SD_P_calc_ref']
                
                u_i_ecef = [-(data['sp'][0] - X_iter) / dist_i_r, -(data['sp'][1] - Y_iter) / dist_i_r, -(data['sp'][2] - Z_iter) / dist_i_r]
                u_rc_ecef = [-(rc['sp'][0] - X_iter) / rc['dist_ref_r'], -(rc['sp'][1] - Y_iter) / rc['dist_ref_r'], -(rc['sp'][2] - Z_iter) / rc['dist_ref_r']]
                
                u_i_enu = multiplicar_matriz_vector_3x3(R_enu, u_i_ecef)
                u_rc_enu = multiplicar_matriz_vector_3x3(R_enu, u_rc_ecef)
                
                dx_geom = [u_i_enu[0] - u_rc_enu[0], u_i_enu[1] - u_rc_enu[1], u_i_enu[2] - u_rc_enu[2]]
                
                w_P = w_P_cache[s]
                DD_P_obs = data['sd_P'] - rc['sd_P']
                res_P = DD_P_obs - DD_P_calc
                
                L.append([res_P])
                H.append(dx_geom)
                
                if iteracion > 0:
                    w_P = w_P / max(1.0, abs(prev_residuals[res_idx]) / 2.0)
                W_diag.append(w_P)
                res_idx += 1

            H_T = transpose_matrix(H)
            if not H_T or not W_diag: return None, "FAILED", None
            
            try:
                H_T_W = [[H_T[r][idx] * W_diag[idx] for idx in range(len(W_diag))] for r in range(len(H_T))]
            except IndexError:
                return None, "FAILED", None

            N_mat = matmul(H_T_W, H)
            for r in range(len(N_mat)):
                N_mat[r][r] += abs(N_mat[r][r]) * 1e-6 + 1e-6
                
            U_vec = matmul(H_T_W, L)
            Q = invert_matrix_nxn(N_mat)
            if not Q: return None, "FAILED", None
            
            pdop = math.sqrt(Q[0][0] + Q[1][1] + Q[2][2])
            Delta_ENU = matmul(Q, U_vec)
            if not Delta_ENU or len(Delta_ENU) < 3 or not Delta_ENU[0]: return None, "FAILED", None

            dE, dN, dU = Delta_ENU[0][0], Delta_ENU[1][0], Delta_ENU[2][0]
            dX = R_enu[0][0]*dE + R_enu[1][0]*dN + R_enu[2][0]*dU
            dY = R_enu[0][1]*dE + R_enu[1][1]*dN + R_enu[2][1]*dU
            dZ = R_enu[0][2]*dE + R_enu[1][2]*dN + R_enu[2][2]*dU

            X_iter += dX; Y_iter += dY; Z_iter += dZ
                
            prev_residuals = []
            for r in range(len(H)):
                v_val = sum(H[r][idx] * Delta_ENU[idx][0] for idx in range(len(H[0]))) - L[r][0]
                prev_residuals.append(v_val)
            
            # [MODIFICACIÓN QUIRÚRGICA: Relajación Freno de Convergencia a 1cm]
            if max(abs(dE), abs(dN), abs(dU)) < 1e-2:
                return (X_iter - dx_tide, Y_iter - dy_tide, Z_iter - dz_tide), "FLOAT (IRLS Rescate ENU)", pdop
                
        return (X_iter - dx_tide, Y_iter - dy_tide, Z_iter - dz_tide), "FLOAT (IRLS Rescate ENU)", pdop
    except Exception as e:
        return None, f"FAILED_EXCEPTION:_{str(e)}", None

# =====================================================================
# VÍA 2 -> MÓDULO D: NUEVO MOTOR DGPS ESTRICTO CÓDIGO PURO CON ENU
# =====================================================================
def calcular_IRLS_MODO_D(sd_epoca, nav, sp3, X_b, Y_b, Z_b, tr, mask_angle, min_snr=18.5, geom_cache=None):
    try:
        tow_b = sd_epoca.get('_tow_b', tr)
        y_m, m_m, d_m, h_m, mn_m, sec_m = sd_epoca['_meta']
        
        if geom_cache is not None and 'tide' in geom_cache:
            dx_tide, dy_tide, dz_tide = geom_cache['tide']
        else:
            dx_tide, dy_tide, dz_tide = correccion_mareas_solidas(X_b, Y_b, Z_b, tow_b, y_m, m_m, d_m)
            if geom_cache is not None:
                if tr not in geom_cache: geom_cache[tr] = {}
                geom_cache['tide'] = (dx_tide, dy_tide, dz_tide)
                
        X_b_corr, Y_b_corr, Z_b_corr = X_b + dx_tide, Y_b + dy_tide, Z_b + dz_tide
        X_iter, Y_iter, Z_iter = X_b_corr, Y_b_corr, Z_b_corr 
        
        if geom_cache is not None and 'R_enu' in geom_cache:
            R_enu = geom_cache['R_enu']
        else:
            lat_init, lon_init, _ = ecef_a_geodesicas(X_b_corr, Y_b_corr, Z_b_corr)
            R_enu = obtener_matriz_rotacion_enu(lat_init, lon_init)
            if geom_cache is not None: geom_cache['R_enu'] = R_enu
        
        sat_positions = {}
        for s, d in sd_epoca.items():
            if s == '_meta' or s == '_tow_b' or d['sd_P'] is None: continue 
            
            sp_r = None; el_r = 0.0; az_r = 0.0
            if geom_cache is not None and tr in geom_cache and s in geom_cache[tr]:
                sp_r = geom_cache[tr][s]['sp_r']
                el_r = geom_cache[tr][s]['el']
                az_r = geom_cache[tr][s]['az']
            else:
                tau_r = d['pr_r'] / C_LIGHT
                t_emision_r = tr - tau_r
                
                if sp3 and s in sp3:
                    sp3_res_r = interpolate_sp3(sp3, s, t_emision_r)
                    if sp3_res_r:
                        theta_r = OMEGA_E * tau_r
                        xs_r = sp3_res_r[0] * math.cos(theta_r) + sp3_res_r[1] * math.sin(theta_r)
                        ys_r = -sp3_res_r[0] * math.sin(theta_r) + sp3_res_r[1] * math.cos(theta_r)
                        sp_r = (xs_r, ys_r, sp3_res_r[2], sp3_res_r[3])
                        
                if not sp_r:
                    sp_r = calcular_posicion_satelite_wgs84(seleccionar_efemeride_optima(nav.get(s), t_emision_r), t_emision_r, tau_r, s[0])
                    
                if sp_r:
                    el_r, az_r = calcular_topocentricas(sp_r[0], sp_r[1], sp_r[2], X_b_corr, Y_b_corr, Z_b_corr)
                    if geom_cache is not None:
                        if tr not in geom_cache: geom_cache[tr] = {}
                        geom_cache[tr][s] = {'sp_r': sp_r, 'el': el_r, 'az': az_r}
                
            if sp_r:
                snr_val = d.get('snr', 30.0)
                if el_r >= max(8.0, mask_angle) and snr_val >= min_snr:
                    sat_positions[s] = {'sp': sp_r, 'el': el_r, 'az': az_r, 'sd_P': d['sd_P'], 'snr': snr_val}
        
        if len(sat_positions) < 4: return None, "FAILED", None
        
        # [MODIFICACIÓN QUIRÚRGICA: Truncamiento Heurístico (Top 12 Sats)]
        sorted_sats = sorted(
            sat_positions.keys(), 
            key=lambda k: sat_positions[k].get('snr', 30.0) * math.sin(math.radians(sat_positions[k]['el'])), 
            reverse=True
        )
        sat_list_full = sorted_sats[:12]
        
        constellations = set([s[0] for s in sat_list_full])
        ref_sats = {}
        sat_list = []
        
        for c in constellations:
            c_sats = [s for s in sat_list_full if s[0] == c]
            if len(c_sats) >= 2:
                ref_sats[c] = max(c_sats, key=lambda k: sat_positions[k]['el'])
                c_sats.remove(ref_sats[c])
                sat_list.extend(c_sats)
        
        if len(sat_list) < 3: return None, "FAILED", None
        
        base_rho = {}
        for s_key, s_data in sat_positions.items():
            base_rho[s_key] = math.sqrt((s_data['sp'][0]-X_b_corr)**2 + (s_data['sp'][1]-Y_b_corr)**2 + (s_data['sp'][2]-Z_b_corr)**2)
            
        w_P_cache = {}
        for i, s in enumerate(sat_list):
            el_rad_i = math.radians(sat_positions[s]['el'])
            el_rad_ref = math.radians(sat_positions[ref_sats[s[0]]]['el'])
            snr_i = sat_positions[s].get('snr', 30.0)
            snr_ref = sat_positions[ref_sats[s[0]]].get('snr', 30.0)
            factor_snr_i = min(1.0, max(0.1, snr_i / 40.0))
            factor_snr_ref = min(1.0, max(0.1, snr_ref / 40.0))
            w_P_cache[s] = (math.sin(el_rad_i) ** 2) * (math.sin(el_rad_ref) ** 2) * factor_snr_i * factor_snr_ref

        prev_residuals = [0.0] * len(sat_list)
        pdop = 99.9

        # [MODIFICACIÓN QUIRÚRGICA: Límite estricto a 4 iteraciones]
        for iteracion in range(4):
            H = []; L = []; W_diag = [] 
            
            ref_calcs = {}
            for c, r_sat in ref_sats.items():
                r_data = sat_positions[r_sat]
                dist_ref_r = math.sqrt((r_data['sp'][0]-X_iter)**2 + (r_data['sp'][1]-Y_iter)**2 + (r_data['sp'][2]-Z_iter)**2)
                
                ref_calcs[c] = {
                    'dist_ref_r': dist_ref_r,
                    'SD_P_calc_ref': dist_ref_r - base_rho[r_sat],
                    'sp': r_data['sp'],
                    'el': r_data['el'],
                    'snr': r_data.get('snr', 30.0),
                    'sd_P': r_data['sd_P']
                }
            
            res_idx = 0
            for i, s in enumerate(sat_list):
                c = s[0]
                data = sat_positions[s]
                rc = ref_calcs[c]
                
                dist_i_r = math.sqrt((data['sp'][0]-X_iter)**2 + (data['sp'][1]-Y_iter)**2 + (data['sp'][2]-Z_iter)**2)
                
                SD_P_calc_i = dist_i_r - base_rho[s]
                DD_P_calc = SD_P_calc_i - rc['SD_P_calc_ref']
                
                u_i_ecef = [-(data['sp'][0] - X_iter) / dist_i_r, -(data['sp'][1] - Y_iter) / dist_i_r, -(data['sp'][2] - Z_iter) / dist_i_r]
                u_rc_ecef = [-(rc['sp'][0] - X_iter) / rc['dist_ref_r'], -(rc['sp'][1] - Y_iter) / rc['dist_ref_r'], -(rc['sp'][2] - Z_iter) / rc['dist_ref_r']]
                
                u_i_enu = multiplicar_matriz_vector_3x3(R_enu, u_i_ecef)
                u_rc_enu = multiplicar_matriz_vector_3x3(R_enu, u_rc_ecef)
                
                dx_geom = [u_i_enu[0] - u_rc_enu[0], u_i_enu[1] - u_rc_enu[1], u_i_enu[2] - u_rc_enu[2]]
                
                w_P = w_P_cache[s]
                DD_P_obs = data['sd_P'] - rc['sd_P']
                res_P = DD_P_obs - DD_P_calc
                
                L.append([res_P])
                H.append(dx_geom)
                
                if iteracion > 0:
                    w_P = w_P / max(1.0, abs(prev_residuals[res_idx]) / 2.0)
                W_diag.append(w_P)
                res_idx += 1

            H_T = transpose_matrix(H)
            if not H_T or not W_diag: return None, "FAILED", None
            
            try:
                H_T_W = [[H_T[r][idx] * W_diag[idx] for idx in range(len(W_diag))] for r in range(len(H_T))]
            except IndexError:
                return None, "FAILED", None

            N_mat = matmul(H_T_W, H)
            for r in range(len(N_mat)):
                N_mat[r][r] += abs(N_mat[r][r]) * 1e-6 + 1e-6
                
            U_vec = matmul(H_T_W, L)
            Q = invert_matrix_nxn(N_mat)
            if not Q: return None, "FAILED", None
            
            pdop = math.sqrt(Q[0][0] + Q[1][1] + Q[2][2])
            Delta_ENU = matmul(Q, U_vec)
            if not Delta_ENU or len(Delta_ENU) < 3 or not Delta_ENU[0]: return None, "FAILED", None

            dE, dN, dU = Delta_ENU[0][0], Delta_ENU[1][0], Delta_ENU[2][0]
            dX = R_enu[0][0]*dE + R_enu[1][0]*dN + R_enu[2][0]*dU
            dY = R_enu[0][1]*dE + R_enu[1][1]*dN + R_enu[2][1]*dU
            dZ = R_enu[0][2]*dE + R_enu[1][2]*dN + R_enu[2][2]*dU

            X_iter += dX; Y_iter += dY; Z_iter += dZ
                
            prev_residuals = []
            for r in range(len(H)):
                v_val = sum(H[r][idx] * Delta_ENU[idx][0] for idx in range(len(H[0]))) - L[r][0]
                prev_residuals.append(v_val)
            
            # [MODIFICACIÓN QUIRÚRGICA: Relajación Freno de Convergencia a 1cm]
            if max(abs(dE), abs(dN), abs(dU)) < 1e-2:
                return (X_iter - dx_tide, Y_iter - dy_tide, Z_iter - dz_tide), "FLOAT (DGPS Código Puro ENU)", pdop
                
        return (X_iter - dx_tide, Y_iter - dy_tide, Z_iter - dz_tide), "FLOAT (DGPS Código Puro ENU)", pdop
    except Exception as e:
        return None, f"FAILED_EXCEPTION:_{str(e)}", None

# =====================================================================
# ESTADÍSTICAS Y FILTRADO VINCULANTE (HARD FILTER FLEXIBILIZADO)
# =====================================================================
def estadistica_desacoplada(coordenadas, conf_plani, conf_alti, err_hor_max, err_ver_max, medianas=None):
    if not coordenadas: return None, None, None, 0.0, 0.0, 0.0, 0, 0.0
    
    N_list = [c[0] for c in coordenadas]
    E_list = [c[1] for c in coordenadas]
    Z_list = [c[2] for c in coordenadas]

    def get_median(lst):
        s = sorted(lst); n = len(s)
        if n == 0: return 0.0
        return s[n//2] if n % 2 == 1 else (s[n//2 - 1] + s[n//2]) / 2.0

    if medianas:
        med_N, med_E, med_Z = medianas
    else:
        med_N = get_median(N_list)
        med_E = get_median(E_list)
        med_Z = get_median(Z_list)
    
    valid_coords = []
    for c in coordenadas:
        dh = math.hypot(c[0] - med_N, c[1] - med_E)
        dv = abs(c[2] - med_Z)
        
        if (err_hor_max > 0.0 and dh > err_hor_max) or (err_ver_max > 0.0 and dv > err_ver_max):
            continue
        valid_coords.append(c)

    if not valid_coords: return None, None, None, 0.0, 0.0, 0.0, 0, 0.0
    
    N_v = [c[0] for c in valid_coords]
    E_v = [c[1] for c in valid_coords]
    Z_v = [c[2] for c in valid_coords]
    f_v = [c[3] for c in valid_coords if len(c) > 3 and "FIXED" in c[3]]

    def calc_mean_std(arr):
        n = len(arr)
        m = sum(arr) / float(n)
        return m, (math.sqrt(sum((x - m)**2 for x in arr) / float(n)) if n > 1 else 0.0)

    N_m, N_s = calc_mean_std(N_v)
    E_m, E_s = calc_mean_std(E_v)
    Z_m, Z_s = calc_mean_std(Z_v)
    
    N_f = [x for x in N_v if abs(x - N_m) <= (conf_plani * 1.5) * N_s] if N_s > 0.0 else N_v
    E_f = [x for x in E_v if abs(x - E_m) <= (conf_plani * 1.5) * E_s] if E_s > 0.0 else E_v
    Z_f = [x for x in Z_v if abs(x - Z_m) <= (conf_alti * 1.5) * Z_s] if Z_s > 0.0 else Z_v

    fix_ratio = (len(f_v) / float(len(valid_coords))) * 100.0 if valid_coords else 0.0
    return sum(N_f)/float(max(1, len(N_f))), sum(E_f)/float(max(1, len(E_f))), sum(Z_f)/float(max(1, len(Z_f))), N_s, E_s, Z_s, min(len(N_f), len(E_f), len(Z_f)), fix_ratio

# =====================================================================
# GENERADORES DE INFORMES (FRONTEND FORENSE)
# =====================================================================
def generar_informe_homogeneizacion_detallado(base_name, rover_name, base_raw, rover_raw, rover_sinc, modo_str, msg, c_base, c_rover, t_exec):
    def get_stats(obs):
        c = {'G':0, 'E':0, 'C':0, 'R':0, 'S':0, 'J':0}
        tiempos = sorted(list(obs.keys()), key=lambda k: obs[k].get('_meta', (0,0,0,0,0,0)))
        if not tiempos: return c, 0, None, None, 0.0, 0, "Desconocida", 0, 0.0
        
        epocas = len(obs)
        t_ini, t_fin = obs[tiempos[0]]['_meta'], obs[tiempos[-1]]['_meta']
        intervalos = [tiempos[i] - tiempos[i-1] for i in range(1, epocas)]
        tasa_muestreo = sum(intervalos)/float(len(intervalos)) if intervalos else 0.0
        gaps = sum(1 for i in intervalos if i > tasa_muestreo * 1.5)
        
        sats_unicos = set()
        tiene_l1 = False; tiene_l5 = False
        snr_total = 0.0; snr_count = 0
        
        for t in tiempos:
            for s, data in obs[t].items():
                if s != '_meta':
                    if s[0] in c: c[s[0]] += 1
                    sats_unicos.add(s)
                    if 'L1' in data: tiene_l1 = True
                    if 'L5' in data: tiene_l5 = True
                    if 'S1' in data and data['S1'] > 0.0:
                        snr_total += data['S1']; snr_count += 1
                    if 'S5' in data and data['S5'] > 0.0:
                        snr_total += data['S5']; snr_count += 1
                        
        tipo_senal = "L1+L5 (Doble Frecuencia)" if (tiene_l1 and tiene_l5) else ("L1 (Monofrecuencia)" if tiene_l1 else "C1/C5 (Solo Código)")
        avg_snr = (snr_total / float(snr_count)) if snr_count > 0 else 0.0
        total_sats = len(sats_unicos)
        
        return {k: v/float(epocas) for k, v in c.items()}, epocas, t_ini, t_fin, tasa_muestreo, gaps, tipo_senal, total_sats, avg_snr
    
    cb, eb, b_ini, b_fin, tr_b, g_b, senal_b, sats_b, snr_b = get_stats(base_raw)
    cr, er, r_ini, r_fin, tr_r, g_r, senal_r, sats_r, snr_r = get_stats(rover_raw)
    cs, es, s_ini, s_fin, tr_s, _, senal_s, sats_s, snr_s = get_stats(rover_sinc)
    t_exito = (es / float(er) * 100.0) if er > 0 else 0.0
    
    dist_baseline = math.sqrt((c_base['N'] - c_rover['N'])**2 + (c_base['E'] - c_rover['E'])**2 + (c_base['Z'] - c_rover['Z'])**2)
    
    sug_iter = max(3, min(10, math.ceil(1200.0 / float(max(1, es)))))
    
    b_ini_str = f"{b_ini[0]}-{b_ini[1]:02d}-{b_ini[2]:02d} {b_ini[3]:02d}:{b_ini[4]:02d}:{b_ini[5]}" if b_ini else "N/A"
    b_fin_str = f"{b_fin[0]}-{b_fin[1]:02d}-{b_fin[2]:02d} {b_fin[3]:02d}:{b_fin[4]:02d}:{b_fin[5]}" if b_fin else "N/A"
    r_ini_str = f"{r_ini[0]}-{r_ini[1]:02d}-{r_ini[2]:02d} {r_ini[3]:02d}:{r_ini[4]:02d}:{r_ini[5]}" if r_ini else "N/A"
    r_fin_str = f"{r_fin[0]}-{r_fin[1]:02d}-{r_fin[2]:02d} {r_fin[3]:02d}:{r_fin[4]:02d}:{r_fin[5]}" if r_fin else "N/A"
    
    d_ini = datetime.datetime(r_ini[0], r_ini[1], r_ini[2], r_ini[3], r_ini[4], int(r_ini[5])) if r_ini else datetime.datetime.now()
    d_fin = datetime.datetime(r_fin[0], r_fin[1], r_fin[2], r_fin[3], r_fin[4], int(r_fin[5])) if r_fin else datetime.datetime.now()
    duracion_str = str(d_fin - d_ini)
    
    fecha_calculo = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    informe = f"""
========================================================================
    AUDITORÍA FORENSE DE EMPAREJAMIENTO DE ÉPOCAS
========================================================================
[0] TRAZABILIDAD TEMPORAL Y ESPACIAL
  [-] Fecha y Hora de Cálculo   : {fecha_calculo}
  [-] Tiempo de Ejecución Script: {f_14(t_exec)} segundos
  [-] Distancia Base-Rover (3D) : {f_14(dist_baseline)} m
  [-] Coord. Base Fija (N,E,Z)  : {f_14(c_base['N'])}, {f_14(c_base['E'])}, {f_14(c_base['Z'])}
  [-] Coord. Rover Calib (N,E,Z): {f_14(c_rover['N'])}, {f_14(c_rover['E'])}, {f_14(c_rover['Z'])}

[1] PARÁMETROS DE CONTROL (BASE) : {base_name}
  [-] Tipo de Señal GNSS        : {senal_b}
  [-] Satélites Únicos Vistos   : {sats_b}
  [-] Potencia Promedio (SNR)   : {f_14(snr_b)} dBHz
  [-] Épocas Crudas Registradas : {eb}
  [-] Ventana de Medición       : {b_ini_str} al {b_fin_str}

[2] PARÁMETROS DEL MÓVIL (ROVER) : {rover_name}
  [-] Tipo de Señal GNSS        : {senal_r}
  [-] Satélites Únicos Vistos   : {sats_r}
  [-] Potencia Promedio (SNR)   : {f_14(snr_r)} dBHz
  [-] Épocas Crudas Registradas : {er}
  [-] Ventana de Medición       : {r_ini_str} al {r_fin_str}
  [-] Duración Neta Solapamiento: {duracion_str} (HH:MM:SS)

[3] MATRIZ RESULTANTE (ESTRICTA, CON INTERPOLACIÓN DINÁMICA)
  [-] Épocas Útiles Sincronizadas: {es}
  [-] Tasa de Éxito sobre Rover  : {f_14(t_exito)}%
  [-] Iteraciones EKF Sugeridas  : {sug_iter} (Dinámico por densidad)

[4] ENRUTADOR AUTOMÁTICO DE CÁLCULO
  [-] Módulo Asignado           : {modo_str}
  [-] Justificación Técnica     : {msg}
========================================================================
"""
    return informe

def generar_informe_ascii(tipo, p_dict):
    estado_sol = "FLOAT"
    
    if p_dict.get('estrategia') == 'MODO_B_ASINCRONO':
        estado_sol = 'FLOAT (DGPS IRLS Asincrónico + ENU Riguroso)'
    elif p_dict.get('estrategia') == 'MODO_D_DGPS':
        estado_sol = 'FLOAT (DGPS Código Puro Estricto + ENU)'
        
    err_h_str = f"± {f_14(p_dict['err_h'])} m (Vinculante)" if p_dict['err_h'] > 0.0 else 'Inactiva'
    err_v_str = f"± {f_14(p_dict['err_v'])} m (Vinculante)" if p_dict['err_v'] > 0.0 else 'Inactiva'
    sp3_str = p_dict.get('sp3_file') if p_dict.get('sp3_file') else "[CRÍTICO] Fallback no permitido"
    nav_str = p_dict.get('nav_file', "auto_nav.nav")
    
    fecha_calculo = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dist_baseline = math.sqrt((p_dict['b_n'] - p_dict['r_n_calc'])**2 + (p_dict['b_e'] - p_dict['r_e_calc'])**2 + (p_dict['b_z'] - p_dict['r_z_calc'])**2)

    informe = f"""
========================================================================
             INFORME DE PROCESAMIENTO GNSSJP PRO (V18.3)
========================================================================

[*] IDENTIFICACIÓN DE ESTACIONES Y RESULTADO ({estado_sol})
------------------------------------------------------------------------
  [-] Fecha y Hora de Cálculo: {fecha_calculo}
  [-] Punto Base (Pivote)    : {p_dict.get('nombre_base', 'BASE')}
  [-] Punto Móvil (Medido)   : {p_dict.get('nombre_medido', 'ROVER')}
  [-] Altura Antena Base     : {f_14(p_dict.get('h_b', 0.0))} m
  [-] Altura Antena Móvil    : {f_14(p_dict.get('h_r_nuevo', 0.0))} m
  [-] Tiempo de Ejecución    : {f_14(p_dict.get('t_exec', 0.0))} segundos
  [-] Tolerancia Horizontal  : {err_h_str}
  [-] Tolerancia Vertical    : {err_v_str}
  [-] Máscara Elevación      : {f_14(p_dict['mask'])}° (Optimizado libre)
  [-] Filtro Planimétrico    : {f_14(p_dict['cp'])} Sigma
  [-] Filtro Altimétrico     : {f_14(p_dict['ca'])} Sigma

[1] TRAZABILIDAD DEL PROYECTO Y ARCHIVOS
------------------------------------------------------------------------
  [-] Archivo Control (Base) : {p_dict['base_file']}
  [-] Archivo Móvil (Rover)  : {p_dict['rover_file']}
  [-] Archivo Efemérides NAV : {nav_str} (Ionósfera Klobuchar)
  [-] Archivo Preciso SP3    : {sp3_str} (Órbitas y Relojes)
  [-] Proyección Espacial    : WGS84 / UTM Zona {p_dict.get('utm_h', 19)}{p_dict.get('utm_hem', 'N')}
  [-] Distancia Línea Base   : {f_14(dist_baseline)} m

[2] CALIDAD GEOMÉTRICA Y ESTADÍSTICA (QA / QC)
------------------------------------------------------------------------
  [-] PDOP Geométrico Promed.: {f_14(p_dict.get('pdop', 99.9))}
  [-] Ratio Confiab. (LAMBDA): {f_14(p_dict.get('lambda_ratio', 0.0))}
  [-] Error Horizontal (RMS) : ± {f_14(math.hypot(p_dict['std_n'], p_dict['std_e']))} m
  [-] Error Espacial (3D RMS): ± {f_14(math.sqrt(p_dict['std_n']**2 + p_dict['std_e']**2 + p_dict['std_z']**2))} m

[3] RESULTADOS VECTORIALES FINALES (DIFERENCIAL PURO)
------------------------------------------------------------------------
  * COORDENADA DE CONTROL (BASE FIJA TERRENO):
      Norte : {f_14(p_dict['b_n'])} m
      Este  : {f_14(p_dict['b_e'])} m
      Cota  : {f_14(p_dict['b_z'])} m

  * COORDENADA CALCULADA (AJUSTE {estado_sol} AL TERRENO):
      Norte : {f_14(p_dict['r_n_calc'])} m
      Este  : {f_14(p_dict['r_e_calc'])} m
      Cota  : {f_14(p_dict['r_z_calc'])} m
========================================================================
"""
    return informe

# =====================================================================
# RUTAS FLASK (ENRUTADOR AUTÓNOMO Y DOBLE VÍA AISLADA)
# =====================================================================
@app.route('/')
def index():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(base_dir, 'index.html')
    return send_file(index_path)

@app.route('/API/interrumpir', methods=['POST'])
def interrumpir_proceso():
    uid = request.form.get('uid', request.remote_addr)
    ws = get_workspace(uid)
    with open(os.path.join(ws, 'interrupt.flag'), 'w') as f:
        f.write("1")
    return Response("Interrumpido", status=200)

@app.route('/API/tab1_homogenizar', methods=['POST'])
def tab1_homogenizar():
    start_time = time.time()
    
    uid = request.form.get('uid', request.remote_addr)
    ws = get_workspace(uid)
    
    with STATE_LOCK:
        if os.path.exists(ws):
            shutil.rmtree(ws, ignore_errors=True)
        os.makedirs(ws, exist_ok=True)
    
    url_base = request.form.get('url_base')
    url_rover = request.form.get('url_rover')
    
    utm_n = safe_f(request.form.get('utm_norte'), 0.0)
    utm_e = safe_f(request.form.get('utm_este'), 0.0)
    utm_c = safe_f(request.form.get('utm_cota'), 0.0)
    utm_h = safe_i(request.form.get('utm_huso'), 19)
    utm_hem = request.form.get('utm_hemisferio', 'N')
    
    utm_n_r = safe_f(request.form.get('utm_norte_r'), 0.0)
    utm_e_r = safe_f(request.form.get('utm_este_r'), 0.0)
    utm_c_r = safe_f(request.form.get('utm_cota_r'), 0.0)
    
    h_b = safe_f(request.form.get('altura_base'), 0.0) 
    h_r = safe_f(request.form.get('altura_rover'), 0.0) 
    nombre_base = request.form.get('nombre_base', 'BASE_DESCONOCIDA')
    nombre_rover = request.form.get('nombre_rover', 'ROVER_DESCONOCIDO')
    modo_hardware = request.form.get('modo_hardware', 'iguales')

    guardar_estado(uid, 'utm_norte', utm_n)
    guardar_estado(uid, 'utm_este', utm_e)
    guardar_estado(uid, 'utm_cota', utm_c)
    guardar_estado(uid, 'utm_huso', utm_h)
    guardar_estado(uid, 'utm_hemisferio', utm_hem)
    guardar_estado(uid, 'utm_norte_r', utm_n_r)
    guardar_estado(uid, 'utm_este_r', utm_e_r)
    guardar_estado(uid, 'utm_cota_r', utm_c_r)
    guardar_estado(uid, 'altura_base', h_b)
    guardar_estado(uid, 'altura_rover', h_r)
    guardar_estado(uid, 'nombre_base', nombre_base)
    guardar_estado(uid, 'nombre_rover', nombre_rover)
    guardar_estado(uid, 'modo_hardware', modo_hardware)
    
    if not url_base or not url_rover: 
        return Response("> [ERROR CRÍTICO] Enlaces de Google Drive faltantes.\n", mimetype='text/plain')
    
    p_b_raw = os.path.join(ws, 'base_raw.obs')
    p_r_raw = os.path.join(ws, 'rover_calibracion_raw.obs')

    def procesar():
        try:
            yield "> [RED] Descargando RINEX Base desde Google Drive...\n"
            descargar_desde_gdrive(url_base, p_b_raw)
            yield "> [RED] Descargando RINEX Rover desde Google Drive...\n"
            descargar_desde_gdrive(url_rover, p_r_raw)

            yield f"\n> [SISTEMA] Iniciando Etapa 1: Emparejamiento Base Pivote y Rover de Calibración...\n"
            base_raw_dict = parse_rinex_obs_completo(p_b_raw)
            rover_raw_dict = parse_rinex_obs_completo(p_r_raw)
            
            yield "> [ENRUTADOR] Evaluando calidad y señales RINEX...\n"
            modo_str, ratio, msg = analizar_calidad_y_senales_rinex(base_raw_dict, rover_raw_dict, modo_hardware=modo_hardware, max_gap_tolerado=2.0)
            yield f"  [-] Módulo pre-asignado: {modo_str}\n"
            yield f"  [-] Justificación: {msg}\n\n"
            
            base_sinc, rover_sinc = {}, {}
            total_epochs = len(rover_raw_dict)
            c = 0
            tiempos_base_preordenados = sorted(list(base_raw_dict.keys()), key=lambda k: base_raw_dict[k].get('_meta', (0,0,0,0,0,0)))
            
            for tr in sorted(list(rover_raw_dict.keys()), key=lambda k: rover_raw_dict[k].get('_meta', (0,0,0,0,0,0))):
                if time.time() - start_time > 28.0:
                    yield "\n> [ALERTA] Freno de mano de 28.0s alcanzado. Render interrumpiendo Etapa 1...\n"
                    break

                c += 1
                if total_epochs > 0 and c % max(1, total_epochs // 10) == 0: 
                    yield f"[PROGRESO] Cotejando épocas con interpolación dinámica flexible (max_gap=2.0s)... {int((c / float(total_epochs)) * 100.0)}%\n"
                
                base_interp = interpolar_base_a_rover(base_raw_dict, tr, max_gap=2.0, tiempos_base=tiempos_base_preordenados)
                
                if base_interp:
                    base_sinc[tr] = base_interp
                    base_sinc[tr]['_meta'] = rover_raw_dict[tr]['_meta']
                    rover_sinc[tr] = rover_raw_dict[tr]
            
            if not base_sinc: yield "\n> [ERROR FATAL] Cero épocas en común. Revisar rango horario."; return
            p_b_h = os.path.join(ws, 'base_calib_homo.obs')
            p_r_h = os.path.join(ws, 'rover_calib_homo.obs')
            generar_rinex_sincronizado(p_b_raw, p_b_h, base_sinc)
            generar_rinex_sincronizado(p_r_raw, p_r_h, rover_sinc)
            
            guardar_estado(uid, 'base_raw', p_b_raw)
            guardar_estado(uid, 'base_calib_homo', p_b_h)
            guardar_estado(uid, 'rover_calib_homo', p_r_h)
            
            name_base = "Drive_Base_Pivote.obs"
            name_rover = "Drive_Rover_Calib.obs"
            guardar_estado(uid, 'name_base_raw', name_base)
            guardar_estado(uid, 'name_rover_calib_raw', name_rover)
            
            c_base = {'N': utm_n, 'E': utm_e, 'Z': utm_c}
            c_rover = {'N': utm_n_r, 'E': utm_e_r, 'Z': utm_c_r}
            
            id_b = f"{nombre_base} ({name_base})"
            id_r = f"{nombre_rover} ({name_rover})"
            t_exec = time.time() - start_time
            yield generar_informe_homogeneizacion_detallado(id_b, id_r, base_raw_dict, rover_raw_dict, rover_sinc, modo_str, msg, c_base, c_rover, t_exec)
            yield "\n[SUCCESS]"
        except Exception as e: yield f"\n> [ERROR] Falla estructural: {str(e)}"
    return Response(procesar(), mimetype='text/plain', headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'})

@app.route('/API/tab2_efemerides', methods=['POST'])
def tab2_efemerides():
    uid = request.form.get('uid', request.remote_addr)
    ws = get_workspace(uid)
    
    f_sp3 = request.files.get('file_sp3')
    
    sp3_path = None
    if f_sp3 and f_sp3.filename != '':
        sp3_path = os.path.join(ws, 'manual_sp3.sp3')
        f_sp3.save(sp3_path)
        guardar_estado(uid, 'sp3_path', sp3_path)
        guardar_estado(uid, 'name_sp3_file', f_sp3.filename)
    else:
        guardar_estado(uid, 'sp3_path', None)
        guardar_estado(uid, 'name_sp3_file', None)

    def procesar():
        try:
            yield "> [SISTEMA] Iniciando Inyección Híbrida de Efemérides...\n"
            if sp3_path: 
                yield f"  [-] Archivo SP3 Preciso cargado manualmente: {f_sp3.filename}\n"
            else: 
                yield "  [!] ALERTA CRÍTICA: No se detectó archivo SP3. Los módulos bloquearán el cálculo en las siguientes pestañas.\n"

            yield "\n> [RED] Conectando con Red Global de Repositorios GNSS (Extracción Ionosférica NAV)...\n"
            bp = leer_estado(uid, 'base_raw')
            if not bp or not os.path.exists(bp): 
                yield "> [ERROR FATAL] Falta RINEX Base en memoria para extraer fecha.\n"; return
            
            ft = obtener_fecha_obs(bp)
            if not ft: yield "> [ERROR FATAL] Imposible extraer la fecha del RINEX Base.\n"; return
            
            year, month, day = ft[0], ft[1], ft[2]
            dt = datetime.datetime(year, month, day)
            doy = dt.timetuple().tm_yday
            yy = str(year)[-2:]
            
            nav_gz = os.path.join(ws, f"auto_nav_{year}_{doy:03d}.nav.gz")
            nav_path = os.path.join(ws, f"auto_nav_{year}_{doy:03d}.nav")
            
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            if not os.path.exists(nav_path):
                urls_to_try = [
                    f"https://garner.ucsd.edu/pub/rinex/{year}/{doy:03d}/brdc{doy:03d}0.{yy}n.gz",
                    f"http://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy:03d}/brdc{doy:03d}0.{yy}n.gz",
                    f"https://www.epncb.oma.be/ftp/obs/BRDC/{year}/{doy:03d}/brdc{doy:03d}0.{yy}n.gz",
                    f"http://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy:03d}/BRDC00IGS_R_{year}{doy:03d}0000_01D_MN.rnx.gz"
                ]
                
                descargado = False
                for url_nav in urls_to_try:
                    try:
                        protocolo = "HTTP PURO" if url_nav.startswith("http://") else "HTTPS"
                        yield f"  [-] Intentando espejo ({protocolo}): {url_nav.split('/')[-1]}...\n"
                        req = urllib.request.Request(url_nav, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req, context=ctx, timeout=5) as res:
                            with open(nav_gz, 'wb') as f: f.write(res.read())
                        descargado = True
                        yield f"  [+] Descarga exitosa verificada mediante {protocolo}.\n"
                        break 
                    except Exception as e:
                        err_limpio = str(e).replace('<', '[').replace('>', ']')
                        yield f"  [!] Falló ({err_limpio}). Saltando a siguiente servidor...\n"
                        continue
                
                if not descargado:
                    raise Exception("HTTP 404/Timeout Total: La red IGS global bloqueó la conexión o el archivo no existe.")
                
                yield "  [-] Descomprimiendo archivo NAV...\n"
                with gzip.open(nav_gz, 'rb') as f_in, open(nav_path, 'wb') as f_out: 
                    shutil.copyfileobj(f_in, f_out)
                
                if os.path.exists(nav_gz): os.remove(nav_gz)
            
            guardar_estado(uid, 'nav_path', nav_path)
            guardar_estado(uid, 'name_nav_file', os.path.basename(nav_path))
            yield f"  [-] Archivo NAV listo y ensamblado en memoria.\n\n[SUCCESS]"
        except Exception as e:
            yield f"\n> [ERROR FATAL] Fallo en descarga automática NAV: {str(e)}\n"

    return Response(procesar(), mimetype='text/plain', headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'})

@app.route('/API/tab3_calibrar', methods=['POST'])
def tab3_calibrar():
    start_time = time.time()
    
    uid = request.form.get('uid', request.remote_addr)
    ws = get_workspace(uid)
    
    flag_file = os.path.join(ws, 'interrupt.flag')
    if os.path.exists(flag_file):
        os.remove(flag_file)
        
    utm_n = leer_estado(uid, 'utm_norte')
    utm_e = leer_estado(uid, 'utm_este')
    utm_c = leer_estado(uid, 'utm_cota')
    utm_h = leer_estado(uid, 'utm_huso')
    utm_hem = leer_estado(uid, 'utm_hemisferio')
    utm_n_r = leer_estado(uid, 'utm_norte_r')
    utm_e_r = leer_estado(uid, 'utm_este_r')
    utm_c_r = leer_estado(uid, 'utm_cota_r')
    h_b = safe_f(leer_estado(uid, 'altura_base'), 0.0)
    h_r = safe_f(leer_estado(uid, 'altura_rover'), 0.0)
    modo_hardware = leer_estado(uid, 'modo_hardware') or 'iguales'

    p_max_gap = safe_f(request.form.get('param_max_gap'), 0.5)
    p_snr = safe_f(request.form.get('param_snr'), 25.0)
    p_iter = max(1, safe_i(request.form.get('param_iter'), 6))

    def procesar():
        try:
            if utm_e == 0.0 or utm_n == 0.0 or utm_n_r == 0.0 or utm_e_r == 0.0: 
                yield "> [ERROR] Coordenadas Base y Rover no inyectadas correctamente.\n"; return
            
            nav_path = leer_estado(uid, 'nav_path')
            sp3_path = leer_estado(uid, 'sp3_path')
            p_b_h = leer_estado(uid, 'base_calib_homo')
            p_r_h = leer_estado(uid, 'rover_calib_homo')

            if not p_b_h or not p_r_h: 
                yield "> [ERROR FATAL] Faltan archivos RINEX. Ve a la Pestaña 1.\n"; return
                
            if not nav_path or not sp3_path:
                yield "\n> [ERROR CRÍTICO RECHAZADO]\n"
                yield "  [-] El cálculo geodésico estricto prohíbe el uso de broadcast nav para posicionamiento.\n"
                yield "  [-] FALTA ARCHIVO SP3 (Órbitas precisas) o NAV (Modelo Ionosférico).\n"
                yield "  [-] Vuelva a la Pestaña 2 y suba ambos productos obligatorios.\n"; return

            obs_b_raw = parse_rinex_obs_completo(p_b_h)
            obs_r_raw = parse_rinex_obs_completo(p_r_h)
            nav = parse_rinex_nav_real(nav_path)
            sp3 = parse_sp3_preciso(sp3_path)
            
            modo_str, ratio, msg = analizar_calidad_y_senales_rinex(obs_b_raw, obs_r_raw, modo_hardware=modo_hardware, max_gap_tolerado=p_max_gap)
            yield f"> [ENRUTADOR] Análisis completado: {msg}\n"
            
            lat_b, lon_b, _ = utm_a_geodesicas(utm_e, utm_n, utm_h, utm_hem)
            X_b, Y_b, Z_b = geodesicas_a_ecef(lat_b, lon_b, utm_c + h_b)

            geom_cache = {}

            yield f"> [SISTEMA] Iniciando Búsqueda Determinista Libre | {modo_str} (IRLS + ENU | max_gap={p_max_gap}s | iter={p_iter})...\n"
            if modo_str == "MODO_D_DGPS": sd_suavizada = aislar_diferencias_MODO_D(obs_b_raw, obs_r_raw)
            else: sd_suavizada = aislar_diferencias_MODO_B(obs_b_raw, obs_r_raw)
            
            if not sd_suavizada: yield "> [ERROR] No hay épocas sincronizadas válidas.\n"; return
            
            t_sample_full = list(sd_suavizada.keys())
            total_eps = len(t_sample_full)
            
            # [MODIFICACIÓN QUIRÚRGICA: Muestreo del 100% de la data sin límite]
            t_sample = t_sample_full 
            
            yield f"[PROGRESO OPTIMIZADOR RENDER] Muestreo Sistemático Absoluto Activo:\n"
            yield f"  [-] Épocas totales en archivo: {total_eps}\n"
            yield f"  [-] Épocas estadísticas a evaluar: {len(t_sample)}\n"
            
            yield "[PROGRESO] Fase 1: Extracción de Límites y Poblando Caché (Pre-Scan IRLS)...\n"
            coords_raw = []
            for t in t_sample:
                if time.time() - start_time > 28.0:
                    yield "\n> [ALERTA] Freno de mano de 28.0s activado. Abortando Fase 1 para evitar timeout de Render.\n"
                    break
                if os.path.exists(flag_file): break
                
                if modo_str == "MODO_D_DGPS": sem, status, _ = calcular_IRLS_MODO_D(sd_suavizada[t], nav, sp3, X_b, Y_b, Z_b, t, 8.0, min_snr=p_snr, geom_cache=geom_cache)
                else: sem, status, _ = calcular_IRLS_MODO_B(sd_suavizada[t], nav, sp3, X_b, Y_b, Z_b, t, 8.0, min_snr=p_snr, geom_cache=geom_cache)
                
                if sem:
                    la, lo, al = ecef_a_geodesicas(sem[0], sem[1], sem[2])
                    nt, et = geodesicas_a_utm(la, lo, utm_h)
                    coords_raw.append((nt, et, al, status))
            
            if os.path.exists(flag_file): yield "\n[!] Operación interrumpida prematuramente por el operador.\n"
            if not coords_raw: yield "> [ERROR] Nube de puntos bruta colapsada en Pre-Scan.\n"; return
            
            deltas_h = sorted([math.hypot(c[0] - utm_n_r, c[1] - utm_e_r) for c in coords_raw])
            deltas_v = sorted([abs(c[2] - utm_c_r) for c in coords_raw])
            idx_optimo = max(1, len(deltas_h) // 3)
            best_eh = max(0.01, float(deltas_h[idx_optimo]) * 1.5)
            best_ev = max(0.01, float(deltas_v[idx_optimo]) * 1.5)
            
            yield f"  [*] Límite Horizontal Inyectado: {f_14(best_eh)} m\n"
            yield f"  [*] Límite Vertical Inyectado: {f_14(best_ev)} m\n\n"
            
            yield f"[PROGRESO] Fase 2: Malla Tridimensional Acelerada Libre (Caché de Coordenadas Activa)...\n"
            global_best_score = float('inf')
            best_rmse = float('inf')
            best_params = {}
            
            # [MODIFICACIÓN QUIRÚRGICA: Centro Máscara en 12.0° y Piso en 10.0°]
            m_center, m_span = 12.0, 5.0  
            cp_center, cp_span = 2.0, 1.5 
            ca_center, ca_span = 2.0, 1.5 
            
            def get_local_median(lst):
                s = sorted(lst); n = len(s)
                if n == 0: return 0.0
                return s[n//2] if n % 2 == 1 else (s[n//2 - 1] + s[n//2]) / 2.0
            
            time_out = False
            for nivel in range(p_iter):
                if time_out or os.path.exists(flag_file): break
                yield f"  [+] Refinando espacio de búsqueda libre (Zoom {nivel+1}/{p_iter})...\n"
                
                m_grid = [max(10.0, x) for x in [m_center - m_span, m_center, m_center + m_span]]
                cp_grid = [max(2.0, x) for x in [cp_center - cp_span, cp_center, cp_center + cp_span]]
                ca_grid = [max(2.0, x) for x in [ca_center - ca_span, ca_center, ca_center + ca_span]]
                
                nivel_best_rmse = float('inf')
                nivel_best_params = {}
                
                for m in set(m_grid):
                    if time_out or os.path.exists(flag_file): break
                    
                    coords = []
                    for t in t_sample:
                        if time.time() - start_time > 28.0:
                            time_out = True
                            break
                        if os.path.exists(flag_file): break
                        
                        if modo_str == "MODO_D_DGPS": sem, status, _ = calcular_IRLS_MODO_D(sd_suavizada[t], nav, sp3, X_b, Y_b, Z_b, t, m, min_snr=p_snr, geom_cache=geom_cache)
                        else: sem, status, _ = calcular_IRLS_MODO_B(sd_suavizada[t], nav, sp3, X_b, Y_b, Z_b, t, m, min_snr=p_snr, geom_cache=geom_cache)
                        
                        if sem:
                            la, lo, al = ecef_a_geodesicas(sem[0], sem[1], sem[2])
                            nt, et = geodesicas_a_utm(la, lo, utm_h)
                            coords.append((nt, et, al, status))
                    
                    if not coords: continue
                    
                    m_N = get_local_median([c[0] for c in coords])
                    m_E = get_local_median([c[1] for c in coords])
                    m_Z = get_local_median([c[2] for c in coords])
                    med_estaticas = (m_N, m_E, m_Z)
                    
                    for cp in set(cp_grid):
                        for ca in set(ca_grid):
                            res = estadistica_desacoplada(coords, cp, ca, best_eh, best_ev, med_estaticas)
                            if res[0] is None: continue
                            nf, ef, zf, std_n, std_e, std_z, ret, fix_ratio = res
                            rmse_3d = math.sqrt((nf - utm_n_r)**2 + (ef - utm_e_r)**2 + ((zf - h_r) - utm_c_r)**2)
                            
                            if rmse_3d < nivel_best_rmse:
                                nivel_best_rmse = rmse_3d
                                nivel_best_params = {'m': float(m), 'cp': float(cp), 'ca': float(ca), 'rmse': float(rmse_3d)}
                                if rmse_3d < global_best_score:
                                    global_best_score = rmse_3d
                                    best_rmse = rmse_3d
                                    best_params = {'mask': float(m), 'cp': float(cp), 'ca': float(ca), 'eh': float(best_eh), 'ev': float(best_ev), 'max_gap': float(p_max_gap), 'snr': float(p_snr), 'rmse': float(rmse_3d), 'ret': int(ret)}
                
                if nivel_best_rmse != float('inf') and not time_out:
                    yield f"  [*] Fin Iteración {nivel+1} | Mejor RMSE Local: {f_14(nivel_best_params['rmse'])} m\n"
                    
                if global_best_score != float('inf'):
                    m_center, m_span = float(best_params['mask']), m_span / 2.0
                    cp_center, cp_span = float(best_params['cp']), cp_span / 2.0
                    ca_center, ca_span = float(best_params['ca']), ca_span / 2.0
                else:
                    m_span /= 2.0; cp_span /= 2.0; ca_span /= 2.0

            if os.path.exists(flag_file) or time_out:
                yield "\n> [SISTEMA] SE FORZÓ LA DETENCIÓN DEL BUCLE. PROCEDIENDO A EXTRAER EL MEJOR DATO RECOPILADO...\n"
                if global_best_score == float('inf'):
                     yield "> [ERROR] Timeout: El modelo no tuvo tiempo de converger ni 1 iteración. Reduzca la data base.\n"
                     return

            if best_rmse != float('inf'):
                guardar_estado(uid, 'opt_mask', float(best_params['mask']))
                guardar_estado(uid, 'opt_cp', float(best_params['cp']))
                guardar_estado(uid, 'opt_ca', float(best_params['ca']))
                guardar_estado(uid, 'opt_max_gap', float(best_params.get('max_gap', p_max_gap)))
                guardar_estado(uid, 'opt_snr', float(best_params.get('snr', p_snr)))
                guardar_estado(uid, 'opt_eh', float(best_params['eh']))
                guardar_estado(uid, 'opt_ev', float(best_params['ev']))
                
                guardar_estado(uid, 'estrategia_activa', modo_str)
                
                fecha_calculo = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                t_exec_script = time.time() - start_time

                yield "\n========================================================\n"
                yield f"      [INFORME] PARÁMETROS ÓPTIMOS LIBRES ({modo_str})\n"
                yield "========================================================\n"
                yield f"  [-] Punto Base (Pivote): {leer_estado(uid, 'nombre_base')}\n"
                yield f"  [-] Punto Rover (Calib): {leer_estado(uid, 'nombre_rover')}\n"
                yield f"  [-] Fecha y Hora de Cálculo: {fecha_calculo}\n"
                yield f"  [-] Tiempo de Ejecución Script: {f_14(t_exec_script)} segundos\n"
                yield f"  [-] Coord. Base Fija (N,E,Z): {f_14(utm_n)}, {f_14(utm_e)}, {f_14(utm_c)}\n"
                yield f"  [-] Coord. Rover Cal (N,E,Z): {f_14(utm_n_r)}, {f_14(utm_e_r)}, {f_14(utm_c_r)}\n"
                yield "--------------------------------------------------------\n"
                yield f"  [-] Tolerancia Sync Dinámica (max_gap): {f_14(best_params.get('max_gap', p_max_gap))}\n"
                yield f"  [-] Máscara Elevación (°): {f_14(best_params['mask'])}\n"
                yield f"  [-] Filtro Sigma Plan (cp): {f_14(best_params['cp'])}\n"
                yield f"  [-] Filtro Sigma Alt (ca): {f_14(best_params['ca'])}\n"
                yield f"  [-] Error Permitido Horizontal (m): {f_14(best_params['eh'])}\n"
                yield f"  [-] Error Permitido Vertical (m): {f_14(best_params['ev'])}\n"
                yield "--------------------------------------------------------\n"
                yield f"  [*] Menor Distancia 3D al Punto: {f_14(best_params['rmse'])} m\n"
                yield f"  [*] Épocas Retenidas: {best_params['ret']}\n"
                yield "========================================================\n"
                yield "\n[SUCCESS]"
            else:
                yield "\n> [ERROR] El modelo no convergió. Filtros demasiado agresivos o Timeout temprano.\n"
        except Exception as e: yield f"\n> [ERROR FATAL] {str(e)}"
    return Response(procesar(), mimetype='text/plain', headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'})

@app.route('/API/tab4_procesar', methods=['POST'])
def tab4_procesar():
    start_time = time.time()
    
    uid = request.form.get('uid', request.remote_addr)
    ws = get_workspace(uid)
    
    utm_n = safe_f(leer_estado(uid, 'utm_norte'), 0.0)
    utm_e = safe_f(leer_estado(uid, 'utm_este'), 0.0)
    utm_c = safe_f(leer_estado(uid, 'utm_cota'), 0.0)
    utm_h = safe_i(leer_estado(uid, 'utm_huso'), 19)
    utm_hem = leer_estado(uid, 'utm_hemisferio') or 'N'
    h_b = safe_f(leer_estado(uid, 'altura_base'), 0.0)
    modo_hardware = leer_estado(uid, 'modo_hardware') or 'iguales'
    nombre_base = leer_estado(uid, 'nombre_base') or 'BASE_DESCONOCIDA'
    
    p_mask = safe_f(leer_estado(uid, 'opt_mask'), 3.5)
    p_cp = safe_f(leer_estado(uid, 'opt_cp'), 2.0)
    p_ca = safe_f(leer_estado(uid, 'opt_ca'), 2.0)
    err_hor_max = safe_f(leer_estado(uid, 'opt_eh'), 0.0)
    err_ver_max = safe_f(leer_estado(uid, 'opt_ev'), 0.0)
    p_max_gap = safe_f(leer_estado(uid, 'opt_max_gap'), 2.0)
    estrategia = leer_estado(uid, 'estrategia_activa') or "MODO_D_DGPS"

    url_rover_nuevo = request.form.get('url_rover_nuevo')
    nombre_medido = request.form.get('nombre_medido', 'PUNTO_DESCONOCIDO')
    h_r_nuevo = safe_f(request.form.get('altura_rover_nuevo'), 0.0)
    
    if p_mask is None or utm_n is None:
        return Response("> [ERROR FATAL] Parámetros or coordenadas no encontrados. Ejecute la Pestaña 3 primero.\n", mimetype='text/plain')

    if not url_rover_nuevo or url_rover_nuevo.strip() == '': 
        return Response("> [ERROR] Falta el enlace de Drive del nuevo archivo RINEX Rover.\n", mimetype='text/plain')

    p_r_nuevo = os.path.join(ws, 'rover_nuevo_raw.obs')

    def procesar():
        try:
            yield "> [RED] Descargando Nuevo RINEX Rover desde Google Drive...\n"
            descargar_desde_gdrive(url_rover_nuevo, p_r_nuevo)
            rf_nuevo_filename = "Drive_Nuevo_Rover.obs"
            
            nav_path = leer_estado(uid, 'nav_path')
            sp3_path = leer_estado(uid, 'sp3_path')
            p_b_raw = leer_estado(uid, 'base_raw') 

            if not p_b_raw or not os.path.exists(p_b_raw): 
                yield "> [ERROR FATAL] Falta archivo RINEX Base original.\n"; return
                
            if not nav_path or not sp3_path:
                yield "\n> [ERROR CRÍTICO RECHAZADO]\n"
                yield "  [-] El cálculo geodésico estricto prohíbe el uso de broadcast nav para posicionamiento.\n"
                yield "  [-] FALTA ARCHIVO SP3 (Órbitas precisas) o NAV (Modelo Ionosférico).\n"
                yield "  [-] Vuelva a la Pestaña 2 y suba ambos productos obligatorios.\n"; return

            obs_b_raw = parse_rinex_obs_completo(p_b_raw)
            obs_r_raw = parse_rinex_obs_completo(p_r_nuevo) 
            nav = parse_rinex_nav_real(nav_path)
            sp3 = parse_sp3_preciso(sp3_path)
            
            modo_str, ratio, msg = analizar_calidad_y_senales_rinex(obs_b_raw, obs_r_raw, modo_hardware=modo_hardware, max_gap_tolerado=p_max_gap)
            yield f"> [ENRUTADOR] Análisis completado: {msg}\n"
            
            lat_b, lon_b, _ = utm_a_geodesicas(utm_e, utm_n, utm_h, utm_hem)
            X_b, Y_b, Z_b = geodesicas_a_ecef(lat_b, lon_b, utm_c + h_b)
            
            global_pdop = 99.9
            global_lambda = 0.0

            yield f"\n> [SISTEMA] Iniciando Procesamiento Definitivo Épocas Optimizadas | {modo_str} (IRLS + ENU | max_gap={p_max_gap}s)...\n"
            yield "[PROGRESO] Extrayendo Observables Diferenciales con interpolación temporal flexible...\n"
            
            rover_tows = sorted(list(obs_r_raw.keys()), key=lambda k: obs_r_raw[k].get('_meta', (0,0,0,0,0,0)))
            obs_b_sync = {}
            tiempos_base_preordenados = sorted(list(obs_b_raw.keys()), key=lambda k: obs_b_raw[k].get('_meta', (0,0,0,0,0,0)))
            
            for tr in rover_tows:
                if time.time() - start_time > 28.0:
                    yield "\n> [ALERTA] Freno de mano de 28.0s activado. Abortando interpolación para evitar timeout...\n"
                    break
                base_interp = interpolar_base_a_rover(obs_b_raw, tr, max_gap=p_max_gap, tiempos_base=tiempos_base_preordenados)
                if base_interp:
                    obs_b_sync[tr] = base_interp
                    obs_b_sync[tr]['_meta'] = obs_r_raw[tr]['_meta']

            if modo_str == "MODO_D_DGPS": sd_suavizada = aislar_diferencias_MODO_D(obs_b_sync, obs_r_raw)
            else: sd_suavizada = aislar_diferencias_MODO_B(obs_b_sync, obs_r_raw)
            
            if not sd_suavizada: yield "\n> [ERROR] No hay épocas sincronizadas válidas.\n"; return
            
            coords = []
            pdop_list = []
            t_eps = len(sd_suavizada); c = 0
            for t in sd_suavizada:
                c += 1
                
                if time.time() - start_time > 28.0:
                    yield "\n> [ALERTA] Freno de mano de 28.0s alcanzado. Generando informe con las épocas procesadas hasta el momento...\n"
                    break

                if c % max(1, t_eps // 10) == 0: yield f"[PROGRESO] Resolviendo Matrices IRLS DGPS (ENU)... {int((c / float(t_eps)) * 100.0)}%\n"
                
                snr_val = leer_estado(uid, 'opt_snr') or 25.0
                if modo_str == "MODO_D_DGPS": sem, status, pdop_val = calcular_IRLS_MODO_D(sd_suavizada[t], nav, sp3, X_b, Y_b, Z_b, t, p_mask, min_snr=snr_val, geom_cache=None)
                else: sem, status, pdop_val = calcular_IRLS_MODO_B(sd_suavizada[t], nav, sp3, X_b, Y_b, Z_b, t, p_mask, min_snr=snr_val, geom_cache=None)
                
                if sem:
                    la, lo, al = ecef_a_geodesicas(sem[0], sem[1], sem[2])
                    nt, et = geodesicas_a_utm(la, lo, utm_h)
                    coords.append((float(nt), float(et), float(al), str(status)))
                    if pdop_val: pdop_list.append(float(pdop_val))
                    
            if not coords: yield "\n> [ERROR] Fracaso algorítmico total en Inversión NxN.\n"; return
            global_pdop = sum(pdop_list) / float(max(1, len(pdop_list)))

            res_estadistica = estadistica_desacoplada(coords, p_cp, p_ca, err_hor_max, err_ver_max)
            if res_estadistica[0] is None:
                yield "\n> [ERROR] Operación Abortada: El 100% de las épocas superan el Error Máximo configurado.\n"; return
                
            nf, ef, zf, std_n, std_e, std_z, ret, fix_ratio = res_estadistica
            
            nf_final = float(nf)
            ef_final = float(ef)
            zf_final_ground = float(zf) - h_r_nuevo
            
            exec_time = time.time() - start_time
            
            p_dict = {
                'mask': float(p_mask), 'cp': float(p_cp), 'ca': float(p_ca),
                'max_gap': float(p_max_gap), 'snr': 0.0,
                'err_h': float(err_hor_max), 'err_v': float(err_ver_max),
                'nf': float(nf_final), 'ef': float(ef_final), 'zf': float(zf_final_ground), 
                'ret': int(ret), 'total': len(coords), 'std_n': float(std_n), 'std_e': float(std_e), 'std_z': float(std_z),
                'ez': float(std_z), 'fix_r': float(fix_ratio), 'pdop': float(global_pdop), 'lambda_ratio': float(global_lambda),
                'base_file': leer_estado(uid, 'name_base_raw') or "Drive_Base.obs",
                'rover_file': rf_nuevo_filename,
                'nombre_base': str(nombre_base),
                'nombre_medido': str(nombre_medido),
                'h_b': float(h_b),
                'h_r_nuevo': float(h_r_nuevo),
                'nav_file': leer_estado(uid, 'name_nav_file') or "auto_nav.nav",
                'sp3_file': leer_estado(uid, 'name_sp3_file'),
                'b_n': float(utm_n), 'b_e': float(utm_e), 'b_z': float(utm_c),
                'r_n_calc': float(nf_final), 'r_e_calc': float(ef_final), 'r_z_calc': float(zf_final_ground),
                'utm_h': int(utm_h), 'utm_hem': str(utm_hem),
                'estrategia': str(estrategia),
                'shift_applied': False,
                't_exec': float(exec_time)
            }
            
            yield "[PROGRESO] Procesamiento Geodésico Diferencial Puro Finalizado.\n"
            yield generar_informe_ascii("MEDICION", p_dict)
            yield "\n[SUCCESS]"
        except Exception as e: yield f"\n> [ERROR FATAL] {str(e)}"
    return Response(procesar(), mimetype='text/plain', headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)

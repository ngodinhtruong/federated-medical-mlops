/**
 * FL Monitor — app.js
 * Drawer history, inline client tasks, tab logs, chart
 */
const REFRESH = 5000;
let chart = null;
let _tasks = [];

const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

const fmt = (n, d=4) => (n == null || n === '-') ? '-' : Number(n).toFixed(d);

function ago(s) {
    if (!s) return 'N/A';
    const d = Math.floor((Date.now() - new Date(s.endsWith('Z') ? s : s+'Z')) / 1000);
    if (d < 60) return d + 's ago';
    if (d < 3600) return Math.floor(d/60) + 'm ago';
    if (d < 86400) return Math.floor(d/3600) + 'h ago';
    return Math.floor(d/86400) + 'd ago';
}

async function api(u) {
    try { const r = await fetch(u); return r.ok ? await r.json() : null; } catch { return null; }
}

/* ===== DRAWER ===== */
$('#histBtn').onclick = () => { $('#drawer').classList.add('on'); $('#drawerBackdrop').classList.add('on'); };
$('#drawerCloseBtn').onclick = closeDrawer;
$('#drawerBackdrop').onclick = closeDrawer;
function closeDrawer() { $('#drawer').classList.remove('on'); $('#drawerBackdrop').classList.remove('on'); }

/* ===== JSON MODAL ===== */
function showJson(title, obj) {
    $('#modalTitle').textContent = title;
    $('#modalPre').textContent = JSON.stringify(obj, null, 2);
    $('#modalBg').classList.add('on');
}
// Make showJson available on window for inline onclick
window.showJson = showJson;

$('#modalX').onclick = () => $('#modalBg').classList.remove('on');
$('#modalBg').onclick = e => { if (e.target === $('#modalBg')) $('#modalBg').classList.remove('on'); };

/* ===== OVERVIEW ===== */
async function updOverview() {
    const d = await api('/api/overview');
    if (!d) { $('#sysDot').className = 'dot red'; $('#sysText').textContent = 'Offline'; return; }
    $('#sysDot').className = 'dot green pulse';
    $('#sysText').textContent = d.system_status || 'Online';
    $('#vClients').textContent = d.active_clients || 0;
    $('#vCycles').textContent = d.total_cycles || 0;
    if (d.champion) {
        $('#vChamp').textContent = d.champion.model_type || '?';
        $('#vF1').textContent = d.champion.metrics?.f1 != null ? fmt(d.champion.metrics.f1) : '-';
    } else { $('#vChamp').textContent = 'None'; $('#vF1').textContent = '-'; }
}

/* ===== RUNNING TASKS ===== */
async function updTasks() {
    const d = await api('/api/running-tasks');
    _tasks = d?.tasks || [];
}

/* ===== CLIENTS ===== */
async function updClients() {
    const d = await api('/api/clients');
    if (!d?.clients?.length) { $('#clientsList').innerHTML = '<p class="muted center">No clients</p>'; return; }
    const now = Date.now();
    let h = '';
    for (const c of d.clients) {
        const ts = c.timestamp ? new Date(c.timestamp.endsWith('Z') ? c.timestamp : c.timestamp+'Z').getTime() : 0;
        const on = (now - ts) < 600000;
        const stag = on ? '<span class="tag tag-on">Online</span>' : '<span class="tag tag-off">Offline</span>';
        let dq = '';
        if (c.data_quality) dq = c.data_quality.drift_alert ? '<span class="tag tag-drift">Drift</span>' : '<span class="tag tag-ok">OK</span>';

        // inline tasks
        const myTasks = _tasks.filter(t => t.client_id === c.client_id);
        let taskH = '';
        for (const t of myTasks) {
            const lbl = t.task.replace(/_/g,' ').replace(/\b\w/g, x=>x.toUpperCase());
            taskH += `<div class="cl-task"><span class="dot green pulse" style="width:5px;height:5px"></span>${lbl}</div> `;
        }

        // safe JSON for data attribute
        const safeJ = JSON.stringify(c).replace(/&/g,'&amp;').replace(/'/g,'&#39;').replace(/"/g,'&quot;');

        h += `<div class="cl">
            <div class="cl-top">
                <span class="cl-name">Client ${c.client_id}</span>
                <div class="cl-badges">${dq}${stag}<button class="jbtn" onclick="showJson('Client ${c.client_id}',JSON.parse(this.dataset.j))" data-j="${safeJ}">JSON</button></div>
            </div>
            <div class="cl-meta">Cycle ${c.cycle_id||'-'} · Data ${c.has_data?'✓':'✗'} · Train ${c.can_train?'✓':'✗'} · ${ago(c.timestamp)}</div>
            ${taskH ? '<div>'+taskH+'</div>' : ''}
        </div>`;
    }
    $('#clientsList').innerHTML = h;
}

/* ===== CHAMPION ===== */
async function updChamp() {
    const d = await api('/api/champion');
    if (!d?.champion) { $('#champBox').innerHTML = '<p class="muted center">No champion yet</p>'; return; }
    const c = d.champion, m = c.metrics || {};
    $('#champBox').innerHTML = `
        <div class="ch-name">${c.model_type||'?'}</div>
        <div class="ch-grid">
            <div class="ch-item"><div class="ch-lbl">F1 Score</div><div class="ch-v">${fmt(m.f1)}</div></div>
            <div class="ch-item"><div class="ch-lbl">Loss</div><div class="ch-v">${fmt(m.loss)}</div></div>
            <div class="ch-item"><div class="ch-lbl">Accuracy</div><div class="ch-v">${fmt(m.accuracy)}</div></div>
            <div class="ch-item"><div class="ch-lbl">Precision</div><div class="ch-v">${fmt(m.precision)}</div></div>
        </div>
        <div class="ch-meta">Cycle ${c.cycle_id||'-'} · ${c.chosen||'-'} · ${ago(c.updated_at)}</div>
        <button class="jbtn" style="margin-top:6px" onclick="fetch('/api/json?path=registry/champion.json').then(r=>r.json()).then(d=>showJson('Champion',d.data||d))">View JSON</button>`;
}

/* ===== HISTORY ===== */
async function updHist() {
    const d = await api('/api/challenges');
    if (!d?.challenges?.length) { $('#histBody').innerHTML = '<tr><td colspan="7" class="muted center">No history</td></tr>'; return; }
    let h = '';
    for (const c of d.challenges) {
        const m = c.challenger_metrics || {};
        const safeJ = JSON.stringify(c).replace(/&/g,'&amp;').replace(/'/g,'&#39;').replace(/"/g,'&quot;');
        h += `<tr>
            <td><b>${c.challenger_model_type||'-'}</b></td>
            <td>${(c.challenger_cycle_id||'-').substring(0,16)}</td>
            <td>${fmt(m.f1)}</td>
            <td>${fmt(m.loss)}</td>
            <td class="${c.promoted?'p-yes':'p-no'}">${c.promoted?'🏆':'—'}</td>
            <td>${ago(c.timestamp)}</td>
            <td><button class="jbtn" onclick="showJson('Challenge',JSON.parse(this.dataset.j))" data-j="${safeJ}">JSON</button></td>
        </tr>`;
    }
    $('#histBody').innerHTML = h;
}

/* ===== SERVER LOGS ===== */
const LMAP = {
    'fl-server-mlp':    { el:'#tMlp',    dot:'#dMlp' },
    'fl-server-cnn':    { el:'#tCnn',    dot:'#dCnn' },
    'fl-server-logreg': { el:'#tLogreg', dot:'#dLogreg' },
};

async function updLogs() {
    const d = await api('/api/server-logs?tail=80');
    if (!d?.servers) return;
    for (const [n, info] of Object.entries(d.servers)) {
        const m = LMAP[n]; if (!m) continue;
        $(m.el).textContent = info.logs || '(no output)';
        $(m.dot).className = info.status === 'running' ? 'dot green pulse' : 'dot red';
        $(m.el).scrollTop = $(m.el).scrollHeight;
    }
}

/* ===== CHART ===== */
async function updChart() {
    const d = await api('/api/challenges');
    if (!d?.challenges) return;
    const g = {};
    for (const c of d.challenges) {
        const mt = c.challenger_model_type || '?';
        (g[mt] = g[mt] || []).push({ cy: c.challenger_cycle_id||'', f1: c.challenger_metrics?.f1||0 });
    }
    for (const k in g) g[k].sort((a,b) => a.cy.localeCompare(b.cy));

    const cols = { MLP:['rgba(99,102,241,.2)','#6366f1'], CNN:['rgba(16,185,129,.2)','#10b981'], LogisticRegression:['rgba(245,158,11,.2)','#f59e0b'] };
    const ds = Object.entries(g).map(([mt,pts]) => ({
        label:mt, data:pts.map(p=>p.f1),
        borderColor:(cols[mt]||['rgba(148,163,184,.2)','#94a3b8'])[1],
        backgroundColor:(cols[mt]||['rgba(148,163,184,.2)','#94a3b8'])[0],
        tension:.3, fill:true, pointRadius:3, pointHoverRadius:5,
    }));
    const mx = Math.max(...Object.values(g).map(a=>a.length),0);
    const lb = Array.from({length:mx},(_,i)=>'#'+(i+1));

    if (chart) { chart.data.labels=lb; chart.data.datasets=ds; chart.update('none'); }
    else {
        chart = new Chart($('#chart').getContext('2d'), {
            type:'line', data:{labels:lb,datasets:ds},
            options:{responsive:true,maintainAspectRatio:false,
                interaction:{intersect:false,mode:'index'},
                plugins:{legend:{labels:{color:'#94a3b8',font:{family:'Inter',size:10}}}},
                scales:{
                    x:{grid:{color:'rgba(99,102,241,.06)'},ticks:{color:'#64748b',font:{size:9}}},
                    y:{min:0,max:1,grid:{color:'rgba(99,102,241,.06)'},ticks:{color:'#64748b',font:{size:9}},title:{display:true,text:'F1',color:'#64748b'}},
                },
            },
        });
    }
}

/* ===== CLOCK & MAIN LOOP ===== */
function tick() { $('#clock').textContent = new Date().toLocaleTimeString('vi-VN',{hour12:false}); }

async function refresh() {
    await Promise.all([updOverview(), updTasks()]);
    await Promise.all([updClients(), updChamp(), updHist(), updLogs(), updChart()]);
}

tick(); setInterval(tick,1000);
refresh(); setInterval(refresh, REFRESH);

PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>River | Private table</title>
<link rel="icon" href="data:,">
<style>
:root{color-scheme:dark;font:16px/1.45 "Helvetica Neue",Helvetica,sans-serif;background:#171918;color:#eeeae1;--muted:#a9aea7;--line:#383c38;--accent:#d2b780;--felt:#203e34;--ink:#202724;--red:#a43d38}
*{box-sizing:border-box}body{margin:0}main{max-width:820px;margin:auto;padding:0 36px 28px}[hidden]{display:none!important}
header{height:84px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);gap:16px}h1{font:500 27px/1 Georgia,serif;letter-spacing:-1px;margin:0}.wordmark{display:flex;align-items:baseline;gap:12px}.wordmark span{font-size:10px;letter-spacing:.2em;color:var(--muted)}.mode{font-size:12px;color:var(--muted);white-space:nowrap}p{margin:0}.muted{color:var(--muted)}
.table-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:23px 0 18px}h2{font-size:14px;font-weight:500;margin:0}.table-rules{font-size:12px;color:var(--muted)}
.table{position:relative;isolation:isolate;height:400px;display:flex;flex-direction:column;align-items:center;justify-content:space-between;padding:12px 24px}
.table::before{content:"";position:absolute;z-index:-1;inset:34px 0;border:1px solid #536157;border-radius:180px;background:var(--felt);box-shadow:0 0 0 7px #252c28}
.seat{display:flex;align-items:center;justify-content:center;gap:20px;background:#212622;border:1px solid #47534b;border-radius:4px;padding:9px 17px;min-width:190px;box-shadow:0 3px 7px #0002}.seat-name{display:flex;align-items:center;gap:7px;font-size:13px}.seat-name span{color:#c2c6bb;font-size:11px}.seat-name .dealer{color:var(--accent)}.stack{font:500 19px/1.2 "Helvetica Neue",Helvetica,sans-serif;font-variant-numeric:tabular-nums}.seat-group{display:flex;flex-direction:column;align-items:center;gap:9px}.cards{display:flex;justify-content:center;gap:6px}.card{position:relative;display:block;flex:none;width:54px;height:76px;border:1px solid #dedbd1;border-radius:4px;background:#f4f0e6;color:var(--ink);box-shadow:0 2px 3px #0002;font-family:Georgia,serif}.card.red{color:var(--red)}.card-corner{position:absolute;top:5px;left:6px;font:bold 19px/1 Georgia,serif}.card-suit{position:absolute;right:7px;bottom:5px;font-size:30px;line-height:1}.card.back{border:3px solid #c3c9bb;background:repeating-linear-gradient(45deg,#31483f 0,#31483f 3px,#526258 3px,#526258 4px);box-shadow:inset 0 0 0 1px #1d332b,0 2px 3px #0002}.card.empty{background:#18342b;border:1px solid #64766b77;box-shadow:none}.opponent-cards .card{width:33px;height:44px;border-width:2px}.board{display:flex;flex-direction:column;align-items:center;gap:15px;padding:12px 0}.pot{font:500 23px/1 "Helvetica Neue",Helvetica,sans-serif;font-variant-numeric:tabular-nums}.pot small{font-size:11px;font-weight:400;letter-spacing:.1em;text-transform:uppercase;color:#bbc9bc;margin-right:8px}.street{font-size:10px;letter-spacing:.17em;text-transform:uppercase;color:#bbc9bc;min-height:14px}.hero-cards .card{width:48px;height:67px}
.control-area{max-width:520px;margin:25px auto 0}.status-row{display:flex;align-items:center;justify-content:space-between;gap:14px;min-height:44px;margin-bottom:8px}.status{font-size:14px;color:#c8cec4}.status.error{color:#f0b19a}.refresh{padding:0 10px;min-height:36px;font-size:12px;background:transparent;border-color:transparent;color:var(--muted);flex:none}.result{display:flex;align-items:baseline;justify-content:space-between;padding:4px 0 14px;gap:12px}.result strong{font-size:22px;font-weight:500;white-space:nowrap}.result span{font-size:12px;color:var(--muted)}
button,input{font:inherit}button{min-height:46px;border:1px solid #52584f;border-radius:4px;padding:10px 16px;background:#292e29;color:#eeeae1;cursor:pointer;font-weight:500}button:hover:not(:disabled){background:#343b33;border-color:#828b7b}button.primary{background:var(--accent);color:#25231d;border-color:var(--accent)}button.primary:hover:not(:disabled){background:#e4c996;border-color:#e4c996}button:disabled{opacity:.35;cursor:not-allowed}button:focus-visible,input:focus-visible,summary:focus-visible{outline:2px solid #ead09e;outline-offset:4px}.actions{display:grid;grid-template-columns:1fr 1fr;gap:9px}.raise-row{display:grid;grid-template-columns:1fr 1fr;gap:9px;align-items:end;margin-top:13px}.raise-label{display:flex;justify-content:space-between;gap:8px;margin:0 0 5px;font-size:12px;color:var(--muted)}input[type=number]{background:#1c201c;color:#eeeae1;border:1px solid #52584f;border-radius:4px;padding:10px 12px;width:100%;height:46px;font-variant-numeric:tabular-nums}.next{width:100%}.retry{width:100%;margin:5px 0 12px;border-color:#b99270;color:#f0c1a2}.connect{padding:65px 0 30px;max-width:430px;margin:auto}.connect h2{font:26px Georgia,serif;margin-bottom:10px}.connect p{color:var(--muted);margin-bottom:26px;font-size:15px}.connect label{font-size:13px}input[type=file]{display:block;max-width:100%;margin-top:10px;font-size:13px;color:var(--muted)}input::file-selector-button{background:#d2b780;color:#25231d;border:0;border-radius:3px;padding:10px 14px;margin-right:10px;cursor:pointer}
.inspector{border-top:1px solid var(--line);margin-top:28px;color:var(--muted);font-size:12px}summary{cursor:pointer;min-height:44px;align-content:center;list-style:none}summary::-webkit-details-marker{display:none}summary::after{content:"+";float:right;margin-left:12px}details[open]>summary::after{content:"−"}.inspector>summary{font-size:12px;letter-spacing:.04em}.inspector-body{padding:4px 0 18px}.inspector details{border-bottom:1px solid #30362f}.inspector details:last-child{border:0}dl{display:grid;grid-template-columns:115px 1fr;gap:6px 12px;margin:10px 0 20px}dd{margin:0;overflow-wrap:anywhere;color:#d1d6cb}pre{font:12px/1.5 "SF Mono",Menlo,monospace;white-space:pre-wrap;overflow-wrap:anywhere;margin:0 0 15px;background:#111611;padding:14px;border-radius:3px;color:#cbd3c5}.receipt{overflow-wrap:anywhere;font:11px/1.6 Menlo,monospace;margin:12px 0}.receipt:empty{display:none}footer{display:flex;justify-content:space-between;gap:16px;margin-top:14px;font-size:11px;color:var(--muted)}
@media(min-width:700px){main{padding-top:16px}.table{margin:8px 12px}.table::before{inset:34px -6px}.table-heading{padding-left:12px;padding-right:12px}}
@media(max-width:520px){main{padding:0 17px 24px}header{height:67px}.wordmark{gap:8px}h1{font-size:25px}.wordmark span{font-size:9px;letter-spacing:.12em}.mode{font-size:11px}.table-heading{padding:20px 0 16px}.table-rules{font-size:11px}.table{height:346px;padding:4px 8px}.table::before{inset:28px 3px;border-radius:130px;box-shadow:0 0 0 5px #252c28}.seat{padding:8px 14px;min-width:174px;gap:18px}.stack{font-size:18px}.seat-group{gap:7px}.board{gap:11px;padding:7px 0}.card{width:46px;height:65px}.card-corner{font-size:17px}.card-suit{font-size:27px}.hero-cards .card{width:44px;height:61px}.opponent-cards .card{width:29px;height:38px}.control-area{margin-top:18px}.inspector{margin-top:22px}}
@media(prefers-reduced-motion:no-preference){button{transition:background .12s,border-color .12s}}
</style>
</head>
<body><main>
<header><div class="wordmark"><h1>Heads Up</h1><span>POKER LAB</span></div><span class="mode">Local · play money</span></header>
<section id="connect" class="connect" aria-label="Connect to table">
<h2>Take your seat.</h2><p>Open the seat file from your local table to join.</p>
<label for="credentials">Your seat credential file</label><input id="credentials" type="file" accept=".json,application/json">
</section>
<section id="connected" aria-label="Poker table" hidden>
<div class="table-heading"><h2 id="hand">Table</h2><span id="table-rules" class="table-rules"></span></div>
<div class="table">
<div class="seat-group"><div class="seat"><div class="seat-name"><b id="opponent-label">Opponent</b><span id="opponent-position"></span></div><strong id="opponent-stack" class="stack" aria-label="Opponent stack">—</strong></div><div id="opponent-cards" class="cards opponent-cards" aria-label="Opponent cards"><span class="card back" aria-label="Hidden card"></span><span class="card back" aria-label="Hidden card"></span></div></div>
<div class="board"><div class="pot"><small>Pot</small><span id="pot">—</span></div><div id="board" class="cards" aria-label="Community cards"></div><div id="street" class="street"></div></div>
<div class="seat-group"><div id="hole" class="cards hero-cards" aria-label="Your cards"></div><div class="seat"><div class="seat-name"><b id="hero-label">You</b><span id="hero-position"></span></div><strong id="hero-stack" class="stack" aria-label="Your stack">—</strong></div></div>
</div>
</section>
<div class="control-area">
<div class="status-row"><p id="status" class="status" role="status">Choose your seat file to connect.</p><button id="refresh" class="refresh" disabled>Refresh table</button></div>
<button id="retry" class="retry" hidden>Retry pending action</button>
<div id="result" class="result" hidden></div>
<button id="ready" class="primary next" disabled hidden>Ready</button>
<div id="betting-controls" hidden>
<div class="actions"><button id="fold" disabled>Fold</button><button id="check" disabled>Check</button><button id="call" disabled>Call</button></div>
<div class="raise-row"><div><div class="raise-label"><label for="raise-to">Raise to</label><span id="raise-bounds"></span></div><input id="raise-to" type="number" step="1" min="0" inputmode="numeric" aria-describedby="raise-help" disabled></div><button id="raise" class="primary" disabled>Raise</button></div>
</div>
</div>
<details id="inspector" class="inspector" hidden><summary>Inspector</summary><div class="inspector-body">
<dl id="metadata"></dl><p id="raise-help">Raise to is the total bet on this street, in chips. Stacks reset each hand.</p><p id="receipt" class="receipt" aria-live="polite"></p>
<details><summary>Policy observation</summary><pre id="observation">null</pre></details>
<details><summary>Table metadata and rules</summary><pre id="state-metadata">null</pre></details>
<details><summary>Public hand history</summary><pre id="history">[]</pre></details>
<details><summary>Completed-hand result</summary><pre id="settlement">null</pre></details>
</div></details>
<footer><span>Heads-up no-limit hold’em</span><span>No cash value</span></footer>
</main>
<script>
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const identifier = value => typeof value === 'string' && /^[0-9a-f]{32}$/.test(value);
  const integer = value => Number.isSafeInteger(value) && value >= 0;
  const seat = value => value === 0 || value === 1;
  const pair = value => Array.isArray(value) && value.length === 2 && value.every(integer);
  const card = value => typeof value === 'string' && /^[2-9TJQKA][cdhs]$/.test(value);
  const fields = ['schema_version','table_id','player_id','url','token'];
  let credentials = null, state = null, pending = null, accepted = null;
  let busy = false, refreshing = false, paused = false, minimumRevision = 0, lastRaiseRevision = null;
  let journalKey = null;

  function message(text, error = false) {
    $('status').textContent = text;
    $('status').className = `status${error ? ' error' : ''}`;
  }

  function writeJournal(nextPending, nextAccepted = accepted) {
    sessionStorage.setItem(journalKey, JSON.stringify({pending:nextPending, accepted:nextAccepted}));
  }

  function validCommand(command) {
    if (!command || command.schema_version !== 1 || command.table_id !== credentials.table_id ||
        !identifier(command.request_id) || !integer(command.hand_index) || !integer(command.revision)) return false;
    if (command.kind === 'ready') return command.action === null;
    return command.kind === 'act' && command.action &&
      ['fold','check','call','raise_to'].includes(command.action.kind) && integer(command.action.amount);
  }

  function validReceipt(receipt, command) {
    return receipt && receipt.schema_version === 1 && receipt.table_id === credentials.table_id &&
      receipt.player_id === credentials.player_id && identifier(receipt.request_id) &&
      integer(receipt.revision) && (!command ||
      (receipt.request_id === command.request_id && receipt.revision === command.revision + 1));
  }

  function validState(value) {
    if (!value || value.schema_version !== 1 || value.play_money !== true ||
        value.table_id !== credentials.table_id || value.player_id !== credentials.player_id ||
        value.opponent_id !== 1 - value.player_id || !integer(value.revision) ||
        !integer(value.hand_index) || !integer(value.hands_total) || value.hand_index >= value.hands_total ||
        value.hero_seat !== (value.player_id ^ (value.hand_index % 2)) ||
        !['waiting','playing','hand_complete','complete'].includes(value.phase) ||
        !Array.isArray(value.ready) || value.ready.length !== 2 || value.ready.some(v => typeof v !== 'boolean') ||
        !Array.isArray(value.history) || value.history.length > 4096 || !value.config) return false;
    const config = value.config;
    if (!['small_blind','big_blind','stack_bb','rake_bps','rake_cap'].every(key => integer(config[key])) ||
        config.small_blind < 1 || config.big_blind < config.small_blind || config.stack_bb < 1 ||
        typeof config.no_flop_no_drop !== 'boolean') return false;
    const obs = value.observation;
    if (obs !== null && (!obs || value.phase !== 'playing' || obs.actor !== value.hero_seat || obs.button !== 1 ||
        !Array.isArray(obs.hole) || obs.hole.length !== 2 || !obs.hole.every(card) ||
        !Array.isArray(obs.board) || !obs.board.every(card) ||
        ({preflop:0,flop:3,turn:4,river:5})[obs.street] !== obs.board.length ||
        !pair(obs.stacks) || !pair(obs.bets) || !integer(obs.pot) || !integer(obs.to_call) ||
        typeof obs.can_fold !== 'boolean' || !Array.isArray(obs.history) ||
        JSON.stringify(obs.history) !== JSON.stringify(value.history) ||
        (obs.min_raise_to !== null && !integer(obs.min_raise_to)) ||
        (obs.max_raise_to !== null && !integer(obs.max_raise_to)) ||
        (obs.min_raise_to === null) !== (obs.max_raise_to === null))) return false;
    return true;
  }

  async function request(path, body) {
    const response = await fetch(path, {
      method:body ? 'POST' : 'GET', cache:'no-store', credentials:'omit', redirect:'error',
      headers:{Authorization:`Bearer ${credentials.token}`, ...(body ? {'Content-Type':'application/json'} : {})},
      body:body ? JSON.stringify(body) : undefined,
      signal:AbortSignal.timeout(5000),
    });
    const text = await response.text();
    if (text.length > 1048576) throw new Error('The table response exceeded the size limit.');
    let data;
    try { data = JSON.parse(text); } catch { throw new Error('The table returned unreadable data.'); }
    return {response, data};
  }

  function acceptState(next) {
    if (!validState(next)) throw new Error('The table returned an invalid or mismatched state.');
    if (next.revision < minimumRevision || (state && next.revision < state.revision))
      throw new Error('The table returned an older state. Controls are paused.');
    if (state && next.revision === state.revision && JSON.stringify(next) !== JSON.stringify(state))
      throw new Error('The table changed without advancing its revision.');
    if (state && JSON.stringify(next.config) !== JSON.stringify(state.config))
      throw new Error('The table rules changed. Controls are paused.');
    state = next;
  }

  async function fetchState() {
    const {response, data} = await request('/v1/state');
    if (!response.ok) throw new Error(response.status === 401 ? 'The table rejected the seat file.' : `Could not read table state (HTTP ${response.status}).`);
    acceptState(data);
  }

  function appendCards(target, cards, placeholders, hidden = false) {
    target.replaceChildren();
    const suits = {c:'♣',d:'♦',h:'♥',s:'♠'};
    for (const code of [...cards, ...Array(Math.max(0, placeholders - cards.length)).fill(null)]) {
      const element = document.createElement('span');
      element.className = `card${code ? ('dh'.includes(code[1]) ? ' red' : '') : hidden ? ' back' : ' empty'}`;
      if (code) {
        const rank = document.createElement('span'), suit = document.createElement('span');
        rank.className = 'card-corner'; rank.textContent = code[0] === 'T' ? '10' : code[0];
        suit.className = 'card-suit'; suit.textContent = suits[code[1]];
        element.append(rank, suit);
      }
      element.setAttribute('role','img');
      element.setAttribute('aria-label', code ?? (hidden ? 'Hidden card' : 'Empty card slot'));
      target.append(element);
    }
  }

  function render() {
    $('connected').hidden = !state;
    $('connect').hidden = !!state;
    $('inspector').hidden = !state;
    $('betting-controls').hidden = state?.phase !== 'playing';
    $('credentials').disabled = busy || refreshing || credentials !== null;
    $('refresh').disabled = !credentials || busy || refreshing || !!pending;
    $('retry').hidden = !pending;
    $('retry').disabled = busy || refreshing;
    const blocked = busy || refreshing || paused || !!pending || !state;
    const obs = state?.observation;
    $('ready').disabled = blocked || !['waiting','hand_complete'].includes(state?.phase) || state.ready[state.player_id];
    $('ready').hidden = !state || !['waiting','hand_complete'].includes(state.phase);
    $('ready').textContent = state?.phase === 'hand_complete' ? (state.hand_index + 1 === state.hands_total ? 'Finish match' : 'Next hand') : 'Ready';
    $('fold').disabled = blocked || !obs?.can_fold;
    $('check').disabled = blocked || !obs || obs.to_call !== 0;
    $('call').disabled = blocked || !obs || obs.to_call === 0;
    $('check').hidden = !!obs?.to_call;
    $('call').hidden = !obs?.to_call;
    $('call').textContent = obs?.to_call ? `Call ${obs.to_call}` : 'Call';
    $('raise-to').disabled = blocked || !obs || obs.min_raise_to === null;
    $('raise').disabled = $('raise-to').disabled;
    if (accepted) $('receipt').textContent = `Accepted request ${accepted.request_id} · revision ${accepted.revision}`;
    if (!state) return;
    $('hand').textContent = `Hand ${String(state.hand_index + 1).padStart(2,'0')} / ${String(state.hands_total).padStart(2,'0')}`;
    $('table-rules').textContent = `${state.config.small_blind} / ${state.config.big_blind} blinds · ${state.config.stack_bb} BB`;
    const metadata = {
      'Table ID':state.table_id, 'Player ID':state.player_id, 'Engine seat':state.hero_seat,
      'Revision':state.revision, 'Blinds':`${state.config.small_blind} / ${state.config.big_blind}`,
      'Starting stack':`${state.config.stack_bb} BB`, 'Rake':`${state.config.rake_bps / 100}% · cap ${state.config.rake_cap} chips`,
    };
    $('metadata').replaceChildren();
    for (const [label, value] of Object.entries(metadata)) {
      const term = document.createElement('dt'), detail = document.createElement('dd');
      term.textContent = label; detail.textContent = String(value); $('metadata').append(term,detail);
    }
    $('hero-position').textContent = state.hero_seat ? 'D / SB' : 'BB';
    $('hero-position').className = state.hero_seat ? 'dealer' : '';
    $('opponent-position').textContent = state.hero_seat ? 'BB' : 'D / SB';
    $('opponent-position').className = state.hero_seat ? '' : 'dealer';
    const board = obs?.board ?? state.result?.board ?? state.history.at(-1)?.board ?? [];
    appendCards($('board'), board, 5);
    appendCards($('hole'), obs?.hole ?? [], 2, state.phase === 'playing');
    $('pot').textContent = obs?.pot ?? state.result?.contested_pot ?? '—';
    $('street').textContent = obs?.street ?? (state.result ? 'Hand complete' : state.phase === 'waiting' ? 'Waiting to deal' : 'Opponent’s turn');
    const stacks = obs?.stacks ?? state.result?.net_stacks ?? (state.phase === 'waiting' ? Array(2).fill(state.config.stack_bb * state.config.big_blind) : null);
    $('hero-stack').textContent = stacks ? stacks[state.hero_seat] : '—';
    $('opponent-stack').textContent = stacks ? stacks[1-state.hero_seat] : '—';
    $('result').hidden = !state.result;
    if (state.result) {
      const amount = state.result.net_returns[state.hero_seat];
      const net = document.createElement('strong'), note = document.createElement('span');
      net.textContent = `${amount >= 0 ? '+' : '−'}${Math.abs(amount)} chips`;
      note.textContent = `This hand · rake ${state.result.rake}`;
      $('result').replaceChildren(net,note);
    }
    if (obs && obs.min_raise_to !== null) {
      $('raise-to').min = obs.min_raise_to; $('raise-to').max = obs.max_raise_to;
      if (lastRaiseRevision !== state.revision) $('raise-to').value = obs.min_raise_to;
      $('raise-bounds').textContent = `${obs.min_raise_to}–${obs.max_raise_to}`;
    } else {
      $('raise-to').value = ''; $('raise-bounds').textContent = 'No raise available';
    }
    lastRaiseRevision = state.revision;
    $('observation').textContent = JSON.stringify(obs, null, 2);
    const {observation,history,result,...publicMetadata} = state;
    $('state-metadata').textContent = JSON.stringify(publicMetadata,null,2);
    $('history').textContent = JSON.stringify(history,null,2);
    $('settlement').textContent = JSON.stringify(result,null,2);
  }

  function idleMessage() {
    if (pending) return 'Action unconfirmed. Retry safely resends it.';
    if (state.phase === 'complete') return 'Session complete.';
    if (state.phase === 'hand_complete') return state.ready[state.player_id] ? 'Waiting for your opponent.' : 'Hand complete.';
    if (state.phase === 'waiting') return state.ready[state.player_id] ? 'Waiting for your opponent.' : 'Your seat is ready.';
    return state.observation ? 'Your turn.' : 'Opponent’s turn.';
  }

  async function refresh(manual = false) {
    if (!credentials || busy || refreshing || pending || (paused && !manual)) return;
    const priorRevision = state?.revision;
    refreshing = true; render();
    try {
      await fetchState(); paused = false;
      if (manual || state.revision !== priorRevision) message(idleMessage());
    } catch (error) {
      paused = true; message(error.message || 'Connection interrupted. Press Refresh table to reconnect.', true);
    } finally { refreshing = false; render(); }
  }

  async function sendPending() {
    if (!credentials || !pending || busy || refreshing) return;
    busy = true; render();
    const command = pending;
    try {
      const {response, data} = await request('/v1/command', command);
      if (response.status === 409 && data.detail?.code === 'stale_state') {
        writeJournal(null); pending = null;
        await fetchState(); paused = false;
        message('That action was not accepted because the turn changed. Review the current state and choose again.',true);
      } else if (response.status === 422) {
        writeJournal(null); pending = null;
        await fetchState(); paused = false;
        message('The table rejected that action. Review the legal controls and choose again.',true);
      } else {
        if (!response.ok) throw new Error(`Action not confirmed (HTTP ${response.status}). Retry preserves the same request.`);
        if (!validReceipt(data,command)) throw new Error('The action receipt did not match the pending request. Controls are paused.');
        writeJournal(null,data); accepted = data; pending = null; minimumRevision = data.revision;
        await fetchState(); paused = false; message(idleMessage());
      }
    } catch (error) {
      paused = true; message(error.message || 'Action delivery is uncertain. Retry the saved action.',true);
    } finally { busy = false; render(); }
  }

  async function submit(action = null) {
    if (!state || busy || refreshing || paused || pending) return;
    const command = {schema_version:1, table_id:state.table_id, request_id:crypto.randomUUID().replaceAll('-',''),
      hand_index:state.hand_index, revision:state.revision, kind:action ? 'act' : 'ready', action};
    try { writeJournal(command); } catch { paused = true; message('This browser could not save the action journal. Nothing was sent.',true); render(); return; }
    pending = command; await sendPending();
  }

  $('credentials').addEventListener('change', async event => {
    const file = event.target.files[0];
    if (!file || credentials || busy) return;
    busy = true; render();
    try {
      if (file.size > 4096) throw new Error('Choose a valid seat credential JSON file.');
      const value = JSON.parse(await file.text());
      if (!value || typeof value !== 'object' || Object.keys(value).some(key => !fields.includes(key)) ||
          value.schema_version !== 1 || !identifier(value.table_id) || !seat(value.player_id) ||
          value.url !== location.origin || typeof value.token !== 'string' || !/^[A-Za-z0-9_-]{32,256}$/.test(value.token))
        throw new Error('The seat file must belong to this exact table origin.');
      credentials = value;
      journalKey = `headsup-browser:${credentials.table_id}:${credentials.player_id}`;
      const saved = JSON.parse(sessionStorage.getItem(journalKey) ?? '{"pending":null,"accepted":null}');
      if (!saved || typeof saved !== 'object' || !('pending' in saved) || !('accepted' in saved) ||
          (saved.pending !== null && !validCommand(saved.pending)) ||
          (saved.accepted !== null && !validReceipt(saved.accepted)))
        throw new Error('The saved action journal is invalid. Keep this tab for diagnosis.');
      pending = saved.pending; accepted = saved.accepted; minimumRevision = accepted?.revision ?? 0;
      await fetchState(); paused = false; message(idleMessage());
    } catch (error) {
      paused = true;
      message(error instanceof SyntaxError ? 'The file or saved journal is not valid JSON.' : error.message,true);
      if (!state) { credentials = null; pending = null; accepted = null; journalKey = null; event.target.value = ''; }
    } finally { busy = false; render(); }
  });
  $('refresh').addEventListener('click', () => refresh(true));
  $('retry').addEventListener('click', sendPending);
  function onAction(id, action) {
    $(id).addEventListener('click', event => {
      if (event.detail > 1) return;
      action();
    });
    $(id).addEventListener('keydown', event => {
      if (event.repeat && (event.key === 'Enter' || event.key === ' ')) event.preventDefault();
    });
  }
  onAction('ready', () => submit());
  for (const kind of ['fold','check','call']) onAction(kind, () => submit({kind,amount:0}));
  onAction('raise', () => {
    const input = $('raise-to'), amount = Number(input.value), obs = state?.observation;
    if (!obs || input.value.trim() === '' || !Number.isSafeInteger(amount) ||
        obs.min_raise_to === null || amount < obs.min_raise_to || amount > obs.max_raise_to) {
      message('Enter an integer raise-to total within the displayed legal range.',true); return;
    }
    submit({kind:'raise_to',amount});
  });
  setInterval(() => refresh(), 800);
  render();
})();
</script>
</body></html>'''

import {readPokerNow, normalizePokerNow} from './pokernow_state.mjs';
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

export function pokerNowMemoryIdentity(config) {
  if (config?.privatePlayMoney!==true||config.noRakeAntesStraddleVerified!==true) throw new Error('confirm_private_play_chip_table');
  if (!/^https:\/\/www\.pokernow\.com\/games\/[A-Za-z0-9_-]+$/.test(config.roomUrl)) throw new Error('invalid_memory_room');
  if (!Array.isArray(config.playerIds)||config.playerIds.length!==2||new Set(config.playerIds).size!==2||
      !config.playerIds.every(id=>typeof id==='string'&&/^[A-Za-z0-9_-]{1,128}$/.test(id))||
      !config.playerIds.includes(config.heroId)) throw new Error('pin_two_test_players');
  if (![config.smallBlind,config.bigBlind].every(value=>Number.isSafeInteger(value)&&value>0&&value<=1_000_000_000)||
      config.smallBlind>config.bigBlind) throw new Error('verify_table_rules');
  return {platform:'pokernow',room_url:config.roomUrl,hero_id:config.heroId,
    opponent_id:config.playerIds.find(id=>id!==config.heroId),small_blind:config.smallBlind,big_blind:config.bigBlind};
}

export async function startDecisionWorker({seed=20261106,config=null,memoryDirectory=null,memory='lifetime',populationPrior=null,policy='adaptive'}={}) {
  if (!Number.isSafeInteger(seed)||!['lifetime','recent'].includes(memory)||!['adaptive','river'].includes(policy)) throw new Error('invalid_worker_settings');
  if ([memoryDirectory,populationPrior].some(path=>path!==null&&(typeof path!=='string'||!path.trim()))) throw new Error('invalid_memory_path');
  if (config===null&&(memoryDirectory!==null||populationPrior!==null||memory!=='lifetime')) throw new Error('persistent_memory_requires_config');
  const root = fileURLToPath(new URL('../',import.meta.url));
  const args=['-m','headsup','decision-worker','--seed',String(seed),'--bot',policy];
  if (config!==null) {
    const identity=pokerNowMemoryIdentity(config);
    args.push('--memory-dir',resolve(memoryDirectory??`${root}state/opponents`),
      '--room-url',identity.room_url,'--hero-id',identity.hero_id,'--opponent-id',identity.opponent_id,
      '--small-blind',String(identity.small_blind),'--big-blind',String(identity.big_blind),'--memory',memory);
    if (populationPrior!==null) args.push('--population-prior',resolve(populationPrior));
  }
  const process = spawn(`${root}.venv/bin/python`,args,{cwd:root,stdio:['pipe','pipe','pipe']});
  const lines = createInterface({input:process.stdout});
  let pending = null, stopped = false;
  const stop = reason => {
    stopped = true;
    if (pending) {clearTimeout(pending.timeout);pending.reject(new Error(reason));pending=null;}
  };
  process.stderr.resume();
  process.on('error',()=>stop('worker_process_failed'));
  process.on('exit',()=>stop('worker_exited_wait_for_new_hand'));
  lines.on('line',line=>{
    if (!pending) {stop('unexpected_worker_output');process.kill();return;}
    const current = pending;
    pending = null; clearTimeout(current.timeout);
    try {current.resolve(JSON.parse(line));}
    catch {current.reject(new Error('invalid_worker_output'));stop('invalid_worker_output');process.kill();}
  });
  const request = (payload,{timeoutMs=3000}={}) => new Promise((resolve,reject)=>{
    if (stopped||pending) {reject(new Error(stopped?'worker_stopped':'concurrent_worker_request'));return;}
    const timeout = setTimeout(()=>{stop('worker_response_timeout');process.kill();},timeoutMs);
    pending = {resolve,reject,timeout};
    process.stdin.write(JSON.stringify(payload)+'\n',error=>{if(error){stop('worker_input_failed');process.kill();}});
  });
  const close = () => {stop('worker_closed');process.stdin.end();};
  try {
    const hello = await request({kind:'ping'},{timeoutMs:15000});
    if (hello.status!=='worker_ready') throw new Error('worker_not_ready');
    return {request,close};
  } catch(error) {close();throw error;}
}

const stateKey = snapshot => JSON.stringify({
  room:snapshot.room_url, paused:snapshot.paused, disconnected:snapshot.disconnected, variant:snapshot.variant,
  pot:snapshot.pot, board:snapshot.board, boards:snapshot.board_count,
  players:snapshot.players.map(({normal_clock_percent, bank_clock_percent, ...player})=>player),
});
const compact = text => text.replace(/\s+/g,' ').trim();
const logKey = entry => JSON.stringify({text:compact(entry.text),players:entry.players});

export async function readPokerNowLog(tab, {fresh = false} = {}) {
  let snapshot = await readPokerNow(tab);
  if (fresh && snapshot.log_open) {
    await tab.playwright.getByRole('button',{name:'x Close',exact:true}).click();
    snapshot = await readPokerNow(tab);
  }
  if (!snapshot.log_open) {
    await tab.playwright.getByRole('button',{name:'l Log / Ledger',exact:true}).click();
    try {await tab.playwright.locator('.modal.log-modal').waitFor({state:'visible',timeoutMs:1500});}
    catch(error) {
      const current=await readPokerNow(tab);
      if (current.disconnected) throw new Error('table_disconnected');
      if (current.room_url!==snapshot.room_url) throw new Error('room_changed');
      if (!current.log_open) throw new Error('hand_log_closed');
      throw error;
    }
  }
  const deadline = performance.now()+1500;
  do {
    const result = await readPokerNow(tab);
    if (result.disconnected) throw new Error('table_disconnected');
    if (result.room_url!==snapshot.room_url) throw new Error('room_changed');
    if (!result.log_open) throw new Error('hand_log_closed');
    if (result.log_status==='empty') throw new Error('hand_log_empty');
    if (result.log.length) return result;
    await new Promise(resolve=>setTimeout(resolve,100));
  } while (performance.now()<deadline);
  throw new Error('hand_log_not_loaded');
}

export async function runPrivatePokerNowTurn({tab,request,config,shouldStop=()=>false,expectedTurn=null}) {
  let sent = false;
  const started = performance.now();
  const checkStop = () => {if (shouldStop()) throw new Error('session_stopped');};
  try {
    checkStop();
    const ready = await request({kind:'ping'});
    if (ready.status !== 'worker_ready' || ready.pending) throw new Error('worker_not_ready');
    const first = await readPokerNow(tab);
    if (first.room_url !== config.roomUrl || first.paused ||
        !first.players.some(player=>player.id===config.heroId&&player.hero&&player.active)) throw new Error('not_active_private_turn');
    const logged = await readPokerNowLog(tab);
    if (stateKey(first) !== stateKey(logged)) throw new Error('turn_changed_during_log_read');
    await tab.playwright.getByRole('button',{name:'x Close',exact:true}).click();
    let prepared = await readPokerNow(tab);
    if (stateKey(first) !== stateKey(prepared)) throw new Error('turn_changed');
    const canRaise = prepared.buttons.some(button=>button.enabled&&button.classes.split(/\s+/).includes('raise'));
    if (canRaise || prepared.raise_form) {
      if (!prepared.raise_form) {
        const label = prepared.buttons.find(button=>button.enabled&&button.classes.split(/\s+/).includes('raise')).text;
        await tab.playwright.getByRole('button',{name:/\bBET$/.test(label)?'r Bet':'r Raise',exact:true}).click();
      }
      await tab.playwright.getByRole('button',{name:'Min Raise',exact:true}).click();
      prepared = await readPokerNow(tab);
    }
    if (stateKey(first) !== stateKey(prepared)) throw new Error('turn_changed_during_bet_setup');
    prepared.log = logged.log;
    const normalized = normalizePokerNow(prepared,config,{raiseMinimumVerified:canRaise||prepared.raise_form!==null});
    if (normalized.status !== 'observed') throw new Error(normalized.reason);
    normalized.request.observed_unix_ms = first.captured_unix_ms;
    const input = normalized.request;
    if (expectedTurn && (input.hand_id!==expectedTurn.hand_id || input.observation.history.length!==expectedTurn.turn_index)) throw new Error('expected_turn_changed');
    checkStop();
    let decision = await request(input);
    if (decision.status === 'waiting') {
      if (!Number.isFinite(decision.wait_ms) || decision.wait_ms<0 || decision.wait_ms>2000) throw new Error('unexpected_wait_budget');
      await new Promise(resolve=>setTimeout(resolve,Math.ceil(decision.wait_ms)));
      decision = await request({kind:'status',hand_id:input.hand_id,turn_index:input.observation.history.length});
    }
    if (decision.status !== 'ready' || !decision.action) throw new Error(`worker_${decision.error??decision.status}`);
    checkStop();
    const latest = await readPokerNow(tab);
    if (stateKey(prepared) !== stateKey(latest)) throw new Error('turn_changed_before_send');
    const current = normalizePokerNow({...latest,log:logged.log},config,{raiseMinimumVerified:canRaise||latest.raise_form!==null});
    if (current.status !== 'observed' || JSON.stringify(current.request.observation)!==JSON.stringify(input.observation)) throw new Error('observation_changed_before_send');
    if (Math.abs(current.request.deadline_unix_ms-input.deadline_unix_ms)>350) throw new Error('clock_estimate_changed');
    const reserve = Math.max(1000,decision.reserve_ms);
    const deadline = Math.min(input.deadline_unix_ms,current.request.deadline_unix_ms);
    const inTime = () => {
      if (!Number.isFinite(deadline)||Date.now()>=deadline-reserve) throw new Error('deadline_reserve_exhausted');
    };
    inTime();
    const {kind,amount} = decision.action;
    const obs = input.observation;
    const allowed = kind==='raise_to' ? Number.isSafeInteger(amount)&&obs.min_raise_to!==null&&amount>=obs.min_raise_to&&amount<=obs.max_raise_to :
      amount===0&&((kind==='check'&&obs.to_call===0)||(kind==='call'&&obs.to_call>0)||(kind==='fold'&&obs.can_fold));
    if (!allowed) throw new Error('illegal_worker_action');
    let button;
    if (kind==='raise_to') {
      await tab.playwright.getByRole('textbox').fill(String(amount));
      const filled = await readPokerNow(tab);
      if (stateKey(prepared)!==stateKey(filled)||filled.raise_form?.value!==String(amount)) throw new Error('raise_form_changed');
      button = tab.playwright.getByRole('button',{name:Math.max(...obs.bets)===0?'Bet':'Raise',exact:true});
    } else {
      if (latest.raise_form) {
        await tab.playwright.getByRole('button',{name:'esc Back',exact:true}).click();
        const closed = await readPokerNow(tab);
        if (stateKey(prepared)!==stateKey(closed)) throw new Error('turn_changed_closing_form');
      }
      const name = kind==='check'?'k Check':kind==='fold'?'f Fold':`c Call ${obs.to_call}`;
      button = tab.playwright.getByRole('button',{name,exact:true});
    }
    if (!await button.isEnabled()) throw new Error('action_disabled');
    checkStop();
    inTime();
    const clickBudget = Math.floor(deadline-reserve-Date.now());
    if (clickBudget<=0) throw new Error('deadline_reserve_exhausted');
    sent = true;
    await button.click({timeoutMs:Math.min(500,clickBudget)});
    const after = await readPokerNow(tab);
    let confirmed = await readPokerNowLog(tab);
    const baseline = logged.log;
    if (!baseline.every((entry,index)=>confirmed.log[index]&&logKey(entry)===logKey(confirmed.log[index]))) throw new Error('confirmation_log_changed');
    const expected = kind==='raise_to' ? (Math.max(...obs.bets)===0?`bets ${amount}`:`raises to ${amount}`) :
      kind==='call'?`calls ${obs.bets[obs.actor]+obs.to_call}`:kind==='check'?'checks':'folds';
    const confirmationDeadline = performance.now()+1000;
    const hasNewActionOrBoundary = entries => entries.some(entry=>{
      if (/^-- (starting|ending) hand/.test(entry.text)) return true;
      if (entry.players.length!==1) return false;
      const label = `${entry.players[0].name} `;
      const line = compact(entry.text);
      return line.startsWith(label)&&/^(checks|calls|bets|raises|folds)\b/.test(line.slice(label.length));
    });
    while (!hasNewActionOrBoundary(confirmed.log.slice(baseline.length))&&performance.now()<confirmationDeadline) {
      if (!baseline.every((entry,index)=>confirmed.log[index]&&logKey(entry)===logKey(confirmed.log[index]))) throw new Error('confirmation_log_changed');
      confirmed = await readPokerNow(tab);
    }
    if (!baseline.every((entry,index)=>confirmed.log[index]&&logKey(entry)===logKey(confirmed.log[index]))) throw new Error('confirmation_log_changed');
    const newEntries = confirmed.log.slice(baseline.length);
    let publicAction = null;
    for (const entry of newEntries) {
      if (/^-- (starting|ending) hand/.test(entry.text)) throw new Error('hand_changed_before_confirmation');
      if (entry.players.length!==1) continue;
      const label = `${entry.players[0].name} `;
      const actionText = compact(entry.text).startsWith(label)?compact(entry.text).slice(label.length):'';
      if (!/^(checks|calls|bets|raises|folds)\b/.test(actionText)) continue;
      if (entry.players[0].id===config.heroId&&actionText===expected) publicAction = entry;
      break;
    }
    if (!publicAction) throw new Error('action_not_confirmed_no_retry');
    const event = {actor:obs.actor,street:obs.street,action:decision.action,pot_before:obs.pot,to_call_before:obs.to_call,
      paid:kind==='raise_to'?amount-obs.bets[obs.actor]:kind==='call'?obs.to_call:0,board:obs.board};
    const acknowledgement = await request({kind:'confirm',hand_id:input.hand_id,history:[...obs.history,event]});
    if (acknowledgement.status!=='confirmed') throw new Error('worker_confirmation_failed');
    return {status:'confirmed',hand_id:input.hand_id,action:decision.action,decision_ms:decision.decision_ms,
      scheduled_delay_ms:decision.scheduled_delay_ms,read_to_confirm_ms:performance.now()-started,
      actual_observation_to_confirm_ms:Date.now()-first.captured_unix_ms,
      input,after,confirmed_log:confirmed.log,source:'one private supervised turn; no automatic retries'};
  } catch (error) {
    return {status:sent?'uncertain':'blocked',action_sent:sent,reason:error instanceof Error?error.message:'bridge_failed',
      elapsed_ms:performance.now()-started,retry_allowed:false};
  }
}

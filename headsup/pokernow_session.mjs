import {readPokerNow, parsePokerNowHand} from './pokernow_state.mjs';
import {readPokerNowLog, runPrivatePokerNowTurn, startDecisionWorker, pokerNowMemoryIdentity} from './pokernow_bridge.mjs';
import {isAbsolute, relative, resolve} from 'node:path';

const controllers = new Set();
const same = (left,right) => JSON.stringify(left)===JSON.stringify(right);

export async function createPrivatePokerNowSession({tab,config,worker=null,seed=20261106,maxHands=3,maxSeconds=300,
  persistentLearning=true,memoryDirectory=null,memory='lifetime',populationPrior=null,policy='adaptive',onEvent=()=>{}}) {
  if (config?.privatePlayMoney!==true || config.noRakeAntesStraddleVerified!==true) throw new Error('confirm_private_play_chip_table');
  if (!Number.isSafeInteger(maxHands)||maxHands<1||maxHands>100||!Number.isFinite(maxSeconds)||maxSeconds<1||maxSeconds>1800) throw new Error('invalid_session_limits');
  if (typeof persistentLearning!=='boolean'||!['lifetime','recent'].includes(memory)||!['adaptive','river'].includes(policy)) throw new Error('invalid_learning_settings');
  if ([memoryDirectory,populationPrior].some(path=>path!==null&&(typeof path!=='string'||!path.trim()))) throw new Error('invalid_memory_path');
  if (!persistentLearning&&(memoryDirectory!==null||populationPrior!==null||memory!=='lifetime')) throw new Error('persistent_memory_disabled');
  const pinned = structuredClone(config);
  const expectedIdentity=persistentLearning?pokerNowMemoryIdentity(pinned):null;
  const key = `${pinned.roomUrl}:${pinned.heroId}`;
  if (controllers.has(key)) throw new Error('seat_already_controlled');
  controllers.add(key);
  const started = performance.now();
  const deadline = started+maxSeconds*1000;
  let engine=worker,active=null,baseline=null,busy=false,stopped=false,stopNow=false,stopAfterHand=false;
  let phase='starting',reason=null,away=false,learning=null,learningUpdates=0;
  let interruptedLogReads=0;
  const hands=[],turns=[];
  const state = () => ({phase,reason,stopped,completed_hands:hands.length,decisions:turns.length,
    current_hand:active?.hand_id??null,max_hands:maxHands,elapsed_ms:performance.now()-started,away,
    learning:structuredClone(learning),learning_updates:learningUpdates});
  const emit = (kind,detail={}) => onEvent({kind,...detail,state:state()});
  const mustStop = () => stopNow||performance.now()>=deadline;
  const updateLearning = response => {
    if (!persistentLearning) {
      if (response.learning?.enabled) throw new Error('unexpected_persistent_worker');
      learning={enabled:false};
      return;
    }
    const report=response.learning,identity=report?.identity;
    if (report?.enabled!==true||!identity||Object.keys(identity).length!==Object.keys(expectedIdentity).length||
        !Object.entries(expectedIdentity).every(([key,value])=>identity[key]===value)) throw new Error('worker_learning_identity_mismatch');
    if (report.memory!==memory||response.policy!==policy) throw new Error('worker_learning_settings_mismatch');
    if (populationPrior!==null&&(report.population_prior_enabled!==true||report.population_prior_path!==resolve(populationPrior))) throw new Error('worker_population_prior_mismatch');
    if (!Number.isSafeInteger(report.profile_hands)||report.profile_hands<0||!Number.isSafeInteger(report.total_responses)||report.total_responses<0||
        !/^[0-9a-f]{64}$/.test(report.profile_sha256)||typeof report.state_path!=='string'||!isAbsolute(report.state_path)||
        typeof report.population_prior_enabled!=='boolean'||report.minimum_responses!==24||
        !Number.isSafeInteger(response.learning_updates)||response.learning_updates<learningUpdates) throw new Error('invalid_worker_learning_report');
    if (report.population_prior_enabled&&(!/^[0-9a-f]{64}$/.test(report.population_prior_sha256)||
        typeof report.population_prior_path!=='string'||!isAbsolute(report.population_prior_path))) throw new Error('invalid_worker_population_report');
    if (memoryDirectory!==null) {
      const stored=relative(resolve(memoryDirectory),report.state_path);
      if (isAbsolute(stored)||stored==='..'||stored.startsWith('../')) throw new Error('worker_memory_directory_mismatch');
    }
    if (learning&&report.profile_hands<learning.profile_hands) throw new Error('opponent_memory_regressed');
    learning=structuredClone(report);learningUpdates=response.learning_updates;
  };
  const validateTable = snapshot => {
    if (snapshot.disconnected) throw new Error('table_disconnected');
    if (snapshot.room_url!==pinned.roomUrl) throw new Error('room_changed');
    if (!Array.isArray(pinned.playerIds)||pinned.playerIds.length!==2||new Set(pinned.playerIds).size!==2||
        !same(snapshot.players.map(player=>player.id).sort(),[...pinned.playerIds].sort())) throw new Error('players_changed');
    const hero=snapshot.players.find(player=>player.hero);
    if (!hero||hero.id!==pinned.heroId||snapshot.players.filter(player=>player.hero).length!==1) throw new Error('hero_changed');
    const blinds=snapshot.variant?.match(/^NLH\s*~?\s*(\d+)\s*\/\s*(\d+)$/);
    if (!blinds||Number(blinds[1])!==pinned.smallBlind||Number(blinds[2])!==pinned.bigBlind||snapshot.board_count!==1) throw new Error('table_rules_changed');
    away=snapshot.buttons.some(item=>/I.?M BACK/i.test(item.text));
    return hero;
  };
  const closeLog = async snapshot => {
    if (snapshot.log_open) await tab.playwright.getByRole('button',{name:'x Close',exact:true}).click();
  };
  const sitOut = async () => {
    const snapshot=await readPokerNow(tab);
    if (snapshot.disconnected||snapshot.room_url!==pinned.roomUrl||!snapshot.players.some(player=>player.hero&&player.id===pinned.heroId)) return false;
    await closeLog(snapshot);
    const current=await readPokerNow(tab);
    if (current.raise_form) await tab.playwright.getByRole('button',{name:'esc Back',exact:true}).click();
    const controls=await readPokerNow(tab);
    const button=controls.buttons.find(item=>item.classes.split(/\s+/).includes('stand-up'));
    if (button?.text==='AWAY'&&button.enabled) await tab.playwright.getByRole('button',{name:' Away',exact:true}).click();
    const result=await readPokerNow(tab);
    away=result.buttons.some(item=>/I.?M BACK/i.test(item.text));
    return away;
  };
  const halt = async (why,failed=false) => {
    if (stopped) return state();
    reason=why;stopped=true;phase=failed?'blocked':'stopped';
    engine?.close();
    let sitOutError=null;
    try {await sitOut();} catch(error) {away=false;sitOutError=error?.message??String(error);}
    controllers.delete(key);
    emit('stopped',{manual_sit_out_required:!away,sit_out_error:sitOutError});
    return state();
  };
  const complete = (source) => {
    hands.push({hand_id:active.hand_id,hand_number:active.hand_number,history:active.history,
      worker_used:active.worker_used,completion_source:source,settlement_verified:false,
      learning_updated:active.learning_updated,profile_hands:active.profile_hands,profile_sha256:active.profile_sha256});
    baseline={hand_id:active.hand_id,hand_number:active.hand_number};
    emit('hand_completed',{hand_id:active.hand_id,completion_source:source});
    active=null;
  };
  const observeHand = async parsed => {
    if (active&&parsed.hand_id!==active.hand_id) {
      if (!active.finished||!active.terminal||parsed.hand_number!==active.hand_number+1) throw new Error('previous_hand_completion_missing');
      complete('next_hand_after_confirmed_terminal');
    }
    if (!active) {
      if (parsed.hand_id===baseline?.hand_id) return false;
      if (baseline&&parsed.hand_number!==baseline.hand_number+1) throw new Error('hand_sequence_changed');
      if (hands.length>=maxHands||stopAfterHand) return false;
      active={...parsed,history:[],worker_used:false,finished:false,learning_updated:false,
        profile_hands:learning?.profile_hands??null,profile_sha256:learning?.profile_sha256??null};
      emit('hand_started',{hand_id:parsed.hand_id});
    }
    if (!same(parsed.history.slice(0,active.history.length),active.history)) throw new Error('public_history_changed');
    if (parsed.history.slice(active.history.length).some(event=>event.actor===parsed.actor)) throw new Error('unconfirmed_own_action');
    active={...active,...parsed};
    if (parsed.terminal&&!active.finished) {
      if (active.worker_used||persistentLearning) {
        const result=await engine.request({kind:'finish',hand_id:parsed.hand_id,hero_seat:parsed.actor,history:parsed.history});
        if (result.status!=='finished') throw new Error(`worker_finish_${result.error??result.status}`);
        if (persistentLearning&&typeof result.learning_updated!=='boolean') throw new Error('missing_learning_receipt');
        if (persistentLearning&&(result.learning?.profile_hands!==learning.profile_hands+Number(result.learning_updated)||
            result.learning_updates!==learningUpdates+Number(result.learning_updated))) throw new Error('learning_receipt_count_mismatch');
        updateLearning(result);
        active.learning_updated=result.learning_updated??false;
        active.profile_hands=learning?.profile_hands??null;
        active.profile_sha256=learning?.profile_sha256??null;
      }
      active.finished=true;
      if (hands.length+1>=maxHands||stopAfterHand) {
        phase='finishing';
        if (!await sitOut()) throw new Error('sit_out_not_confirmed');
      }
    }
    if (parsed.ended) {
      if (!active.finished) throw new Error('hand_ended_without_terminal_history');
      complete('visible_hand_end');
    }
    return true;
  };
  try {
    engine??=await startDecisionWorker({seed,config:persistentLearning?pinned:null,memoryDirectory,memory,populationPrior,policy});
    const ready=await engine.request({kind:'ping'});
    if (ready.status!=='worker_ready'||ready.pending) throw new Error('worker_not_ready');
    updateLearning(ready);
    const snapshot=await readPokerNow(tab);
    const hero=validateTable(snapshot);
    if (snapshot.pot!=='0'||hero.hole.length||snapshot.players.some(player=>player.active)) throw new Error('start_between_hands');
    const logged=await readPokerNowLog(tab,{fresh:true});
    validateTable(logged);
    const parsed=parsePokerNowHand(logged.log,pinned);
    if (parsed.status!=='parsed'||!parsed.ended) throw new Error('need_completed_hand_baseline');
    baseline={hand_id:parsed.hand_id,hand_number:parsed.hand_number};
    await closeLog(logged);
    phase='waiting';
    emit('ready');
  } catch(error) {
    await halt(error?.message??String(error),true);
    throw error;
  }

  const step = async () => {
    if (stopped) return state();
    if (busy) throw new Error('session_step_in_progress');
    busy=true;
    try {
      if (mustStop()) return await halt(stopNow?'requested_stop':'time_limit');
      if (!active&&(stopAfterHand||hands.length>=maxHands)) return await halt(stopAfterHand?'requested_stop':'hand_limit');
      const snapshot=await readPokerNow(tab);
      validateTable(snapshot);
      if (snapshot.paused) {phase='paused';return state();}
      let logged;
      try {logged=await readPokerNowLog(tab,{fresh:true});}
      catch(error) {
        if (error.message==='hand_log_closed'&&++interruptedLogReads<3) {phase='waiting';return state();}
        throw error;
      }
      interruptedLogReads=0;
      const hero=validateTable(logged);
      if (mustStop()) return await halt(stopNow?'requested_stop':'time_limit');
      const parsed=parsePokerNowHand(logged.log,pinned);
      if (parsed.status!=='parsed') throw new Error(parsed.reason);
      const observed=await observeHand(parsed);
      if ((!active||active.finished)&&(stopAfterHand||hands.length>=maxHands)) {
        if (!active) return await halt(stopAfterHand?'requested_stop':'hand_limit');
      }
      if (!observed||!active||active.terminal||!hero.active||parsed.expected_actor!==parsed.actor) {
        phase=active?.terminal?'finishing':'waiting';
        await closeLog(await readPokerNow(tab));
        return state();
      }
      phase='playing';
      const result=await runPrivatePokerNowTurn({tab,request:engine.request,config:pinned,shouldStop:mustStop,
        expectedTurn:{hand_id:parsed.hand_id,turn_index:parsed.history.length}});
      if (result.status!=='confirmed') throw new Error(`${result.status}:${result.reason}`);
      const receipt=parsePokerNowHand(result.confirmed_log,pinned,{handId:parsed.hand_id});
      if (receipt.status!=='parsed') throw new Error(`receipt_${receipt.reason}`);
      const event=receipt.history[parsed.history.length];
      if (!event||event.actor!==parsed.actor||!same(event.action,result.action)) throw new Error('receipt_action_mismatch');
      active.history=[...parsed.history,event];active.worker_used=true;
      turns.push(result);
      emit('action_confirmed',{hand_id:result.hand_id,action:result.action,latency_ms:result.actual_observation_to_confirm_ms});
      await observeHand(receipt);
      await closeLog(await readPokerNow(tab));
      if (!active&&(stopAfterHand||hands.length>=maxHands)) return await halt(stopAfterHand?'requested_stop':'hand_limit');
      return state();
    } catch(error) {
      return await halt(error?.message??String(error),true);
    } finally {busy=false;}
  };
  return {
    state,
    results:()=>({state:state(),hands:structuredClone(hands),turns:structuredClone(turns)}),
    requestStop:({afterHand=false}={})=>{if(afterHand)stopAfterHand=true;else stopNow=true;},
    step,
    runFor:async(milliseconds=5000)=>{
      if (!Number.isFinite(milliseconds)||milliseconds<1||milliseconds>10000) throw new Error('invalid_poll_window');
      const until=performance.now()+milliseconds;
      do {
        await step();
        if (stopped||performance.now()>=until) break;
        await new Promise(resolve=>setTimeout(resolve,250));
      } while(performance.now()<until);
      return state();
    },
  };
}

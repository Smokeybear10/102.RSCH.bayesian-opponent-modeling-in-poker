export function capturePokerNow() {
  const visible = el => el && el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden';
  const text = (root, selector) => root.querySelector(selector)?.innerText?.trim() ?? null;
  const cards = root => Array.from(root?.querySelectorAll('.card-container.flipped') ?? []).map(el => {
    const rank = text(el, '.value');
    return `${rank === '10' ? 'T' : rank}${text(el, '.sub-suit')}`;
  });
  const players = Array.from(document.querySelectorAll('.table-player')).filter(el=>visible(el)&&!el.classList.contains('table-player-seat')).map(el => ({
    id: el.querySelector('.table-player-name a')?.getAttribute('href')?.split('/').pop() ?? null,
    name: text(el, '.table-player-name'),
    hero: el.classList.contains('you-player'),
    active: el.classList.contains('decision-current'),
    stack: text(el, '.table-player-stack .normal-value'),
    bet: text(el, '.table-player-bet-value .normal-value') ?? '0',
    hole: el.classList.contains('you-player') ? cards(el.querySelector('.table-player-cards')) : [],
    normal_clock_percent: el.querySelector('.normal-time')?.style.width ?? null,
    bank_clock_percent: el.querySelector('.time-to-talk .time-bank')?.style.width ?? null,
  }));
  const form = Array.from(document.querySelectorAll('input[type="range"]')).find(visible);
  const logModal = Array.from(document.querySelectorAll('.modal.log-modal')).find(visible);
  const logText = logModal?.querySelector('.log-modal-entries')?.innerText?.trim();
  return {
    captured_unix_ms: Date.now(),
    variant: text(document, '.blind-value'),
    paused: visible(document.querySelector('.room-owner-paused')),
    disconnected: Boolean(visible(document.querySelector('.disconnect-warning'))),
    players,
    board: cards(document.querySelector('.table-cards.run-1')),
    board_count: Array.from(document.querySelectorAll('.table-cards')).filter(visible).length,
    pot: text(document, '.table-pot-size .add-on .normal-value'),
    buttons: Array.from(document.querySelectorAll('button')).filter(visible).map(el => ({
      text: el.innerText.trim(), enabled: !el.disabled, classes: el.className,
    })),
    raise_form: form ? {value: form.value, max: form.max} : null,
    log_open: Boolean(logModal),
    log_status: !logModal ? 'closed' : logText==='Loading...' ? 'loading' : logText==='Empty Log' ? 'empty' : 'ready',
    log: Array.from(document.querySelectorAll('.log-modal-entries .entry-ctn')).filter(visible).map(el => ({
      text: text(el, '.content'),
      players: Array.from(el.querySelectorAll('abbr[title]')).map(player => ({
        id: player.getAttribute('title')?.replace(/^Player ID: /, ''), name: player.innerText,
      })),
    })).reverse(),
  };
}

export async function readPokerNow(tab) {
  const snapshot = await tab.playwright.evaluate(capturePokerNow);
  return {...snapshot, room_url:await tab.url()};
}

const chips = value => {
  if (typeof value !== 'string' || !/^\d+$/.test(value)) throw new Error('ambiguous_chip_amount');
  const result = Number(value);
  if (!Number.isSafeInteger(result)) throw new Error('invalid_chip_amount');
  return result;
};
const cardCodes = text => Array.from(text.matchAll(/(10|[2-9TJQKA])([♣♦♥♠])/g), match =>
  `${match[1] === '10' ? 'T' : match[1]}${({'♣':'c','♦':'d','♥':'h','♠':'s'})[match[2]]}`);
const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);


export function parsePokerNowHand(log, config, {handId = null} = {}) {
  try {
    if (!config || !Array.isArray(config.playerIds) || config.playerIds.length !== 2 ||
        !config.playerIds.every(id=>typeof id==='string'&&id.length>0) || new Set(config.playerIds).size !== 2 ||
        !config.playerIds.includes(config.heroId)) throw new Error('pin_two_test_players');
    const {smallBlind,bigBlind} = config;
    if (![smallBlind,bigBlind].every(Number.isSafeInteger) || smallBlind<=0 || bigBlind<smallBlind) throw new Error('verify_table_rules');
    if (!Array.isArray(log) || !log.every(entry=>entry&&typeof entry.text==='string'&&Array.isArray(entry.players)&&
        entry.players.every(player=>typeof player.id==='string'&&typeof player.name==='string'&&player.name.length>0))) throw new Error('unreadable_hand_log');
    const starting = entry => entry.text.match(/^-- starting hand #(\d+) \(id: ([A-Za-z0-9_-]+)\)/);
    const startIndex = log.findLastIndex(entry=>{const match=starting(entry);return match&&(handId===null||match[2]===handId);});
    if (startIndex<0) throw new Error('need_current_hand_log');
    const nextIndex = log.findIndex((entry,index)=>index>startIndex&&starting(entry));
    const entries = log.slice(startIndex,nextIndex<0?undefined:nextIndex);
    const start = entries[0], identity = starting(start);
    const number = chips(identity[1]);
    if (start.players.length!==1) throw new Error('hand_identity_unreadable');
    const buttonId = start.players[0].id;
    if (!config.playerIds.includes(buttonId)) throw new Error('dealer_changed');
    const ids = [config.playerIds.find(id=>id!==buttonId),buttonId];
    const actor = ids.indexOf(config.heroId);
    const stackEntries = entries.filter(entry=>entry.text.startsWith('Player stacks:'));
    const stackEntry = stackEntries[0];
    const amounts = stackEntry ? Array.from(stackEntry.text.matchAll(/\((\d+)\)/g),match=>chips(match[1])) : [];
    if (stackEntries.length!==1 || stackEntry.players.length!==2 || amounts.length!==2 ||
        !same(stackEntry.players.map(player=>player.id).sort(),[...ids].sort())) throw new Error('starting_stacks_unreadable');
    const startingStacks = ids.map(id=>amounts[stackEntry.players.findIndex(player=>player.id===id)]);
    if (startingStacks.some(stack=>stack<=0) || !Number.isSafeInteger(startingStacks[0]+startingStacks[1])) throw new Error('starting_stacks_unreadable');
    const stacks = [...startingStacks], history = [], blinds = new Set();
    let bets=[0,0],pot=0,street='preflop',board=[],expectedActor=1,awaiting=new Set([0,1]);
    let folded=false,ended=false,terminal=false;
    for (const entry of entries.slice(1)) {
      const line = entry.text;
      const end = line.match(/^-- ending hand #(\d+) --$/);
      if (end) {
        if (ended || chips(end[1])!==number || !terminal) throw new Error('hand_end_inconsistent');
        ended=true;
        continue;
      }
      if (/^-- (starting|ending) hand/.test(line)) throw new Error('unsupported_hand_boundary');
      const nextStreet = line.match(/^(Flop|Turn|River):/);
      if (nextStreet) {
        if (ended||folded) throw new Error('action_after_hand_end');
        if (blinds.size!==2 || awaiting.size || (bets[0]!==bets[1]&&stacks.every(stack=>stack>0))) throw new Error('betting_round_incomplete');
        const nextBoard=cardCodes(line),required={Flop:3,Turn:4,River:5}[nextStreet[1]];
        if (nextBoard.length!==required || new Set(nextBoard).size!==required || !same(nextBoard.slice(0,board.length),board) ||
            required!==(board.length===0?3:board.length+1)) throw new Error('board_history_changed');
        board=nextBoard;street=nextStreet[1].toLowerCase();bets=[0,0];
        expectedActor=0;awaiting=terminal?new Set():new Set([0,1]);
        continue;
      }
      const settlement = /^Uncalled bet of \d+ returned to /.test(line) ||
        entry.players.some(player=>line.startsWith(`${player.name} collected `)||line.startsWith(`${player.name} shows `));
      if (settlement) {
        if (!terminal) throw new Error('settlement_before_terminal_action');
        continue;
      }
      if (entry.players.length!==1) continue;
      const seat=ids.indexOf(entry.players[0].id),name=entry.players[0].name;
      if (!line.startsWith(`${name} `)) continue;
      const actionText=line.slice(name.length).trim();
      const isAction=/^(posts|calls|checks|raises|bets|folds|goes)\b/.test(actionText);
      if (seat<0) {if(isAction) throw new Error('players_changed');continue;}
      const blind=actionText.match(/^posts a (small|big) blind of (\d+)$/);
      if (blind) {
        const amount=chips(blind[2]);
        if (ended||terminal||street!=='preflop'||history.length||blinds.has(seat)||
            seat!==(blind[1]==='small'?1:0)||amount!==Math.min(stacks[seat],seat===1?smallBlind:bigBlind)) throw new Error('unsupported_blind_posts');
        blinds.add(seat);bets[seat]=amount;stacks[seat]-=amount;pot+=amount;
        if (!stacks[seat]) awaiting.delete(seat);
        continue;
      }
      const match=actionText.match(/^(checks|folds|calls (\d+)|raises to (\d+)|bets (\d+))$/);
      if (!match) {if(isAction) throw new Error('unsupported_action_log');continue;}
      if (ended||terminal) throw new Error('action_after_hand_end');
      if (blinds.size!==2) throw new Error('missing_blinds');
      if (seat!==expectedActor||!awaiting.has(seat)) throw new Error('action_order_inconsistent');
      const kind=match[1]==='checks'?'check':match[1]==='folds'?'fold':match[2]?'call':'raise_to';
      const toCall=Math.min(stacks[seat],Math.max(...bets)-bets[seat]);
      const total=kind==='check'||kind==='fold'?bets[seat]:chips(match[2]??match[3]??match[4]);
      const paid=total-bets[seat];
      if (paid<0||paid>stacks[seat]||(kind==='check'&&toCall!==0)||(kind==='fold'&&toCall===0)||
          (kind==='call'&&(paid!==toCall||toCall===0))||(kind==='raise_to'&&paid<=toCall)) throw new Error('action_chips_inconsistent');
      history.push({actor:seat,street,action:{kind,amount:kind==='raise_to'?total:0},pot_before:pot,to_call_before:toCall,paid,board:[...board]});
      bets[seat]=total;stacks[seat]-=paid;pot+=paid;
      if (kind==='fold') {folded=true;awaiting.clear();}
      else if (kind==='raise_to') awaiting=new Set([1-seat]);
      else awaiting.delete(seat);
      for (const waitingSeat of awaiting) if (!stacks[waitingSeat]) awaiting.delete(waitingSeat);
      terminal=folded||(!awaiting.size&&(street==='river'||(kind==='call'&&Math.min(...stacks)===0)));
      expectedActor=1-seat;
    }
    if (blinds.size!==2) throw new Error('missing_blinds');
    if (nextIndex>=0&&!ended) throw new Error('hand_end_missing_before_next_hand');
    return {status:'parsed',hand_id:identity[2],hand_number:number,player_ids:ids,actor,button:1,starting_stacks:startingStacks,
      stacks,bets,pot,board,street,history,terminal,ended,expected_actor:terminal||!awaiting.size?null:expectedActor,awaiting:[...awaiting]};
  } catch (error) {
    return {status:'blocked',reason:error instanceof Error?error.message:'unreadable_hand_log'};
  }
}

export function normalizePokerNow(snapshot, config, {raiseMinimumVerified = false} = {}) {
  try {
    if (snapshot.disconnected) throw new Error('table_disconnected');
    if (!config || !Array.isArray(config.playerIds) || config.playerIds.length !== 2 ||
        new Set(config.playerIds).size !== 2 || !config.playerIds.includes(config.heroId)) throw new Error('pin_two_test_players');
    if (!/^https:\/\/www\.pokernow\.com\/games\/[A-Za-z0-9_-]+$/.test(config.roomUrl ?? '') ||
        snapshot.room_url !== config.roomUrl || config.noRakeAntesStraddleVerified !== true) throw new Error('verify_private_test_table');
    const {decisionSeconds, timeBankSeconds, smallBlind, bigBlind} = config;
    if (![decisionSeconds, timeBankSeconds, smallBlind, bigBlind].every(Number.isFinite) ||
        decisionSeconds <= 0 || timeBankSeconds < 0 || smallBlind <= 0 || bigBlind < smallBlind ||
        !Number.isSafeInteger(smallBlind) || !Number.isSafeInteger(bigBlind)) throw new Error('verify_table_rules');
    if (!same(snapshot.players.map(player => player.id).sort(), [...config.playerIds].sort())) throw new Error('players_changed');
    if (snapshot.board_count !== 1 || !/^NLH\s*~?\s*\d+\s*\/\s*\d+$/.test(snapshot.variant ?? '')) throw new Error('unsupported_table');
    const displayedBlinds = snapshot.variant.match(/(\d+)\s*\/\s*(\d+)$/);
    if (Number(displayedBlinds[1]) !== smallBlind || Number(displayedBlinds[2]) !== bigBlind) throw new Error('blinds_changed');
    if (snapshot.paused) throw new Error('table_paused');
    const hero = snapshot.players.find(player => player.hero);
    if (!hero || hero.id !== config.heroId || snapshot.players.filter(player => player.hero).length !== 1) throw new Error('hero_changed');
    if (!hero.active || snapshot.players.filter(player => player.active).length !== 1) throw new Error('not_our_turn');
    if (hero.hole.length !== 2 || ![...hero.hole, ...snapshot.board].every(card => /^[2-9TJQKA][cdhs]$/.test(card))) throw new Error('cards_unreadable');
    const parsed = parsePokerNowHand(snapshot.log, config);
    if (parsed.status !== 'parsed') throw new Error(parsed.reason);
    if (parsed.terminal || parsed.ended) throw new Error('hand_ended');
    const {hand_id:handId,player_ids:ids,actor,stacks,bets,pot,street,board,history,expected_actor:expectedActor,awaiting} = parsed;
    if (!same(board,snapshot.board) || pot !== chips(snapshot.pot) ||
        !ids.every((id, seat) => {const player=snapshot.players.find(item=>item.id===id);return stacks[seat]===chips(player.stack)&&bets[seat]===chips(player.bet);})) throw new Error('snapshot_history_mismatch');
    if (expectedActor !== actor || !awaiting.includes(actor)) throw new Error('active_turn_history_mismatch');
    const enabled = name => snapshot.buttons.some(button => button.enabled && button.classes.split(/\s+/).includes(name));
    const toCall = Math.min(stacks[actor], Math.max(...bets)-bets[actor]);
    if (toCall ? !enabled('call') && !snapshot.raise_form : !enabled('check') && !snapshot.raise_form) throw new Error('action_controls_changed');
    const canRaise = enabled('raise') || snapshot.raise_form !== null;
    if (canRaise && (!raiseMinimumVerified || !snapshot.raise_form)) throw new Error('verify_raise_bounds');
    const minRaise = canRaise ? chips(snapshot.raise_form.value) : null;
    const maxRaise = canRaise ? chips(snapshot.raise_form.max) : null;
    if (canRaise && !(minRaise>Math.max(...bets)&&minRaise<=maxRaise&&maxRaise<=stacks[actor]+bets[actor])) throw new Error('invalid_raise_bounds');
    const percentText = hero.normal_clock_percent;
    const bankText = hero.bank_clock_percent;
    if (![percentText,bankText].every(value=>typeof value==='string'&&/^\d+(?:\.\d+)?%$/.test(value))) throw new Error('clock_unreadable');
    const percent = Number(percentText.slice(0,-1));
    const bankPercent = Number(bankText.slice(0,-1));
    if (percent<=0||bankPercent<0||bankPercent>=100||percent+bankPercent>100.01||
        bankPercent>100*timeBankSeconds/(decisionSeconds+timeBankSeconds)+0.01) throw new Error('clock_exhausted_or_changed');
    const remainingMs = Math.floor(percent/(100-bankPercent)*decisionSeconds*1000)-100;
    if (remainingMs <= 1000) throw new Error('clock_reserve_exhausted');
    return {status:'observed',deadline_source:'normal countdown bar, manually verified table timing; time bank excluded',request:{
      kind:'decide',hand_id:handId,observed_unix_ms:snapshot.captured_unix_ms,deadline_unix_ms:snapshot.captured_unix_ms+remainingMs,
      observation:{actor,hole:hero.hole,board,street,button:1,stacks,bets,pot,to_call:toCall,min_raise_to:minRaise,max_raise_to:maxRaise,
        can_fold:toCall>0,history,big_blind:bigBlind,rake_bps:0,rake_cap:0,no_flop_no_drop:true},
    }};
  } catch (error) {
    return {status:'blocked',reason:error instanceof Error?error.message:'unreadable_state'};
  }
}

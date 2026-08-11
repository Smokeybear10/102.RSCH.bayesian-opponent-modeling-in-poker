import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {setTimeout as delay} from 'node:timers/promises';
import {parsePokerNowHand, normalizePokerNow} from '../headsup/pokernow_state.mjs';
import {pokerNowMemoryIdentity, readPokerNowLog, runPrivatePokerNowTurn, startDecisionWorker} from '../headsup/pokernow_bridge.mjs';
import {createPrivatePokerNowSession} from '../headsup/pokernow_session.mjs';

const config = {
  roomUrl: 'https://www.pokernow.com/games/synthetic-test-room',
  heroId: 'hero', playerIds: ['hero', 'villain'], smallBlind: 10, bigBlind: 20,
  privatePlayMoney: true, noRakeAntesStraddleVerified: true,
  decisionSeconds: 20, timeBankSeconds: 10,
};
const player = (id, name = id) => ({id, name});
const entry = (text, players = []) => ({text, players});
const action = (id, text, name = id) => entry(`${name} ${text}`, [player(id, name)]);

function startLog({id = 'hand1', number = 1, button = 'hero', names = {hero: 'hero', villain: 'villain'}} = {}) {
  const bb = button === 'hero' ? 'villain' : 'hero';
  return [
    entry(`-- starting hand #${number} (id: ${id}) No Limit Texas Hold'em (dealer: ${names[button]}) --`, [player(button, names[button])]),
    entry(`Player stacks: #1 ${names.hero} (2000) | #2 ${names.villain} (2000)`, [player('hero', names.hero), player('villain', names.villain)]),
    action(button, 'posts a small blind of 10', names[button]),
    action(bb, 'posts a big blind of 20', names[bb]),
  ];
}

function foldedLog(id = 'baseline', number = 0) {
  return [...startLog({id, number, button: 'villain'}), action('villain', 'folds'), entry(`-- ending hand #${number} --`)];
}

function activeSnapshot() {
  return {
    room_url: config.roomUrl, captured_unix_ms: Date.now(), disconnected: false,
    paused: false, variant: 'NLH ~ 10 / 20', board_count: 1, board: [], pot: '30',
    players: [
      {id: 'hero', name: 'hero', hero: true, active: true, stack: '1990', bet: '10', hole: ['As', 'Ah'], normal_clock_percent: '60%', bank_clock_percent: '20%'},
      {id: 'villain', name: 'villain', hero: false, active: false, stack: '1980', bet: '20', hole: [], normal_clock_percent: null, bank_clock_percent: null},
    ],
    buttons: [{classes: 'call', text: 'CALL 10', enabled: true}],
    raise_form: null, log: startLog(), log_open: true, log_status: 'ready',
  };
}

function mockTab(snapshot) {
  const model = structuredClone(snapshot);
  const clicks = [];
  let interrupted = 0;
  return {
    model, clicks,
    interruptLogReads(count) { interrupted = count; },
    url: async () => model.room_url,
    playwright: {
      evaluate: async () => structuredClone(model),
      locator: () => ({waitFor: async () => { if (!model.log_open) throw new Error('modal closed'); }}),
      getByRole: (_role, {name}) => ({
        isEnabled: async () => true,
        click: async () => {
          clicks.push(name);
          if (name === 'l Log / Ledger') {
            model.log_open = interrupted === 0;
            if (interrupted) interrupted--;
          } else if (name === 'x Close') {
            model.log_open = false;
          } else if (name === ' Away') {
            model.buttons = [{text: "I'M BACK", classes: 'sit-back', enabled: true}];
          } else {
            throw new Error(`unexpected action: ${name}`);
          }
        },
      }),
    },
  };
}

function idleTab() {
  const snapshot = activeSnapshot();
  Object.assign(snapshot, {pot: '0', log: foldedLog(), log_open: false});
  snapshot.players = snapshot.players.map(p => ({...p, active: false, hole: [], stack: '2000', bet: '0'}));
  snapshot.buttons = [{text: 'AWAY', classes: 'stand-up', enabled: true}];
  return mockTab(snapshot);
}

function mockWorker(overrides = {}) {
  let updates = 0;
  const requests = [];
  const report = () => ({
    enabled: true, identity: pokerNowMemoryIdentity(config), memory: 'lifetime',
    profile_hands: updates, total_responses: 0, profile_sha256: 'a'.repeat(64),
    state_path: join(tmpdir(), 'synthetic-profile.sqlite3'), minimum_responses: 24,
    population_prior_enabled: false, population_prior_path: null, population_prior_sha256: null,
    ...overrides,
  });
  return {
    requests, closed: false,
    close() { this.closed = true; },
    async request(value) {
      requests.push(value);
      if (value.kind === 'finish') {
        updates++;
        return {status: 'finished', hand_id: value.hand_id, learning_updated: true,
          learning_updates: updates, policy: 'adaptive', learning: report()};
      }
      assert.equal(value.kind, 'ping');
      return {status: 'worker_ready', pending: false, learning_updates: updates, policy: 'adaptive', learning: report()};
    },
  };
}

function foldTurn(confirmedPlayer = 'hero') {
  const tab = mockTab(activeSnapshot());
  const requests = [];
  const locator = tab.playwright.getByRole;
  tab.playwright.getByRole = (role, options) => options.name === 'f Fold' ? {
    isEnabled: async () => true,
    click: async () => {
      tab.clicks.push('f Fold');
      tab.model.players[0].active = false;
      tab.model.log.push(action(confirmedPlayer, 'folds'));
    },
  } : locator(role, options);
  const request = async payload => {
    requests.push(payload);
    if (payload.kind === 'ping') return {status: 'worker_ready', pending: false};
    if (payload.kind === 'decide') return {status: 'ready', action: {kind: 'fold', amount: 0}, reserve_ms: 1000};
    assert.equal(payload.kind, 'confirm');
    return {status: 'confirmed'};
  };
  return {tab, request, requests};
}

test('parser maps player identity independently of display names and configured order', () => {
  const names = {hero: 'Same Name', villain: 'Same Name'};
  const log = [...startLog({names}), action('hero', 'raises to 50', names.hero), action('villain', 'folds', names.villain)];
  const parsed = parsePokerNowHand(log, {...config, playerIds: [...config.playerIds].reverse()});
  assert.equal(parsed.status, 'parsed');
  assert.equal(parsed.actor, 1);
  assert.deepEqual(parsed.history.map(e => e.actor), [1, 0]);
  assert.equal(parsed.terminal, true);
});

test('dealer rotation maps the same hero to the big-blind position', () => {
  const parsed = parsePokerNowHand(foldedLog(), config);
  assert.equal(parsed.actor, 0);
  assert.equal(parsed.history[0].actor, 1);
  assert.equal(parsed.ended, true);
});

test('a call is interpreted as a street total rather than incremental chips', () => {
  const log = [...startLog(), action('hero', 'raises to 50'), action('villain', 'calls 50')];
  const parsed = parsePokerNowHand(log, config);
  assert.equal(parsed.status, 'parsed');
  assert.deepEqual(parsed.history.map(e => e.paid), [40, 30]);
  assert.equal(parsed.pot, 100);
  assert.deepEqual(parsed.stacks, [1950, 1950]);
});

test('complete checked-down showdown preserves board and terminal history', () => {
  const log = [...startLog(), action('hero', 'calls 20'), action('villain', 'checks')];
  for (const [street, cards] of [['Flop', '2♣, 3♦, 4♥'], ['Turn', '2♣, 3♦, 4♥, 5♠'], ['River', '2♣, 3♦, 4♥, 5♠, 7♣']]) {
    log.push(entry(`${street}: [${cards}]`), action('villain', 'checks'), action('hero', 'checks'));
  }
  log.push(entry('-- ending hand #1 --'));
  const parsed = parsePokerNowHand(log, config);
  assert.equal(parsed.status, 'parsed');
  assert.equal(parsed.history.length, 8);
  assert.equal(parsed.terminal, true);
  assert.equal(parsed.ended, true);
  assert.deepEqual(parsed.board, ['2c', '3d', '4h', '5s', '7c']);
});

for (const [name, log, reason] of [
  ['missing blind', () => [...startLog().slice(0, 3), action('hero', 'folds')], 'missing_blinds'],
  ['wrong action order', () => [...startLog(), action('villain', 'raises to 50')], 'action_order_inconsistent'],
  ['impossible call amount', () => [...startLog(), action('hero', 'calls 50')], 'action_chips_inconsistent'],
  ['unexpected player action', () => [...startLog(), action('outsider', 'raises to 50')], 'players_changed'],
  ['board dealt before betting closes', () => [...startLog(), entry('Flop: [2♣, 3♦, 4♥]')], 'betting_round_incomplete'],
  ['premature hand end', () => [...startLog(), entry('-- ending hand #1 --')], 'hand_end_inconsistent'],
  ['action after fold', () => [...startLog(), action('hero', 'folds'), action('villain', 'checks')], 'action_after_hand_end'],
]) {
  test(`parser rejects ${name}`, () => assert.deepEqual(parsePokerNowHand(log(), config), {status: 'blocked', reason}));
}

test('normalizer derives legal observation and excludes time-bank reserve', () => {
  const snapshot = activeSnapshot();
  const result = normalizePokerNow(snapshot, config);
  assert.equal(result.status, 'observed');
  assert.equal(result.request.deadline_unix_ms - result.request.observed_unix_ms, 14900);
  assert.equal(result.request.observation.to_call, 10);
  assert.equal(result.request.observation.actor, 1);
  assert.deepEqual(result.request.observation.hole, ['As', 'Ah']);
  assert.equal('opponent_hole' in result.request.observation, false);
});

for (const [name, mutate, reason] of [
  ['disconnection', s => s.disconnected = true, 'table_disconnected'],
  ['lineup change', s => s.players[1].id = 'new-player', 'players_changed'],
  ['hero identity mismatch', s => {s.players[0].hero = false; s.players[1].hero = true;}, 'hero_changed'],
  ['blind change', s => s.variant = 'NLH ~ 20 / 40', 'blinds_changed'],
  ['multiple boards', s => s.board_count = 2, 'unsupported_table'],
  ['snapshot and log disagreement', s => s.pot = '31', 'snapshot_history_mismatch'],
  ['expired action reserve', s => s.players[0].normal_clock_percent = '4%', 'clock_reserve_exhausted'],
  ['unreadable clock', s => s.players[0].normal_clock_percent = 'auto', 'clock_unreadable'],
  ['unverified raises', s => s.buttons.push({classes: 'raise', enabled: true, text: 'RAISE'}), 'verify_raise_bounds'],
]) {
  test(`normalizer blocks ${name}`, () => {
    const snapshot = activeSnapshot(); mutate(snapshot);
    assert.equal(normalizePokerNow(snapshot, config).reason, reason);
  });
}

test('memory identity is stable under seat ordering and separates opponents and rooms', () => {
  const identity = pokerNowMemoryIdentity(config);
  assert.deepEqual(identity, pokerNowMemoryIdentity({...config, playerIds: [...config.playerIds].reverse()}));
  assert.notDeepEqual(identity, pokerNowMemoryIdentity({...config, playerIds: ['hero', 'other']}));
  assert.notDeepEqual(identity, pokerNowMemoryIdentity({...config, roomUrl: `${config.roomUrl}-2`}));
  assert.throws(() => pokerNowMemoryIdentity({...config, privatePlayMoney: false}), /confirm_private/);
  assert.throws(() => pokerNowMemoryIdentity({...config, playerIds: ['hero', 'hero']}), /pin_two_test_players/);
});

test('empty visible log is an error rather than evidence of no actions', async () => {
  const tab = idleTab(); tab.model.log = []; tab.model.log_status = 'empty';
  await assert.rejects(readPokerNowLog(tab), /hand_log_empty/);
});

test('turn cancellation occurs before any worker request or UI action', async () => {
  const tab = idleTab();
  const result = await runPrivatePokerNowTurn({tab, config, shouldStop: () => true,
    request: async () => assert.fail('worker must not be consulted')});
  assert.equal(result.status, 'blocked');
  assert.equal(result.action_sent, false);
  assert.equal(result.retry_allowed, false);
  assert.equal(tab.clicks.length, 0);
});

test('pending worker action cannot be blindly resubmitted', async () => {
  const tab = idleTab();
  const result = await runPrivatePokerNowTurn({tab, config, request: async () => ({status: 'worker_ready', pending: true})});
  assert.equal(result.reason, 'worker_not_ready');
  assert.equal(tab.clicks.length, 0);
});

test('a submitted fold requires the exact public player/action receipt', async () => {
  const fixture = foldTurn();
  const result = await runPrivatePokerNowTurn({...fixture, config});
  assert.equal(result.status, 'confirmed');
  assert.equal(fixture.tab.clicks.filter(name => name === 'f Fold').length, 1);
  const confirmation = fixture.requests.find(request => request.kind === 'confirm');
  assert.deepEqual(confirmation.history[0], {actor: 1, street: 'preflop',
    action: {kind: 'fold', amount: 0}, pot_before: 30, to_call_before: 10, paid: 0, board: []});
});

test('a different player receipt leaves the action uncertain and never retries the click', async () => {
  const fixture = foldTurn('villain');
  const result = await runPrivatePokerNowTurn({...fixture, config});
  assert.equal(result.status, 'uncertain');
  assert.equal(result.reason, 'action_not_confirmed_no_retry');
  assert.equal(result.retry_allowed, false);
  assert.equal(fixture.tab.clicks.filter(name => name === 'f Fold').length, 1);
  assert.equal(fixture.requests.some(request => request.kind === 'confirm'), false);
});

test('an expired send reserve blocks before clicking an otherwise legal action', async () => {
  const fixture = foldTurn();
  fixture.tab.model.captured_unix_ms -= 14500;
  const result = await runPrivatePokerNowTurn({...fixture, config});
  assert.equal(result.status, 'blocked');
  assert.equal(result.reason, 'deadline_reserve_exhausted');
  assert.equal(result.action_sent, false);
  assert.equal(fixture.tab.clicks.includes('f Fold'), false);
});

test('session persists an opponent blind-fold even when the hero never acted', async () => {
  const tab = idleTab(), worker = mockWorker();
  const session = await createPrivatePokerNowSession({tab, config, worker, maxHands: 1});
  tab.model.log = foldedLog('hand1', 1);
  const state = await session.step();
  assert.equal(state.reason, 'hand_limit');
  assert.equal(state.away, true);
  assert.equal(state.completed_hands, 1);
  assert.equal(state.learning.profile_hands, 1);
  assert.equal(state.learning_updates, 1);
  assert.equal(state.decisions, 0);
  assert.equal(worker.requests.find(r => r.kind === 'finish').hero_seat, 0);
  await session.step();
  assert.equal(worker.requests.filter(r => r.kind === 'finish').length, 1);
  assert.equal(worker.closed, true);
});

test('session stops on a substituted player before any learning update', async () => {
  const tab = idleTab(), worker = mockWorker();
  const session = await createPrivatePokerNowSession({tab, config, worker});
  tab.model.players[1].id = 'replacement';
  const state = await session.step();
  assert.equal(state.reason, 'players_changed');
  assert.equal(state.learning_updates, 0);
  assert.equal(worker.requests.some(r => r.kind === 'finish'), false);
  assert.equal(worker.closed, true);
});

test('session refuses worker memory for a different opponent', async () => {
  const worker = mockWorker({identity: {...pokerNowMemoryIdentity(config), opponent_id: 'other'}});
  await assert.rejects(createPrivatePokerNowSession({tab: idleTab(), config, worker}), /worker_learning_identity_mismatch/);
  assert.equal(worker.closed, true);
});

test('session refuses a silently disabled persistent worker', async () => {
  const worker = mockWorker({enabled: false});
  await assert.rejects(createPrivatePokerNowSession({tab: idleTab(), config, worker}), /worker_learning_identity_mismatch/);
  assert.equal(worker.closed, true);
});

test('session retries only two interrupted log reads then stops', async () => {
  const tab = idleTab(), worker = mockWorker();
  const session = await createPrivatePokerNowSession({tab, config, worker});
  tab.interruptLogReads(3);
  assert.equal((await session.step()).stopped, false);
  assert.equal((await session.step()).stopped, false);
  assert.equal((await session.step()).reason, 'hand_log_closed');
  assert.equal(worker.requests.filter(r => r.kind === 'finish').length, 0);
  assert.equal(worker.closed, true);
});

test('session stops and closes its worker when a finish receipt reports inconsistent counts', async () => {
  const tab = idleTab(), worker = mockWorker();
  const request = worker.request.bind(worker);
  worker.request = async payload => {
    const response = await request(payload);
    if (payload.kind === 'finish') response.learning.profile_hands += 1;
    return response;
  };
  const session = await createPrivatePokerNowSession({tab, config, worker});
  tab.model.log = foldedLog('hand1', 1);
  assert.equal((await session.step()).reason, 'learning_receipt_count_mismatch');
  assert.equal(session.state().learning_updates, 0);
  assert.equal(worker.closed, true);
});

test('actual JS-to-Python worker saves a completed response and reloads it once', {timeout: 30000}, async t => {
  const directory = await mkdtemp(join(tmpdir(), 'headsup-bridge-'));
  let worker;
  t.after(async () => { worker?.close(); await delay(100); await rm(directory, {recursive: true, force: true}); });
  worker = await startDecisionWorker({config, memoryDirectory: directory, seed: 91});
  const input = normalizePokerNow({...activeSnapshot(),
    raise_form: {value: '40', max: '2000'}}, config, {raiseMinimumVerified: true}).request;
  assert.ok(input);
  let decision = await worker.request(input);
  if (decision.status === 'waiting') {
    await delay(Math.ceil(decision.wait_ms) + 10);
    decision = await worker.request({kind: 'status', hand_id: input.hand_id, turn_index: 0});
  }
  assert.equal(decision.status, 'ready');
  assert.equal(decision.action.kind, 'raise_to');
  const amount = decision.action.amount;
  const history = [{actor: 1, street: 'preflop', action: decision.action,
    pot_before: 30, to_call_before: 10, paid: amount - 10, board: []}];
  assert.equal((await worker.request({kind: 'confirm', hand_id: input.hand_id, history})).status, 'confirmed');
  history.push({actor: 0, street: 'preflop', action: {kind: 'fold', amount: 0},
    pot_before: 20 + amount, to_call_before: amount - 20, paid: 0, board: []});
  const finish = {kind: 'finish', hand_id: input.hand_id, hero_seat: 1, history};
  const result = await worker.request(finish);
  assert.equal(result.status, 'finished');
  assert.equal(result.learning_updated, true);
  assert.equal(result.learning.total_responses, 1);
  assert.equal(result.learning.profile_hands, 1);
  const contexts = Object.values(result.learning.behavior.contexts);
  assert.equal(contexts.length, 1);
  assert.deepEqual(contexts[0].counts, {fold: 1, call: 0, raise: 0});
  assert.equal(contexts[0].ready, false);
  worker.close(); await delay(100);
  worker = await startDecisionWorker({config, memoryDirectory: directory, seed: 92});
  const restarted = await worker.request({kind: 'ping'});
  assert.equal(restarted.learning.profile_sha256, result.learning.profile_sha256);
  assert.equal(restarted.learning.profile_hands, 1);
  const duplicate = await worker.request(finish);
  assert.equal(duplicate.learning_updated, false);
  assert.equal(duplicate.learning.profile_hands, 1);
  assert.deepEqual(duplicate.learning.identity, pokerNowMemoryIdentity(config));
});

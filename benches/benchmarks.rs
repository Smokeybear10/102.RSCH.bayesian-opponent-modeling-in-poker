criterion::criterion_main!(benches);
criterion::criterion_group! {
    name = benches;
    config = criterion::Criterion::default()
        .without_plots()
        .noise_threshold(3.0)
        .significance_level(0.01)
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(1));
    targets =
        sampling_river_evaluation,
        sampling_river_equity,
        sampling_river_observation,
        converting_turn_isomorphism,
        exhausting_flop_observations,
        exhausting_flop_isomorphisms,
        collecting_turn_histogram,
        computing_optimal_transport_variation,
        computing_optimal_transport_heuristic,
        computing_optimal_transport_sinkhorns,
        solving_cfr_rps,
}

fn sampling_river_evaluation(c: &mut criterion::Criterion) {
    c.bench_function("evaluate a 7-card Hand", |b| {
        let hand = Hand::from(Observation::from(Street::Rive));
        b.iter(|| Strength::from(Evaluator::from(hand)))
    });
}

fn sampling_river_observation(c: &mut criterion::Criterion) {
    c.bench_function("collect a 7-card River Observation", |b| {
        b.iter(|| Observation::from(Street::Rive))
    });
}

fn sampling_river_equity(c: &mut criterion::Criterion) {
    let observation = Observation::from(Street::Rive);
    c.bench_function("calculate River equity", |b| {
        b.iter(|| observation.equity())
    });
}

fn exhausting_flop_observations(c: &mut criterion::Criterion) {
    c.bench_function("exhaust all Flop Observations", |b| {
        b.iter(|| ObservationIterator::from(Street::Flop).count())
    });
}

fn exhausting_flop_isomorphisms(c: &mut criterion::Criterion) {
    c.bench_function("exhaust all Flop Isomorphisms", |b| {
        b.iter(|| {
            ObservationIterator::from(Street::Flop)
                .filter(Isomorphism::is_canonical)
                .count()
        })
    });
}

fn converting_turn_isomorphism(c: &mut criterion::Criterion) {
    let observation = Observation::from(Street::Turn);
    c.bench_function("convert a Turn Observation to Isomorphism", |b| {
        b.iter(|| Isomorphism::from(observation))
    });
}

fn collecting_turn_histogram(c: &mut criterion::Criterion) {
    let observation = Observation::from(Street::Turn);
    c.bench_function("collect a Histogram from a Turn Observation", |b| {
        b.iter(|| Histogram::from(observation))
    });
}

fn computing_optimal_transport_variation(c: &mut criterion::Criterion) {
    let ref h1 = Histogram::from(Observation::from(Street::Turn));
    let ref h2 = Histogram::from(Observation::from(Street::Turn));
    c.bench_function("compute optimal transport (1-dimensional)", |b| {
        b.iter(|| Equity::variation(&h1, &h2))
    });
}

fn computing_optimal_transport_heuristic(c: &mut criterion::Criterion) {
    let (metric, h1, h2, _) = EMD::random().inner();
    c.bench_function("compute optimal transport (greedy)", |b| {
        b.iter(|| Heuristic::from((&h1, &h2, &metric)).minimize().cost())
    });
}

fn computing_optimal_transport_sinkhorns(c: &mut criterion::Criterion) {
    let (metric, h1, h2, _) = EMD::random().inner();
    c.bench_function("compute optimal transport (entropy regularized)", |b| {
        b.iter(|| Sinkhorn::from((&h1, &h2, &metric)).minimize().cost())
    });
}

fn solving_cfr_rps(c: &mut criterion::Criterion) {
    c.bench_function("cfr solve rock paper scissors (rps)", |b| {
        b.iter(|| RPS::default().solve());
    });
}

use r1ver::cards::evaluator::Evaluator;
use r1ver::cards::hand::Hand;
use r1ver::cards::isomorphism::Isomorphism;
use r1ver::cards::observation::Observation;
use r1ver::cards::observations::ObservationIterator;
use r1ver::cards::street::Street;
use r1ver::cards::strength::Strength;
use r1ver::clustering::emd::EMD;
use r1ver::clustering::equity::Equity;
use r1ver::clustering::heuristic::Heuristic;
use r1ver::clustering::histogram::Histogram;
use r1ver::clustering::sinkhorn::Sinkhorn;
use r1ver::mccfr::rps::RPS;
use r1ver::mccfr::traits::Blueprint;
use r1ver::transport::coupling::Coupling;
use r1ver::Arbitrary;

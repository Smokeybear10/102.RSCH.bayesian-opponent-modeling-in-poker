use super::checkpoint::Checkpoint;
use super::histogram::Histogram;
use super::lookup::Lookup;
use super::metric::Metric;
use super::pair::Pair;
use super::transitions::Decomp;
use crate::cards::isomorphism::Isomorphism;
use crate::cards::isomorphisms::IsomorphismIterator;
use crate::cards::street::Street;
use crate::gameplay::abstraction::Abstraction;
use crate::Energy;
use std::collections::BTreeMap;

type Neighbor = (usize, f32);

pub struct Layer {
    street: Street,
    metric: Metric,
    points: Vec<Histogram>, // positioned by Isomorphism
    kmeans: Vec<Histogram>, // positioned by K-means abstraction
}

impl Layer {
    /// reference to the all points up to isomorphism
    fn points(&self) -> &Vec<Histogram> /* N */ {
        &self.points
    }
    /// reference to the current kmeans centorid histograms
    fn kmeans(&self) -> &Vec<Histogram> /* K */ {
        &self.kmeans
    }

    /// all-in-one entry point for learning the kmeans abstraction and
    /// writing to disk in pgcopy
    pub fn learn() {
        use crate::save::disk::Disk;
        use std::sync::atomic::Ordering;
        Street::all()
            .into_iter()
            .rev()
            .filter(|&&s| Self::done(s))
            .for_each(|s| log::info!("{:<32}{:<16}{:<32}", "using kmeans layer", s, Self::name()));
        for &s in Street::all().into_iter().rev() {
            if Self::done(s) {
                continue;
            }
            let layer = Self::grow(s);
            if crate::INTERRUPTED.load(Ordering::Relaxed) {
                log::warn!(
                    "interrupted during {}; checkpoint retained, skipping final save",
                    s
                );
                return;
            }
            layer.save();
            Checkpoint::delete(s);
        }
    }

    /// primary clustering algorithm loop
    fn cluster(mut self) -> Self {
        use std::sync::atomic::Ordering;
        let t = self.street().t();
        let (mut start, centroids) = match Checkpoint::load(self.street()) {
            Some((i, hs)) if (i as usize) <= t && hs.len() == self.street().k() => {
                log::info!(
                    "{:<32}{:<32}resuming at iter {}/{}",
                    "loaded      checkpoint",
                    self.street(),
                    i,
                    t
                );
                (i as usize, hs)
            }
            _ => {
                log::info!("{:<32}{:<32}", "initialize  kmeans", self.street());
                let init = self.init();
                if self.street().t() > 0 {
                    Checkpoint::save(self.street(), 0, &init);
                }
                (0, init)
            }
        };
        self.kmeans = centroids;
        log::info!("{:<32}{:<32}", "clustering  kmeans", self.street());
        let progress = crate::progress(t);
        progress.inc(start as u64);
        while start < t {
            if crate::INTERRUPTED.load(Ordering::Relaxed) {
                log::warn!(
                    "graceful interrupt at iter {}/{} ({}); checkpoint preserved",
                    start,
                    t,
                    self.street()
                );
                progress.finish();
                println!();
                return self;
            }
            let ref mut next = self.next();
            let ref mut last = self.kmeans;
            std::mem::swap(next, last);
            start += 1;
            Checkpoint::save(self.street(), start as u32, &self.kmeans);
            progress.inc(1);
        }
        progress.finish();
        println!();
        // repair any empty centroids before metric calculation
        {
            use rand::rngs::SmallRng;
            use rand::seq::IndexedRandom;
            use rand::SeedableRng;
            let replacements: Vec<(usize, Histogram)> = self
                .kmeans
                .iter()
                .enumerate()
                .filter(|(_, c)| c.mass() == 0)
                .map(|(i, _)| {
                    let point = self
                        .points()
                        .choose(&mut SmallRng::seed_from_u64(i as u64))
                        .expect("non-empty points")
                        .clone();
                    (i, point)
                })
                .collect();
            for (i, point) in replacements {
                self.kmeans[i].absorb(&point);
            }
        }
        self
    }

    /// initializes the centroids for k-means clustering using the k-means++ algorithm
    /// 1. choose 1st centroid randomly from the dataset
    /// 2. choose nth centroid with probability proportional to squared distance of nearest neighbors
    /// 3. collect histograms and label with arbitrary (random) `Abstraction`s
    fn init(&self) -> Vec<Histogram> /* K */ {
        use rand::distr::weighted::WeightedIndex;
        use rand::distr::Distribution;
        use rand::rngs::SmallRng;
        use rand::SeedableRng;
        use rayon::iter::IntoParallelRefIterator;
        use rayon::iter::ParallelIterator;
        use std::hash::DefaultHasher;
        use std::hash::Hash;
        use std::hash::Hasher;
        // don't do any abstraction on preflop
        let k = self.street().k();
        let n = self.points().len();
        if self.street() == Street::Pref {
            assert!(n == k);
            return self.points().clone();
        }
        // deterministic pseudo-random clustering
        let ref mut hasher = DefaultHasher::default();
        self.street().hash(hasher);
        let ref mut rng = SmallRng::seed_from_u64(hasher.finish());
        // kmeans++ initialization
        let progress = crate::progress(k * n);
        let mut potentials = vec![1.; n];
        let mut histograms = Vec::new();
        while histograms.len() < k {
            let i = WeightedIndex::new(potentials.iter())
                .expect("valid weights array")
                .sample(rng);
            let x = self
                .points()
                .get(i)
                .expect("sharing index with outer layer");
            histograms.push(x.clone());
            potentials[i] = 0.;
            potentials = self
                .points()
                .par_iter()
                .map(|h| self.fast_emd(x, h))
                .map(|p| p * p)
                .inspect(|_| progress.inc(1))
                .collect::<Vec<Energy>>()
                .iter()
                .zip(potentials.iter())
                .map(|(d0, d1)| Energy::min(*d0, *d1))
                .collect::<Vec<Energy>>();
        }
        progress.finish();
        println!();
        histograms
    }

    /// calculates the next step of the kmeans iteration by
    /// determining K * N optimal transport calculations and
    /// taking the nearest neighbor
    fn next(&self) -> Vec<Histogram> /* K */ {
        use rayon::iter::IntoParallelRefIterator;
        use rayon::iter::ParallelIterator;
        let k = self.street().k();
        let mut loss = 0f32;
        let mut centroids = vec![Histogram::default(); k];
        // assign points to nearest neighbors
        for (point, (neighbor, distance)) in self
            .points()
            .par_iter()
            .map(|h| (h, self.neighborhood(h)))
            .collect::<Vec<_>>()
            .into_iter()
        {
            loss = loss + distance * distance;
            centroids
                .get_mut(neighbor)
                .expect("index from neighbor calculation")
                .absorb(point);
        }
        // reinitialize empty centroids — can occur when fast_emd shifts all points away
        {
            use rand::rngs::SmallRng;
            use rand::seq::IndexedRandom;
            use rand::SeedableRng;
            let replacements: Vec<(usize, Histogram)> = centroids
                .iter()
                .enumerate()
                .filter(|(_, c)| c.mass() == 0)
                .map(|(i, _)| {
                    let point = self
                        .points()
                        .choose(&mut SmallRng::seed_from_u64(i as u64))
                        .expect("non-empty points")
                        .clone();
                    (i, point)
                })
                .collect();
            for (i, point) in replacements {
                centroids[i].absorb(&point);
            }
        }
        log::debug!(
            "{:<32}{:<32}",
            "abstraction cluster RMS error",
            (loss / self.points().len() as f32).sqrt()
        );
        centroids
    }

    /// in ObsIterator order, get a mapping of
    /// Isomorphism -> Abstraction
    fn lookup(&self) -> Lookup {
        log::info!("{:<32}{:<32}", "calculating lookup", self.street());
        use crate::save::disk::Disk;
        use rayon::iter::IntoParallelRefIterator;
        use rayon::iter::ParallelIterator;
        let street = self.street();
        match street {
            Street::Pref | Street::Rive => Lookup::grow(street),
            Street::Flop | Street::Turn => self
                .points()
                .par_iter()
                .map(|h| self.neighborhood(h))
                .collect::<Vec<Neighbor>>()
                .into_iter()
                .map(|(k, _)| self.abstraction(k))
                .zip(IsomorphismIterator::from(street))
                .map(|(abs, iso)| (iso, abs))
                .collect::<BTreeMap<Isomorphism, Abstraction>>()
                .into(),
        }
    }

    /// fast assignment distance: expected distance O(Sp × Sc), no Sinkhorn iterations
    fn fast_emd(&self, x: &Histogram, y: &Histogram) -> Energy {
        self.metric.expected(x, y)
    }
    /// because we have fixed-order Abstractions that are determined by
    /// street and K-index, we should encapsulate the self.street depenency
    fn abstraction(&self, i: usize) -> Abstraction {
        Abstraction::from((self.street(), i))
    }
    /// calculates nearest neighbor and separation distance for a Histogram
    fn neighborhood(&self, x: &Histogram) -> Neighbor {
        self.kmeans()
            .iter()
            .enumerate()
            .map(|(k, h)| (k, self.fast_emd(x, h)))
            .min_by(|(_, dx), (_, dy)| dx.partial_cmp(dy).unwrap())
            .expect("find nearest neighbor")
            .into()
    }

    /// reference to current street
    fn street(&self) -> Street {
        self.street
    }
    /// take outer triangular product of current learned kmeans
    /// Histograms, using whatever is stored as the future metric
    fn metric(&self) -> Metric {
        log::info!("{:<32}{:<32}", "calculating metric", self.street());
        let mut metric = BTreeMap::new();
        for (i, x) in self.kmeans.iter().enumerate() {
            for (j, y) in self.kmeans.iter().enumerate() {
                if i > j {
                    let ref a = self.abstraction(i);
                    let ref b = self.abstraction(j);
                    let index = Pair::from((a, b));
                    let distance = self.metric.emd(x, y) + self.metric.emd(y, x);
                    let distance = distance / 2.;
                    metric.insert(index, distance);
                }
            }
        }
        Metric::from(metric)
    }
    /// in AbsIterator order, get a mapping of
    /// Abstraction -> Histogram
    /// end-of-recurse call
    fn decomp(&self) -> Decomp {
        log::info!("{:<32}{:<32}", "calculating transitions", self.street());
        self.kmeans()
            .iter()
            .cloned()
            .enumerate()
            .map(|(k, centroid)| (self.abstraction(k), centroid))
            .collect::<BTreeMap<Abstraction, Histogram>>()
            .into()
    }
}

impl crate::save::disk::Disk for Layer {
    fn name() -> String {
        format!(
            "{:<16}{:<16}{:<16}",
            Lookup::name(),
            Decomp::name(),
            Metric::name()
        )
    }
    fn done(street: Street) -> bool {
        Lookup::done(street) && Decomp::done(street) && Metric::done(street)
    }
    fn save(&self) {
        self.metric().save();
        self.lookup().save();
        self.decomp().save();
    }
    fn grow(street: Street) -> Self {
        let layer = match street {
            Street::Rive => Self {
                street,
                kmeans: Vec::default(),
                points: Vec::default(),
                metric: Metric::default(),
            },
            _ => Self {
                street,
                kmeans: Vec::default(),
                points: Lookup::load(street.next()).projections(),
                metric: Metric::load(street.next()),
            },
        };
        layer.cluster()
    }
    fn load(_: Street) -> Self {
        unimplemented!()
    }
}

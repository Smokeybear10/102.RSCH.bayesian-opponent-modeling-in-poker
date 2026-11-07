pub mod cards;
pub use cards::*;
pub mod gameplay;
pub use gameplay::*;
pub mod transport;
pub use transport::*;
#[cfg(feature = "wasm")]
pub mod wasm;
#[cfg(feature = "wasm")]
pub use wasm::*;

#[cfg(feature = "native")]
pub mod analysis;
#[cfg(feature = "native")]
pub mod clustering;
#[cfg(feature = "native")]
pub mod mccfr;
#[cfg(feature = "native")]
pub mod players;
#[cfg(feature = "native")]
pub mod save;

/// dimensional analysis types
type Chips = i16;
type Equity = f32;
#[cfg(feature = "native")]
type Energy = f32;
#[cfg(feature = "native")]
type Entropy = f32;
type Utility = f32;
type Probability = f32;

#[cfg(feature = "native")]
static INTERRUPTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

// game tree parameters
const N: usize = 2;
const STACK: Chips = 100;
const B_BLIND: Chips = 2;
const S_BLIND: Chips = 1;
#[cfg(feature = "native")]
const MAX_RAISE_REPEATS: usize = 3;
const MAX_DEPTH_SUBGAME: usize = 16;

/// sinkhorn optimal transport parameters
#[cfg(feature = "native")]
const SINKHORN_TEMPERATURE: Entropy = 0.025;
#[cfg(feature = "native")]
const SINKHORN_ITERATIONS: usize = 128;
#[cfg(feature = "native")]
const SINKHORN_TOLERANCE: Energy = 0.001;

// kmeans clustering parameters
#[cfg(not(feature = "demo"))]
const KMEANS_FLOP_TRAINING_ITERATIONS: usize = 20;
#[cfg(feature = "demo")]
const KMEANS_FLOP_TRAINING_ITERATIONS: usize = 3;

#[cfg(not(feature = "demo"))]
const KMEANS_TURN_TRAINING_ITERATIONS: usize = 24;
#[cfg(feature = "demo")]
const KMEANS_TURN_TRAINING_ITERATIONS: usize = 3;

// Metric::street() infers which street a metric belongs to from its pair count, choose_2(k),
// so the three cluster counts below must stay distinct — equal counts collide and one street's
// metric overwrites another's.
#[cfg(not(feature = "demo"))]
const KMEANS_FLOP_CLUSTER_COUNT: usize = 128;
#[cfg(feature = "demo")]
const KMEANS_FLOP_CLUSTER_COUNT: usize = 8;

#[cfg(not(feature = "demo"))]
const KMEANS_TURN_CLUSTER_COUNT: usize = 144;
#[cfg(feature = "demo")]
const KMEANS_TURN_CLUSTER_COUNT: usize = 12;

#[cfg(not(feature = "demo"))]
const KMEANS_EQTY_CLUSTER_COUNT: usize = 101;
#[cfg(feature = "demo")]
const KMEANS_EQTY_CLUSTER_COUNT: usize = 16;

/// rps mccfr parameteres
#[cfg(feature = "native")]
const ASYMMETRIC_UTILITY: f32 = 2.0;
#[cfg(feature = "native")]
const CFR_BATCH_SIZE_RPS: usize = 1;
#[cfg(feature = "native")]
const CFR_TREE_COUNT_RPS: usize = 8192;

// nlhe mccfr parameters
#[cfg(feature = "native")]
const CFR_BATCH_SIZE_NLHE: usize = 128;
#[cfg(all(feature = "native", not(feature = "demo")))]
const CFR_TREE_COUNT_NLHE: usize = 0x10000000; // 268M trees
#[cfg(feature = "demo")]
const CFR_TREE_COUNT_NLHE: usize = 0x10000; // 65K trees

/// profile average sampling parameters
#[cfg(feature = "native")]
const SAMPLING_THRESHOLD: Entropy = 1.0;
#[cfg(feature = "native")]
const SAMPLING_ACTIVATION: Energy = 0.0;
#[cfg(feature = "native")]
const SAMPLING_EXPLORATION: Probability = 0.01;

// regret matching parameters
#[cfg(feature = "native")]
const POLICY_MIN: Probability = Probability::MIN_POSITIVE;
#[cfg(feature = "native")]
const REGRET_MIN: Utility = -3e5;

/// trait for random generation, mainly (strictly?) for testing
pub trait Arbitrary {
    fn random() -> Self;
}

/// progress bar
#[cfg(feature = "native")]
pub fn progress(n: usize) -> indicatif::ProgressBar {
    let tick = std::time::Duration::from_secs(60);
    let style = "{spinner:.cyan} {elapsed} ~ {percent:>3}% {wide_bar:.cyan}";
    let style = indicatif::ProgressStyle::with_template(style).unwrap();
    let progress = indicatif::ProgressBar::new(n as u64);
    progress.set_style(style);
    progress.enable_steady_tick(tick);
    progress
}

/// initialize logging and setup graceful interrupt listener
#[cfg(feature = "native")]
pub fn log() {
    // Hosted containers often have a read-only working directory, and file logging is a
    // convenience — losing it shouldn't stop the process from starting.
    let logfile = std::fs::create_dir_all("logs").is_ok();
    let config = simplelog::ConfigBuilder::new()
        .set_location_level(log::LevelFilter::Off)
        .set_target_level(log::LevelFilter::Off)
        .set_thread_level(log::LevelFilter::Off)
        .build();
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time moves slow")
        .as_secs();
    let term = simplelog::TermLogger::new(
        log::LevelFilter::Info,
        config.clone(),
        simplelog::TerminalMode::Mixed,
        simplelog::ColorChoice::Auto,
    );
    let mut loggers: Vec<Box<dyn simplelog::SharedLogger>> = vec![term];
    if let Some(file) = logfile
        .then(|| std::fs::File::create(format!("logs/{}.log", time)).ok())
        .flatten()
    {
        loggers.push(simplelog::WriteLogger::new(
            log::LevelFilter::Debug,
            config.clone(),
            file,
        ));
    }
    simplelog::CombinedLogger::init(loggers).expect("initialize logger");
}

/// get a database connection and return the client
#[cfg(feature = "native")]
pub async fn db() -> std::sync::Arc<tokio_postgres::Client> {
    log::info!("connecting to database");
    let tls = tokio_postgres::tls::NoTls;
    let ref url = std::env::var("DB_URL").expect("DB_URL must be set");
    let (client, connection) = tokio_postgres::connect(url, tls)
        .await
        .expect("database connection failed");
    tokio::spawn(connection);
    std::sync::Arc::new(client)
}

#[cfg(feature = "native")]
pub fn kys() {
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.unwrap();
        println!();
        if INTERRUPTED.load(std::sync::atomic::Ordering::Relaxed) {
            log::warn!("second interrupt received, exiting immediately");
            std::process::exit(0);
        }
        log::warn!("interrupt received, finishing current batch... (press again to force)");
        INTERRUPTED.store(true, std::sync::atomic::Ordering::Relaxed);
        tokio::signal::ctrl_c().await.unwrap();
        println!();
        log::warn!("second interrupt received, exiting immediately");
        std::process::exit(0);
    });
}

#[cfg(feature = "native")]
pub fn brb() {
    std::thread::spawn(|| {
        let ref mut buffer = String::new();
        loop {
            buffer.clear();
            if let Ok(_) = std::io::stdin().read_line(buffer) {
                if buffer.trim().to_uppercase() == "Q" {
                    log::warn!("graceful interrupt requested, finishing current batch...");
                    INTERRUPTED.store(true, std::sync::atomic::Ordering::Relaxed);
                    break;
                }
            }
        }
    });
}

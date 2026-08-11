//! Bayesian Opponent Modeling in Poker - A No-Limit Texas Hold'em Poker AI Implementation

use bayesian_opponent_modeling_in_poker::*;

#[tokio::main]
async fn main() {
    crate::log();
    crate::kys();
    crate::brb();

    #[cfg(feature = "cluster")]
    crate::clustering::Layer::learn();

    #[cfg(feature = "trainer")]
    crate::mccfr::NLHE::solve();

    #[cfg(feature = "publish")]
    crate::save::Writer::publish().await.unwrap();

    #[cfg(feature = "analyze")]
    crate::analysis::Server::run().await.unwrap();
}

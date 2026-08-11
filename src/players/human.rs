use crate::gameplay::action::Action;
use crate::gameplay::game::Game;

#[derive(Debug)]
pub struct Human;

impl Human {
    pub fn decide(game: &Game) -> Action {
        use rand::prelude::IndexedRandom;
        let ref mut rng = rand::rng();
        game.legal()
            .choose(rng)
            .copied()
            .expect("decision node has options")
    }
}

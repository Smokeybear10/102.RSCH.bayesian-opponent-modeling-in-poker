use super::action::Action;
use super::game::Game;
use super::path::Path;
use super::turn::Turn;
use crate::cards::hand::Hand;
use crate::cards::hole::Hole;
use crate::cards::isomorphism::Isomorphism;
use crate::cards::observation::Observation;

/// a complete representation of perfect recall [Game] history
/// from the perspective of the hero [Turn].
///
/// note that this struct implicitly assumes:
/// - default stacks
/// - default dealer position
/// - blinds, draws, and player actions are all included in action path
#[derive(Debug, Clone)]
pub struct Recall {
    seen: Observation, // could be replaced by Hole + Board + BetHistory(Vec<Action>)
    past: Vec<Action>,
}

impl From<(Turn, Observation, Vec<Action>)> for Recall {
    fn from((_, seen, past): (Turn, Observation, Vec<Action>)) -> Self {
        Self { seen, past }
    }
}

impl Recall {
    pub fn root(&self) -> Game {
        Game::root().wipe(Hole::from(self.seen))
    }
    pub fn head(&self) -> Game {
        self.past.iter().fold(self.root(), |g, a| g.apply(*a))
    }
    /// the blueprint is keyed on this Path, so it has to be packed the same way the
    /// trainer packs it. Encoder::info builds the history by walking a Node *upward*
    /// toward the root, so the most recent edge lands in the lowest nibble — and the
    /// MAX_DEPTH_SUBGAME truncation in Path's FromIterator then drops the oldest edges,
    /// which is the point of a depth-limited subgame. Collecting our forward-ordered
    /// history directly would invert every nibble and miss the row entirely.
    #[rustfmt::skip]
    pub fn path(&self) -> Path {
        assert!(self.consistent());
        self.edges().into_iter().rev().collect::<Path>()
    }

    /// the history as edges, oldest first.
    ///
    /// each raise is snapped to the sizings the tree actually offered at that point, which
    /// depends on the street and on how many aggressive actions preceded it — the same
    /// inputs Encoder::raises uses when it builds the branches. Snapping against the full
    /// preflop grid instead would map a quarter-pot flop bet onto an Edge the trainer never
    /// generated, and the lookup would miss.
    fn edges(&self) -> Vec<crate::gameplay::edge::Edge> {
        let mut game = self.root();
        let mut edges = Vec::with_capacity(self.past.len());
        for action in self.past.iter() {
            let grid = crate::mccfr::nlhe::encoder::Encoder::raises(&game, Self::aggro(&edges));
            edges.push(game.edgify_within(*action, &grid));
            game = game.apply(*action);
        }
        edges
    }

    /// how many aggressive actions precede the state being keyed.
    ///
    /// this feeds Encoder::choices, which decides how many raise sizings are on offer, so
    /// it has to match what the trainer counted or the `future` key comes out a different
    /// width. Encoder::info counts from the head's *parent*, over edges that haven't yet
    /// crossed a chance node — but because the arriving edge is dropped first, a Draw at
    /// that position is dropped too, and the count keeps running into the previous street.
    /// Odd as that looks, it is the definition the blueprint was written with.
    ///
    /// Path::raises() can't be reused here: it takes from the low nibble, which after the
    /// reversal in path() is the most recent edge, and it can't skip the arriving edge.
    pub fn raises(&self) -> usize {
        Self::aggro(&self.edges())
    }

    /// the trainer's count, given the edges leading up to a state (oldest first).
    /// shared by edges() and raises() so the two can't drift apart.
    fn aggro(edges: &[crate::gameplay::edge::Edge]) -> usize {
        edges
            .iter()
            .rev()
            .skip(1)
            .take_while(|e| e.is_choice())
            .filter(|e| e.is_aggro())
            .count()
    }
    pub fn isomorphism(&self) -> Isomorphism {
        Isomorphism::from(self.seen)
    }

    pub fn consistent(&self) -> bool {
        self.seen.public().clone()
            == self
                .past
                .iter()
                .filter_map(|a| a.hand())
                .fold(Hand::empty(), Hand::add)
    }

    /// whether every action in the past is legal from the state preceding it.
    /// Game::act asserts legality, so this must hold before calling head() or
    /// path() on a Recall built from untrusted input.
    pub fn legal(&self) -> bool {
        let mut game = self.root();
        for action in self.past.iter() {
            if !game.is_allowed(action) {
                return false;
            }
            game = game.apply(*action);
        }
        true
    }
}

/// The blueprint is a key-value store keyed on (past, present, future), so a Recall that
/// derives any one of those differently from the trainer produces a lookup miss that looks
/// exactly like an untrained game state. These pin the two halves of that contract that
/// aren't otherwise observable: nibble order, and which raise grid a bet snaps to.
#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;
    use crate::cards::observation::Observation;
    use crate::gameplay::edge::Edge;
    use crate::mccfr::nlhe::encoder::Encoder;

    fn recall(seen: &str, past: &[&str]) -> Recall {
        let seen = Observation::try_from(seen).expect("parse observation");
        let past = past
            .iter()
            .map(|a| Action::try_from(*a).expect("parse action"))
            .collect::<Vec<_>>();
        Recall::from((Turn::Choice(0), seen, past))
    }

    /// Encoder::info walks a Node upward toward the root, so the most recent edge occupies
    /// the lowest nibble. A forward-ordered Path reverses every nibble and never matches.
    #[test]
    fn path_is_packed_most_recent_first() {
        let recall = recall("AsKd~7h8c9s", &["RAISE 6", "CALL 5", "DEAL 7h8c9s"]);
        let edges = Vec::<Edge>::from(recall.path());
        assert_eq!(edges.first(), Some(&Edge::Draw), "newest edge in nibble 0");
        assert!(
            matches!(edges.last(), Some(Edge::Raise(_))),
            "oldest edge in the high nibble, got {:?}",
            edges.last()
        );
    }

    /// raises() sizes the `future` key, so it has to reproduce the trainer's count exactly
    /// — including the quirk that dropping the arriving edge can drop a Draw with it, which
    /// lets the count run back into the previous street. Matching the blueprint matters more
    /// than the count being what you'd design from scratch.
    #[test]
    fn raises_reproduces_the_trainer_count() {
        let opened = recall("AsKd~", &["RAISE 6"]);
        assert!(opened.legal());
        assert_eq!(opened.raises(), 0, "excludes the arriving edge");

        let reraised = recall("AsKd~", &["RAISE 6", "RAISE 12"]);
        assert!(reraised.legal());
        assert_eq!(reraised.raises(), 1);

        // the Draw sits where the arriving edge is dropped, so take_while never sees it and
        // both preflop raises still count — exactly what Encoder::info does
        let crossed = recall(
            "AsKd~7h8c9s",
            &["RAISE 6", "RAISE 12", "CALL 7", "DEAL 7h8c9s"],
        );
        assert!(crossed.legal());
        assert_eq!(crossed.raises(), 2);

        // once a choice edge follows the Draw, take_while stops at it and the count is
        // confined to the new street
        let acted = recall(
            "AsKd~7h8c9s",
            &["RAISE 6", "RAISE 12", "CALL 7", "DEAL 7h8c9s", "CHECK"],
        );
        assert!(acted.legal());
        assert_eq!(acted.raises(), 0, "the Draw now blocks the take_while");
    }

    /// A bet is snapped to the sizings the tree actually branched to. Only preflop has all
    /// ten; the flop has five, so a quarter-pot flop bet must land on a real flop sizing.
    #[test]
    fn raises_snap_to_the_street_grid() {
        let recall = recall("AsKd~7h8c9s", &["RAISE 6", "CALL 5", "DEAL 7h8c9s", "RAISE 4"]);
        let edge = Vec::<Edge>::from(recall.path())
            .into_iter()
            .next()
            .expect("non-empty path");
        let offered = Encoder::raises(&recall.root(), 0);
        let flop = Encoder::raises(&recall.head(), 0);
        assert!(
            matches!(edge, Edge::Raise(odds) if flop.contains(&odds)),
            "{:?} is not a flop sizing ({:?})",
            edge,
            flop
        );
        assert!(offered.len() > flop.len(), "preflop grid should be wider");
    }

    /// At the root the API and the trainer must agree by construction: Encoder::seed uses
    /// depth 0 on Game::root, so an empty history has to produce exactly that.
    #[test]
    fn root_future_matches_the_trainer_seed() {
        let recall = recall("AsKd~", &[]);
        assert_eq!(recall.raises(), 0);
        assert_eq!(
            Encoder::choices(&recall.head(), recall.raises()),
            Encoder::choices(&recall.root(), 0),
        );
    }
}

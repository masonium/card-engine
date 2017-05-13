/// modules for storing player and opponent model
use std::collections::HashMap;
use cards::{BasicCard, Suit, print_card_map};
use std::fmt;
use std::cmp;

/// A `CardState` is an internal tracker for the likelihood of a
/// specific card.
#[derive(Debug, Clone, Copy, PartialEq)]
enum CardState {
    /// The card has been played, or the player otherwise definitely
    /// doesn't have the card.
    Played,

    /// The player was proven not to have this card right now.
    Void,

    /// The player has the card with the given likelihood.
    Prob(f32),

    /// The player definitely has this card.
    Owns
}

impl CardState {
    // Return the probability that the player owns a card in this
    // state.
    pub fn p(&self) -> f32 {
        match self {
            &CardState::Played => 0.0,
            &CardState::Void => 0.0,
            &CardState::Owns => 1.0,
            &CardState::Prob(ref p) => *p
        }
    }

    /// Return a zero-probability state if the current state is
    /// `Void`. Otherwise, return the same state.
    pub fn void_to_zero(&mut self) -> Self {
        if let CardState::Void = *self  {
            CardState::Prob(0.0)
        } else {
            *self
        }
    }

    /// Return true iff the state is the `Prob` enum
    pub fn is_prob(&self) -> bool {
        match self {
            &CardState::Prob(_) => true,
            _ => false
        }
    }
}

/// A `PlayerBelief` object represents the best estimate of the
/// probabilities of holding a specific card.
pub struct PlayerBelief {
    probs: HashMap<BasicCard, CardState>,
}

impl fmt::Display for CardState {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use self::CardState::*;
        let w = cmp::max(fmt.width().unwrap_or(5), 4);

        match *self {
            Played => write!(fmt, "{:1$}", "----", w),
            Void => write!(fmt, "{:1$}", "VOID", w),
            Owns => write!(fmt, "{:1$}", "MINE", w),
            Prob(p) => write!(fmt, "{:.1$}", p, w-2)
        }
    }
}

impl PlayerBelief {
    pub fn new() -> PlayerBelief {
        let mut probs = HashMap::new();
        //let mut priors = HashMap::new();
        for card in BasicCard::all() {
            probs.insert(card.clone(), CardState::Prob(0.0));
            //priors.insert(card, 0.0);
        }

        PlayerBelief { probs: probs }
    }

    /// print probabilities
    pub fn print_probabilities(&self) {
        print_card_map(&self.probs);
    }

    /// Return the probability that the player has the card.
    pub fn has_card(&mut self, card: &BasicCard) -> f32 {
        self.probs.get(card).expect("All basic cards should be in probability map.").p()
    }

    /// Increase the probability of each card in probability status,
    /// such that the total probability increase is ec.
    fn distribute_uniformly(&mut self, ec: f32) {
        let nc = self.num_candidates();
        for  v in self.probs.values_mut() {
            if let &mut CardState::Prob(ref mut p) = v {
                *p += ec / nc;
            }
        }
    }

    /// Transfer the probability from cards satisfying the predicate
    /// to cards that don't.
    fn transfer_probability_to<F: Fn(BasicCard) -> bool>(&mut self, pred: F) {
        let (p_dist, count) = self.probs.iter()
            .filter_map(|(k, v)| if !pred(*k) && v.is_prob() { Some(v.p()) } else { None } )
            .fold((0.0, 0.0), |state, m| (state.0 + m, state.1 + 1.0));

        let nc = self.num_candidates();
        let p_inc = p_dist / (nc - count) ;

        for (card, p) in self.probs.iter_mut() {
            if let &mut CardState::Prob(ref mut pp) = p {
                if pred(*card) {
                    *pp += p_inc;
                } else {
                    *pp = 0.0;
                }
            }
        }
    }

    // Return the number of cards that are in a probability state.
    fn num_candidates(&self) -> f32 {
        self.probs.values().filter(|v| v.is_prob()).count() as f32
    }

    // mark all cards as non-void
    fn remove_voids(&mut self) {
        for p in self.probs.values_mut() {
            *p = p.void_to_zero();
        }
    }

    // Draw n random cards, after marking cards as non-void.
    pub fn random_cards_drawn(&mut self, n: usize) {
        self.remove_voids();
        self.distribute_uniformly(n as f32);
    }

    /// establish that the suit is empty
    pub fn empty_suit(&mut self, suit: Suit) {
        self.transfer_probability_to(|c| c.suit != suit);
        for (k, v) in self.probs.iter_mut() {
            if k.suit == suit {
                if let &mut CardState::Prob(_) = v {
                    *v = CardState::Void;
                }
            }
        }
    }

    /// Show that a particular card has been picked up.
    pub fn card_drawn(&mut self, card: &BasicCard) {
        // distribute the probability among the remaining cards
        self.transfer_probability_to(|c| c != *card);

        // Set the card to be owned
        self.probs.insert(*card, CardState::Owns);
    }

    /// Mark that the card has been played by this player.
    pub fn card_played(&mut self, card: &BasicCard) {
        self.transfer_probability_to(|c| c != *card);

        // mark that the card has been played
        self.probs.insert(*card, CardState::Played);

        // show that the total number of cards has been reduced
        self.distribute_uniformly(-1.0);
    }

    /// Mark that the card was played by another player.
    pub fn card_seen(&mut self, card: &BasicCard) {
        self.transfer_probability_to(|c| c != *card);
        self.probs.insert(*card, CardState::Played);
    }
}

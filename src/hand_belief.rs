/// modules for storing player and opponent model
use std::collections::HashMap;
use cards::{BasicCard, Suit, Rank, print_card_map, NUM_BASIC_CARDS};
use std::fmt;
use std::cmp;
use ndarray::prelude::*;
use ndarray::DataMut;

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

/// A `HandBelief` object represents the best estimate of the
/// probabilities of holding a specific card.
///
/// This struct assumes that all actions affecting the hand (picking
/// up, discarding, playing cards) are public, even if the precise
/// cards aren't. For instance, this module would be invalid if the
/// player could pick up a secret number of cards.
///
/// `HandBelief` is game-agnostic, and does not generally try to
/// model any game-specific knowledge. Estimates are generally
/// max-entropy in that sense.
pub struct HandBelief {
    probs: HashMap<BasicCard, CardState>,
}

impl HandBelief {
    pub fn new() -> HandBelief {
        let mut probs = HashMap::new();
        for card in BasicCard::all() {
            probs.insert(card.clone(), CardState::Void);
        }

        HandBelief { probs: probs }
    }

    /// Reset the entire hand to void.
    pub fn clear(&mut self) {
        for v in self.probs.values_mut() {
            *v = CardState::Void;
        }
    }

    /// print probabilities
    pub fn print_probabilities(&self) {
        print_card_map(&self.probs);
    }

    /// Return the probability that the player has the card.
    pub fn has_card(&mut self, card: &BasicCard) -> f32 {
        self.probs.get(card).expect("All basic cards should be in probability map.").p()
    }

    /// Return the total number of cards held by the player
    pub fn num_cards(&self) -> f32 {
        self.probs.values().map(|v| v.p()).sum()
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

    /// Write the probabilities onto a vector in the given suit order.
    pub fn onto_vector<T: DataMut<Elem=f32>>(&self, vec: &mut ArrayBase<T, Ix1>, suit_order: &[Suit]) {
        assert_eq!(vec.dim(), NUM_BASIC_CARDS);

        for ((suit, rank), v) in izip!(iproduct!(suit_order, Rank::iterator()), vec.iter_mut()) {
            let card = BasicCard{suit: *suit, rank: *rank};
            *v = self.probs.get(&card).map(|v| v.p()).unwrap_or(0.0);
        }
    }
}

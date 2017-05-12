/// modules for storing player and opponent model
use std::collections::HashMap;
use cards::{BasicCard, Rank, Suit};

///
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CardState {
    /// The card has been played, or the player otherwise definitely
    /// doesn't have the card.
    Played,

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
            &CardState::Owns => 1.0,
            &CardState::Prob(ref p) => *p
        }
    }

    pub fn is_prob(&self) -> bool {
        match self {
            &CardState::Prob(_) => true,
            _ => false
        }
    }

    pub fn is_played(&self) -> bool {
        match self {
            &CardState::Played => true,
            _ => false
        }
    }
}

pub struct PlayerBelief {
    probs: HashMap<BasicCard, CardState>,

    //relative_priors: HashMap<BasicCard, f32>,

    // Total number of non-determined cards
    //num_cards: usize,
}

fn print_card_map(map: &HashMap<BasicCard, CardState>) {
    use self::CardState::*;
    for rank in Rank::iterator() {
        print!("| ");
        for suit in Suit::iterator() {
            let bc = BasicCard { rank: *rank, suit: *suit };
            print!("{}{}: ", rank, suit);
            match *map.get(&bc).unwrap() {
                Played => print!("----"),
                Owns => print!("MINE"),
                Prob(p) => print!("{:.2}", p)
            }
            print!(" | ");
        }
        println!("");
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
    pub fn dump_probabilities(&self) {
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

    /// Transfer the probability from cards satisfying the predicated
    /// to cards that don't.
    fn transfer_probability<F: Fn(BasicCard) -> bool>(&mut self, _pred: F) {
        unimplemented!();
    }

    // Return the number of cards that are in a probability state.
    fn num_candidates(&self) -> f32 {
        self.probs.values().filter(|v| v.is_prob()).count() as f32
    }

    pub fn random_cards_drawn(&mut self, n: usize) {
        self.distribute_uniformly(n as f32);
    }

    /// establish that the suit is empty
    pub fn empty_suit(&mut self, suit: Suit) {
        self.transfer_probability(|c| c.suit == suit);
    }

    /// Show that a particular card has been picked up.
    pub fn card_drawn(&mut self, card: &BasicCard) {
        // distribute the probability among the remaining cards
        self.transfer_probability(|c| c != *card);

        // Set the card to be owned
        self.probs.insert(*card, CardState::Owns);
    }

    /// Mark that the card has been played by this player.
    pub fn card_played(&mut self, card: &BasicCard) {
        self.transfer_probability(|c| c != *card);

        // mark that the card has been played
        self.probs.insert(*card, CardState::Played);

        // show that the total number of cards has been reduced
        self.distribute_uniformly(-1.0);
    }

    /// Mark that card was played by another player.
    pub fn card_seen(&mut self, card: &BasicCard) {
        self.transfer_probability(|c| c != *card);
        self.probs.insert(*card, CardState::Played);
    }
}
